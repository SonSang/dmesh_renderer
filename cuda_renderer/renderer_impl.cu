#include "renderer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace CudaRenderer {
	// Helper function to find the next-highest bit of the MSB
	// on the CPU.
	uint32_t getHigherMsb(uint32_t n)
	{
		uint32_t msb = sizeof(n) * 4;
		uint32_t step = msb;
		while (step > 1)
		{
			step /= 2;
			if (n >> msb)
				msb += step;
			else
				msb -= step;
		}
		if (n >> msb)
			msb++;
		return msb;
	}

	// Generates one key/value pair for all face / tile overlaps. 
	// Run once per batch-wise face (1:N mapping).
	__global__ void duplicateWithKeys(
		int B,
		int P,
		int F,
		const int* faces,
		const float2* verts_image,
		const float* depths,
		const uint32_t* offsets,
		uint64_t* face_keys_unsorted,
		uint32_t* face_values_unsorted,
		dim3 grid)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= B * F)
			return;

		int batch_id = idx / F;
		int face_id = idx % F;

		// Find this face's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

		int face_vert_id_0 = faces[3 * face_id + 0];
		int face_vert_id_1 = faces[3 * face_id + 1];
		int face_vert_id_2 = faces[3 * face_id + 2];
		
		int b_face_vert_id_0 = batch_id * P + face_vert_id_0;
		int b_face_vert_id_1 = batch_id * P + face_vert_id_1;
		int b_face_vert_id_2 = batch_id * P + face_vert_id_2;

		const float2* p0 = verts_image + b_face_vert_id_0;
		const float2* p1 = verts_image + b_face_vert_id_1;
		const float2* p2 = verts_image + b_face_vert_id_2;
		uint2 rect_min, rect_max;
		getRectFromTri(*p0, *p1, *p2, rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the face. Sorting the values 
		// with this key yields face IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		int grid_size = grid.x * grid.y;
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key = key + (grid_size * batch_id);		// Add batch ID to the key
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				face_keys_unsorted[off] = key;
				face_values_unsorted[off] = face_id;
				off++;
			}
		}
	}

	// Check keys to see if it is at the start/end of one tile's range in 
	// the full sorted list. If yes, write start/end of this tile. 
	// Run once per instanced (duplicated) face ID.
	__global__ void identifyTileRanges(int L, uint64_t* face_list_keys, uint2* ranges)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= L)
			return;

		// Read tile ID from key. Update start/end of tile range if at limit.
		uint64_t key = face_list_keys[idx];
		uint32_t currtile = key >> 32;
		if (idx == 0)
			ranges[currtile].x = 0;
		else
		{
			uint32_t prevtile = face_list_keys[idx - 1] >> 32;
			if (currtile != prevtile)
			{
				ranges[prevtile].y = idx;
				ranges[currtile].x = idx;
			}
		}
		if (idx == L - 1)
			ranges[currtile].y = L;
	}
}

CudaRenderer::VertState CudaRenderer::VertState::fromChunk(
	char*& chunk, 
	size_t P
)
{
	VertState vert;
	obtain(chunk, vert.ndc, P, 128);
	obtain(chunk, vert.image, P, 128);
	return vert;
}

CudaRenderer::FaceState CudaRenderer::FaceState::fromChunk(
	char*& chunk, 
	size_t F)
{
	FaceState face;
	obtain(chunk, face.depths, F, 128);
	obtain(chunk, face.min_depths, F, 128);
	obtain(chunk, face.max_depths, F, 128);
	obtain(chunk, face.tiles_touched, F, 128);
	cub::DeviceScan::InclusiveSum(nullptr, face.scan_size, face.tiles_touched, face.tiles_touched, F);
	obtain(chunk, face.scanning_space, face.scan_size, 128);
	obtain(chunk, face.face_offsets, F, 128);
	return face;
}

CudaRenderer::ImageState CudaRenderer::ImageState::fromChunk(
	char*& chunk,
	size_t N)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.final_log_T, N, 128);
	obtain(chunk, img.final_prev_log_T, N, 128);
	obtain(chunk, img.ray_o, N, 128);
	obtain(chunk, img.ray_d, N, 128);
	obtain(chunk, img.first_face, N, 128);
	obtain(chunk, img.first_tet, N, 128);
	obtain(chunk, img.last_face, N, 128);
	obtain(chunk, img.last_tet, N, 128);
	obtain(chunk, img.is_active_forward, N, 128);
	obtain(chunk, img.is_active_backward, N, 128);
	return img;
}

CudaRenderer::BinningState CudaRenderer::BinningState::fromChunk(
	char*& chunk, 
	size_t B)
{
	BinningState binning;
	obtain(chunk, binning.face_list, B, 128);
	obtain(chunk, binning.face_list_unsorted, B, 128);
	obtain(chunk, binning.face_list_keys, B, 128);
	obtain(chunk, binning.face_list_keys_unsorted, B, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.face_list_keys_unsorted, binning.face_list_keys,
		binning.face_list_unsorted, binning.face_list, B);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for tetrahedral grid.
int CudaRenderer::Renderer::forward(
	std::function<char* (size_t)> pointBuffer,
	std::function<char* (size_t)> faceBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	
	const int B, int P, int F, int T,
	const float* background,
	const int width, int height,

	const float* verts,
	const int* faces,
	const float* verts_color,
	const float* faces_opacity,

	const float* mv_mats,
	const float* proj_mats,
	const float* inv_mv_mats,
	const float* inv_proj_mats,
	const float* verts_depth,
	const float* faces_intense,
	
	const int* tets,
	const int* face_tets,
	const int* tet_faces,
	
	int ray_random_seed,			// larger than 0 means using random ray direction
	
	float* out_color,
	float* out_depth,
	float* out_active)
{
	int BP = B * P;
	size_t point_chunk_size = required<VertState>(BP);
	char* point_chunkptr = pointBuffer(point_chunk_size);
	VertState vertState = VertState::fromChunk(point_chunkptr, BP);

	int BF = B * F;
	size_t face_chunk_size = required<FaceState>(BF);
	char* face_chunkptr = faceBuffer(face_chunk_size);
	FaceState faceState = FaceState::fromChunk(face_chunkptr, BF);

	int BI = B * width * height;
	size_t img_chunk_size = required<ImageState>(BI);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, BI);

	// Run preprocessing (points)
	CHECK_CUDA(TET_FORWARD::preprocess_point(
		B,
		P,
		verts,
		mv_mats,
		proj_mats,
		width,
		height,
		vertState.ndc,
		vertState.image), true);

	// Define ray origin and direction for each pixel;
	// std::cout << "=== Right before generating rays..." << std::endl;
	CHECK_CUDA(TET_FORWARD::generate_rays(
		inv_mv_mats,
		inv_proj_mats,
		B,
		width,
		height,
		imgState.ray_o,
		imgState.ray_d,
		ray_random_seed), true);	

	// ===================
	// Find starting face of each ray
	// ===================

	// This is done by triangle version of Gaussian splatting's rasterization scheme;

    // BLOCK_X: Number of pixels for a tile, horizontally
    // BLOCK_Y: Number of pixels for a tile, vertically
    // tile_grid: Number of tiles, horizontally and vertically
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 batch_tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, B);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	CHECK_CUDA(TET_FORWARD::preprocess_face(
		B, P, F,
		verts,
		vertState.ndc,
		vertState.image,
		faces,
		mv_mats,
		proj_mats,
		width,
		height,
		faceState.depths,
		faceState.min_depths,
		faceState.max_depths,
		tile_grid,
		faceState.tiles_touched), true);

	// Compute prefix sum over full list of 
	// touched tile counts by faces into face_offsets.
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(
		cub::DeviceScan::InclusiveSum(
			faceState.scanning_space, 
			faceState.scan_size, 
			faceState.tiles_touched, 
			faceState.face_offsets, 
			B * F), true)

	// Retrieve total number of face instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(
		cudaMemcpy(
			&num_rendered, 
			faceState.face_offsets + (B * F) - 1, sizeof(int), 
			cudaMemcpyDeviceToHost), true);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated face indices to be sorted
	duplicateWithKeys << <(B * F + 255) / 256, 256 >> > (
		B,
		P,
		F,
		faces,
		vertState.image,
		// faceState.depths,
		faceState.min_depths,		// sort by min depth
		faceState.face_offsets,
		binningState.face_list_keys_unsorted,
		binningState.face_list_unsorted,
		tile_grid);
	CHECK_CUDA(, true)

	int bit = getHigherMsb(B * tile_grid.x * tile_grid.y);		// ??

	// Sort complete list of (duplicated) face indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.face_list_keys_unsorted, binningState.face_list_keys,
		binningState.face_list_unsorted, binningState.face_list,
		num_rendered, 0, 32 + bit), true)

	// Even though we use (batch * width * height) number of entries for [imgState.ranges],
	// we use only (batch * Num tile) number of entries, not entire pixels;
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, B * tile_grid.x * tile_grid.y * sizeof(uint2)), true);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.face_list_keys,
			imgState.ranges);
	CHECK_CUDA(, true)

	// Identify first triangle face that each ray collides with
	CHECK_CUDA(TET_FORWARD::first_intersect(
		batch_tile_grid, block,
		verts,
		faces,
		tets,
		face_tets,
		faceState.min_depths,
		faceState.max_depths,
		imgState.ranges,
		binningState.face_list,
		B, F, width, height,
		imgState.ray_o,
		imgState.ray_d,
		imgState.first_face,
		imgState.first_tet), true);

	// ===================
	// Forward rendering by ray marching
	// ===================
	CHECK_CUDA(TET_FORWARD::render(
		batch_tile_grid, block,
		
		verts,
		faces,
		verts_color,
		faces_opacity,
		
		mv_mats,
		proj_mats,
		verts_depth,
		faces_intense,
		
		tets,
		face_tets,
		tet_faces,
		
		imgState.ray_o,
		imgState.ray_d,
		imgState.first_face,
		imgState.first_tet,
		
		width, height,
		P, F,
		
		background,

		imgState.final_log_T,
		imgState.final_prev_log_T,
		imgState.last_face,
		imgState.last_tet,
		imgState.n_contrib,
		imgState.is_active_forward,
		out_color,
		out_depth,
		out_active), true);
}

// Backward rendering procedure for tetrahedral grid.
void CudaRenderer::Renderer::backward(
	const int B, int P, int F, int T,
	const float* background,
	const int width, int height,
	
	const float* verts,
	const int* faces,
	const float* verts_color,
	const float* faces_opacity,

	const float* mv_mats,
	const float* proj_mats,
	const float* inv_mv_mats,
	const float* inv_proj_mats,
	const float* verts_depth,
	const float* faces_intense,

	const int* tets,
	const int* face_tets,
	const int* tet_faces,

	char* point_buffer,
	char* face_buffer,
	char* binning_buffer,
	char* image_buffer,
	
	const float* dL_dpix_color,
	const float* dL_dpix_depth,

	float* dL_dverts_color,
	float* dL_dfaces_opacity)
{
	int BP = B * P;
	VertState vertState = VertState::fromChunk(point_buffer, BP);

	int BF = B * F;
	FaceState faceState = FaceState::fromChunk(face_buffer, BF);
	
	int BI = B * width * height;
	ImageState imgState = ImageState::fromChunk(image_buffer, BI);

	const dim3 batch_tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, B);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. verts_attr
	// from per-pixel loss gradients.
	CHECK_CUDA(TET_BACKWARD::render(
		batch_tile_grid,
		block,
		
		B, P, F,
		width, height,	
		background,

		verts,
		faces,
		verts_color,
		faces_opacity,

		mv_mats,
		proj_mats,
		verts_depth,
		faces_intense,
		
		tets,
		face_tets,
		tet_faces,
		
		imgState.ray_o,
		imgState.ray_d,
		imgState.first_face,
		imgState.first_tet,
		imgState.last_face,
		imgState.last_tet,
		imgState.final_log_T,
		imgState.final_prev_log_T,
		imgState.is_active_forward,
		imgState.n_contrib,
		
		dL_dpix_color,
		dL_dpix_depth,

		dL_dverts_color,
		dL_dfaces_opacity,
		imgState.is_active_backward), true);
}