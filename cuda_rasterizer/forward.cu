#include "forward.h"
#include "auxiliary.h"
#include "cuda_math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


namespace TRI_FORWARD {
	/*
	========================================================
	Preprocess Points
	========================================================
	*/

	// Perform initial steps for each point to rasterization.
	__global__ void preprocessPointCUDA(
		int B, int P,
		const float* verts,
		const float* mv_mats,
		const float* proj_mats,
		int W, int H,
		float3* verts_ndc,
		float2* verts_image)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= B * P)
			return;

		int batch_id = idx / P;
		int point_id = idx % P;
		const float* b_mv = &mv_mats[batch_id * 16];
		const float* b_proj = &proj_mats[batch_id * 16];

		// Transform point to NDC: Now every coordinates should reside in [-1, 1]
		float3 p_view = transformPoint4x3({ verts[3 * point_id + 0], verts[3 * point_id + 1], verts[3 * point_id + 2] }, b_mv);
		float4 p_proj = transformPoint4x4(p_view, b_proj);
		float p_w = 1.0 / clamp_w(p_proj.w);
		float3 p_ndc = { p_proj.x * p_w, p_proj.y * p_w, p_proj.z * p_w };
		verts_ndc[idx].x = p_ndc.x;
		verts_ndc[idx].y = p_ndc.y;
		verts_ndc[idx].z = p_ndc.z;

		// Transform point to image space: Now every coordinates should reside in [0, W] and [0, H]
		verts_image[idx].x = ndc2Pix(p_ndc.x, W);
		verts_image[idx].y = ndc2Pix(p_ndc.y, H);
	}

	void preprocess_point(
		int B, int P,
		const float* verts,
		const float* mv_mats,
		const float* proj_mats,
		int W, int H,
		float3* verts_ndc,
		float2* verts_image)
	{
		int BP = B * P;
		preprocessPointCUDA << <(BP + 255) / 256, 256 >> > (
			B, P,
			verts,
			mv_mats,
			proj_mats,
			W, H,
			verts_ndc,
			verts_image);
	}

	/*
	========================================================
	Preprocess Faces
	========================================================
	*/

	// Perform initial steps for each triangle to rasterization.
	__global__ void preprocessFaceCUDA(
		int B, int P, int F,
		const float* verts,
		const float3* verts_ndc,
		const float2* verts_image,
		const int* faces,
		const float* mv_mats,
		const float* proj_mats,
		const int W, int H,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= B * F)
			return;

		int batch_id = idx / F;
		int face_id = idx % F;

		// Initialize touched tiles to 0. If this isn't changed,
		// this triangle will not be processed further.
		tiles_touched[idx] = 0;

		// Get point info;
		float max_z = 0;
		float min_z = 0;
		float depth = 0;
		float2 face_verts_image[3];
		for (int i = 0; i < 3; i++) {
			int face_vert_id = faces[3 * face_id + i];
			int b_face_vert_id = batch_id * P + face_vert_id;
			const float3* face_vert_ndc = &verts_ndc[b_face_vert_id];
			float z = face_vert_ndc->z;
			if (i == 0) {
				max_z = z;
				min_z = z;
			}
			else {
				max_z = max(max_z, z);
				min_z = min(min_z, z);
			}
			depth += z;
			face_verts_image[i] = verts_image[b_face_vert_id];
		}
		depth = depth / 3.0f;

		// If triangle is completely behind or front of camera, quit.
		if (max_z < -1.0f || min_z > 1.0f)
			return;

		// Compute a bounding rectangle of screen-space tiles that this 
		// triangle overlaps with. Quit if rectangle covers 0 tiles.
		uint2 rect_min, rect_max;
		getRectFromTri(face_verts_image[0],
						face_verts_image[1],
						face_verts_image[2],
						rect_min,
						rect_max,
						grid);

		// If triangle is completely outside of camera, quit.
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			return;

		// Store some useful helper data for the next steps.
		depths[idx] = depth;
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

		// change range of [depth] from [-1, 1] to [0, 1] for ordering later;
		depths[idx] = (depth + 1.0f) * 0.5f;
		if (depths[idx] < 0.0f) depths[idx] = 0.0f;
		if (depths[idx] > 1.0f) depths[idx] = 1.0f;
	}

	void preprocess_face(
		int B, int P, int F,
		const float* verts,
		const float3* verts_ndc,
		const float2* verts_image,
		const int* faces,
		const float* mv_mats,
		const float* proj_mats,
		const int W, int H,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched)
	{
		int BF = B * F;
		preprocessFaceCUDA << <(BF + 255) / 256, 256 >> > (
			B, P, F,
			verts,
			verts_ndc,
			verts_image,
			faces,
			mv_mats,
			proj_mats,
			W, H,
			depths,
			grid,
			tiles_touched);
	}

	/*
	========================================================
	Generate rays for each pixel
	========================================================
	*/
	__global__ void generateRaysCUDA(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d
	)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= B * W * H)
			return;

		int batch_id = idx / (W * H);
		int pixel_id = idx % (W * H);

		// Get the inverse modelview and projection matrices
		const float* inv_modelviewmatrix = &inv_mv_mats[batch_id * 16];
		const float* inv_projmatrix = &inv_proj_mats[batch_id * 16];
		
		// Ray origin is the camera position in world space;
		ray_o[idx].x = inv_modelviewmatrix[3 * 4 + 0];
		ray_o[idx].y = inv_modelviewmatrix[3 * 4 + 1];
		ray_o[idx].z = inv_modelviewmatrix[3 * 4 + 2];

		// Find out (floating point) pixel coords that we will cast a ray through;
		int pixel_x = pixel_id % W;
		int pixel_y = pixel_id / W;

		float2 pixf;
		pixf.x = pixel_x + 0.5f;
		pixf.y = pixel_y + 0.5f;
		
		// Transform pixel coords to world coords;
		// First, bring it to NDC;
		float2 pix_ndc;
		pix_ndc.x = pix2Ndc(pixf.x, W);
		pix_ndc.y = pix2Ndc(pixf.y, H);

		// Then, bring it to world space;
		float4 pix_view = transformPoint4x4({ pix_ndc.x, pix_ndc.y, -1.0f }, inv_projmatrix);
		float4 pix_world = transformPoint4x4({ pix_view.x, pix_view.y, pix_view.z }, inv_modelviewmatrix);
		float3 ray_target = { pix_world.x, pix_world.y, pix_world.z };

		// Ray direction is the normalized vector from camera position to pixel position;
		ray_d[idx] = ray_target - ray_o[idx];
		float len = sqrtf(dot(ray_d[idx], ray_d[idx])) + 0.0000001f;
		ray_d[idx] = ray_d[idx] / len;
	}

	void generate_rays(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d)
	{
		generateRaysCUDA << <(B * W * H + 255) / 256, 256 >> > (
			inv_mv_mats,
			inv_proj_mats,
			B, W, H,
			ray_o,
			ray_d);
	}

	/*
	========================================================
	Rendering
	========================================================
	*/

	// Main rasterization method. Collaboratively works on one tile per
	// block, each thread treats one pixel. Alternates between fetching 
	// and rasterizing data.
	template <uint32_t CHANNELS>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
		const float* __restrict__ verts,
		const int* __restrict__ faces,
		const float* __restrict__ verts_color,
		const float* __restrict__ faces_opaciy,

		const float* __restrict__ mv_mats,
		const float* __restrict__ proj_mats,
		const float* __restrict__ verts_depth,
		const float* __restrict__ faces_intense,

		const float3* __restrict__ ray_o,
		const float3* __restrict__ ray_d,
		
		const float2* __restrict__ verts_image,
		const uint2* __restrict__ ranges,
		const uint32_t* __restrict__ face_list,
		int B, int P, int F, int W, int H,

		float* __restrict__ final_T,
		float* __restrict__ final_prev_T,
		uint32_t* __restrict__ n_contrib,
		
		const float* __restrict__ bg_color,
		float* __restrict__ out_color,
		float* __restrict__ out_depth)
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;

		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;

		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		float2 pixf = { (float)pix.x, (float)pix.y };

		uint32_t pix_id = W * pix.y + pix.x;
		uint32_t b_pix_id = (batch_id * W * H) + pix_id;

		// Check if this thread is associated with a valid pixel or outside.
		bool inside = batch_id < B && pix.x < W && pix.y < H;
		// Done threads can help with fetching, but don't rasterize
		bool done = !inside;

		float3 this_ray_o, this_ray_d;
		if (inside) {
			this_ray_o = ray_o[b_pix_id];
			this_ray_d = ray_d[b_pix_id];
		}

		// Load start/end range of IDs to process in bit sorted list.
		int b_tile_id = (batch_id * horizontal_blocks * vertical_blocks) + 
							(block.group_index().y * horizontal_blocks) + block.group_index().x;
		uint2 range = ranges[b_tile_id];
		const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int toDo = range.y - range.x;

		// Allocate storage for batches of collectively fetched data.
		__shared__ int collected_face_id[BLOCK_SIZE];

		__shared__ float3 collected_face_vert_0[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_1[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_2[BLOCK_SIZE];

		__shared__ float2 collected_face_vert_image_0[BLOCK_SIZE];
		__shared__ float2 collected_face_vert_image_1[BLOCK_SIZE];
		__shared__ float2 collected_face_vert_image_2[BLOCK_SIZE];

		__shared__ float collected_face_vert_color_0[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_1[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_2[BLOCK_SIZE * CHANNELS];

		__shared__ float collected_face_vert_depth_0[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_1[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_2[BLOCK_SIZE];
		
		__shared__ float collected_face_opacity[BLOCK_SIZE];
		__shared__ float collected_face_intense[BLOCK_SIZE];

		// Initialize helper variables
		float pT = 1.0f;
		float T = 1.0f;
		uint32_t contributor = 0;
		uint32_t last_contributor = 0;
		float C[CHANNELS] = { 0 };
		float D[1] = { 0 };

		// Iterate over batches until all done or range is complete
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// End if entire block votes that it is done rasterizing
			int num_done = __syncthreads_count(done);
			if (num_done == BLOCK_SIZE)
				break;

			// Collectively fetch per-face data from global to shared
			int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int face_id = face_list[range.x + progress];
				int b_face_id = batch_id * F + face_id;

				collected_face_id[block.thread_rank()] = face_id;

				int face_vert_id_0 = faces[3 * face_id + 0];
				int face_vert_id_1 = faces[3 * face_id + 1];
				int face_vert_id_2 = faces[3 * face_id + 2];

				int b_face_vert_id_0 = batch_id * P + face_vert_id_0;
				int b_face_vert_id_1 = batch_id * P + face_vert_id_1;
				int b_face_vert_id_2 = batch_id * P + face_vert_id_2;

				get_face_vert(
					verts,
					faces,
					face_id,
					collected_face_vert_0[block.thread_rank()],
					collected_face_vert_1[block.thread_rank()],
					collected_face_vert_2[block.thread_rank()]
				);

				collected_face_vert_image_0[block.thread_rank()] = verts_image[b_face_vert_id_0];
				collected_face_vert_image_1[block.thread_rank()] = verts_image[b_face_vert_id_1];
				collected_face_vert_image_2[block.thread_rank()] = verts_image[b_face_vert_id_2];

				for (int ch = 0; ch < CHANNELS; ch++)
				{
					collected_face_vert_color_0[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_0 * CHANNELS + ch];
					collected_face_vert_color_1[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_1 * CHANNELS + ch];
					collected_face_vert_color_2[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_2 * CHANNELS + ch];
				}

				collected_face_vert_depth_0[block.thread_rank()] = verts_depth[b_face_vert_id_0];
				collected_face_vert_depth_1[block.thread_rank()] = verts_depth[b_face_vert_id_1];
				collected_face_vert_depth_2[block.thread_rank()] = verts_depth[b_face_vert_id_2];

				collected_face_opacity[block.thread_rank()] = faces_opaciy[face_id];
				collected_face_intense[block.thread_rank()] = faces_intense[b_face_id];
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				// Keep track of current position in range
				contributor++;

				// Find out if the current face overlaps the current pixel;
				// if not, skip it.
				pixf.x = pix.x + 0.5f;
				pixf.y = pix.y + 0.5f;
				if (!in_tri(pixf, 
							collected_face_vert_image_0[j], 
							collected_face_vert_image_1[j], 
							collected_face_vert_image_2[j]))
					continue;

				// Find intersection point between ray and triangle;
				// It is used to interpolate vert-wise color and depth;
				const float3& v0 = collected_face_vert_0[j];
				const float3& v1 = collected_face_vert_1[j];
				const float3& v2 = collected_face_vert_2[j];
				float3 tuv = { 0, 0, 0 };
				bool not_edge_case = ray_tri_intersection(
					this_ray_o, this_ray_d,
					v0, v1, v2, tuv);

				if (!not_edge_case)
					continue;

				// clamp, because (iu, iv) could be outside of the triangle.
				float iu = tuv.y, iv = tuv.z;
				float iuc, ivc;
				int iclamp_code;
				clamp_bary_uv(iu, iv, iuc, ivc, iclamp_code);	
				float i0 = 1 - iuc - ivc, i1 = iuc, i2 = ivc;
				
				// Find intersection point's color and depth;
				float iC[CHANNELS] = { 0 };
				float iD[1] = { 0 };
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					iC[ch] = i0 * collected_face_vert_color_0[j * CHANNELS + ch] +
							i1 * collected_face_vert_color_1[j * CHANNELS + ch] +
							i2 * collected_face_vert_color_2[j * CHANNELS + ch];
					iC[ch] = iC[ch] * collected_face_intense[j];
				}
				iD[0] = i0 * collected_face_vert_depth_0[j] +
						i1 * collected_face_vert_depth_1[j] +
						i2 * collected_face_vert_depth_2[j];

				// Find intersecting face's opacity;
				float alpha = collected_face_opacity[j];
				float test_T = T * (1 - alpha);

				// alpha blending.
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += iC[ch] * alpha * T;
				D[0] += iD[0] * alpha * T;
				
				pT = T;
				T = test_T;

				// Keep track of last range entry to update this
				// pixel.
				last_contributor = contributor;

				if (T < T_EPS) {
					done = true;
					break;
				}
			}
		}

		// All threads that treat valid pixel write out their final
		// rendering data to the frame and auxiliary buffers.
		if (inside)
		{
			final_prev_T[b_pix_id] = pT;
			final_T[b_pix_id] = T;
			n_contrib[b_pix_id] = last_contributor;
			
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[(batch_id * CHANNELS * H * W) + (ch * H * W) + pix_id] = 
					C[ch] + T * bg_color[ch];
			out_depth[(batch_id * H * W) + pix_id] = D[0] + T * 1.0f;
		}
	}

	void render(
		const dim3 grid, dim3 block,
			
		const float* verts,
		const int* faces,
		const float* verts_color,
		const float* faces_opacity,

		const float* mv_mats,
		const float* proj_mats,
		const float* verts_depth,
		const float* faces_intense,

		const float3* ray_o,
		const float3* ray_d,

		const float2* verts_image,
		const uint2* ranges,
		const uint32_t* face_list,
		int B, int P, int F, int W, int H,
		
		float* final_T,
		float* final_prev_T,
		uint32_t* n_contrib,
		
		const float* bg_color,
		float* out_color,
		float* out_depth)
	{
		renderCUDA<NUM_CHANNELS> << <grid, block >> > (
			verts,
			faces,
			verts_color,
			faces_opacity,

			mv_mats,
			proj_mats,
			verts_depth,
			faces_intense,

			ray_o,
			ray_d,
			
			verts_image,
			ranges,
			face_list,
			B, P, F, W, H,
			
			final_T,
			final_prev_T,
			n_contrib,
			
			bg_color,
			out_color,
			out_depth);
	}
}