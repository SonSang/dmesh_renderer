#ifndef CUDA_RENDERER_FORWARD_H_INCLUDED
#define CUDA_RENDERER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace TET_FORWARD
{
	// Perform initial steps for each point to rasterization.
	// * verts_ndc: 3D coordinates of each point in NDC space.
	// * verts_image: 2D coordinates of each point in image space (pixel).
	void preprocess_point(
		int B,
		int P,
		const float* verts,
		const float* mv_mats,
		const float* proj_mats,
		int W, int H,
		float3* verts_ndc,
		float2* verts_image
	);

	// Generate rays for each pixel.
	void generate_rays(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d,
		int random_seed);

	// Perform initial steps for each face to rasterization.
	void preprocess_face(int B, int P, int F,
		const float* verts,
		const float3* verts_ndc,
		const float2* verts_image,
		const int* faces,
		const float* mv_mats,
		const float* proj_mats,
		const int W, int H,
		float* depths,
		float* min_depths,
		float* max_depths,
		const dim3 grid,
		uint32_t* tiles_touched);

	// Find the first intersecting face between each pixel ray.
	void first_intersect(
		const dim3 grid, dim3 block,
		const float* verts,
		const int* faces,
		const int* tets,
		const int* face_tets,
		const float* faces_min_depth,
		const float* faces_max_depth,
		const uint2* ranges,
		const uint32_t* face_list,
		int B, int F, int W, int H,
		const float3* ray_o,
		const float3* ray_d,
		int* first_face,
		int* first_tet);

	// Main rendering method by ray marching.
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
		
		const int* tets,
		const int* face_tets,
		const int* tet_faces,
		
		const float3* ray_o,
		const float3* ray_d,
		const int* first_face,
		const int* first_tet,

		int W, int H,
		int P, int F,
		
		const float* bg_color,
		
		float* final_log_T,
		float* final_prev_log_T,
		int* last_face,
		int* last_tet,
		uint32_t* n_contrib,
		bool* is_active,
		float* out_color,
		float* out_depth,
		float* out_active);
}

#endif