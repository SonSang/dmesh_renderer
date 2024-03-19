#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace TRI_FORWARD
{
	// Perform initial steps for each point to rasterization.
	// * verts_ndc: 3D coordinates of each point in NDC space.
	// * verts_image: 2D coordinates of each point in image space (pixel).
	void preprocess_point(
		int B, int P,
		const float* verts,
		const float* mv_mats,
		const float* proj_mats,
		int W, int H,
		float3* verts_ndc,
		float2* verts_image);

	// Perform initial steps for each face to rasterization.
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
		uint32_t* tiles_touched);

	// Generate rays for each pixel.
	void generate_rays(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d);

	// Main rasterization method.
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
		float* out_depth);
}


#endif