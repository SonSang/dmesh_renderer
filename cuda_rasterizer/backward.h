#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace TRI_BACKWARD
{
	void render(
		const dim3 grid, dim3 block,

		const int B, int P, int F,

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

		const uint2* ranges,
		const float3* ray_o,
		const float3* ray_d,
		const uint32_t* face_list,
		int W, int H,
		const float* bg_color,
		const float2* verts_image,
		const float* final_Ts,
		const float* final_prev_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpix_color,
		const float* dL_dpix_depth,

		float* dL_dverts,
		float* dL_dvcolor,
		float* dL_dfopacity,
		float* dL_dvdepth,
		float* dL_dfintense);
}

#endif