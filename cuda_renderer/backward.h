#ifndef CUDA_RENDERER_BACKWARD_H_INCLUDED
#define CUDA_RENDERER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace TET_BACKWARD
{
	void render(
		const dim3 batch_grid, dim3 block,

        int B, int P, int F,
        int W, int H,
		const float* bg_color,
        
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
        const int* last_face,
        const int* last_tet,
        const float* final_log_T,
        const float* final_prev_log_T,
        const bool* is_active_forward,
        const uint32_t* n_contrib,
        
        const float* dL_dpix_color,
        const float* dL_dpix_depth,

        float* dL_dverts_color,
        float* dL_dfaces_opacity,
        bool* is_active_backward);
}

#endif