#ifndef CUDA_RENDERER_H_INCLUDED
#define CUDA_RENDERER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRenderer
{
	class Renderer 
	{
	public:
		static int forward(
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
			float* out_active);

		static void backward(
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
			float* dL_dfaces_opacity
		);
	};
};

#endif