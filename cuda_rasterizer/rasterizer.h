#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static int forward(
			std::function<char* (size_t)> pointBuffer,
			std::function<char* (size_t)> faceBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			
			const int B, int P, int F,
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
			const float* faces_depth,

			float* out_color,
			float* out_depth);

		static void backward(
			const int B, int P, int F, int R,
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

			char* point_buffer,
			char* face_buffer,
			char* binning_buffer,
			char* image_buffer,
			
			const float* dL_dpix_color,
			const float* dL_dpix_depth,

			float* dL_dverts,
			float* dL_dvcolor,
			float* dL_dfopacity,
			float* dL_dvdepth,
			float* dL_dfintense);
	};
};

#endif