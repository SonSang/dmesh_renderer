#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
/*
Tri Renderer
*/
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeTrianglesCUDA(
	const torch::Tensor& background,
	
	const torch::Tensor& verts,
    const torch::Tensor& faces,
	const torch::Tensor& verts_color,
	const torch::Tensor& faces_opacity,

	const torch::Tensor& mv_mats,
	const torch::Tensor& proj_mats,
	const torch::Tensor& inv_mv_mats,
	const torch::Tensor& inv_proj_mats,
	const torch::Tensor& verts_depth,
	const torch::Tensor& faces_intense,

    const int image_height,
    const int image_width);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeTrianglesBackwardCUDA(
 	const torch::Tensor& background,
	
	const torch::Tensor& verts,
	const torch::Tensor& faces,
	const torch::Tensor& verts_color,
	const torch::Tensor& faces_opacity,

	const torch::Tensor& mv_mats,
	const torch::Tensor& proj_mats,
	const torch::Tensor& inv_mv_mats,
	const torch::Tensor& inv_proj_mats,
	const torch::Tensor& verts_depth,
	const torch::Tensor& faces_intense,
	
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,

	const int R,
	
	const torch::Tensor& pointBuffer,
	const torch::Tensor& faceBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);

/*
Tet Renderer
*/
// color, depth, pointBuffer, faceBuffer, binningBuffer, imageBuffer
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderFTetsCUDA(
	const torch::Tensor& background,

	const torch::Tensor& verts,
    const torch::Tensor& faces,
	const torch::Tensor& verts_color,
	const torch::Tensor& faces_opacity,

	const torch::Tensor& mv_mats,
	const torch::Tensor& proj_mats,
	const torch::Tensor& inv_mv_mats,
	const torch::Tensor& inv_proj_mats,
	const torch::Tensor& verts_depth,
	const torch::Tensor& faces_intense,

	const torch::Tensor& tets,
	const torch::Tensor& face_tets,
	const torch::Tensor& tet_faces,

	const int image_height,
	const int image_width,
	const int ray_random_seed);

// grad_verts_attr
std::tuple<torch::Tensor, torch::Tensor>
RenderFTetsBackwardCUDA(
	const torch::Tensor& background,

	const torch::Tensor& verts,
    const torch::Tensor& faces,
	const torch::Tensor& verts_color,
	const torch::Tensor& faces_opacity,

	const torch::Tensor& mv_mats,
	const torch::Tensor& proj_mats,
	const torch::Tensor& inv_mv_mats,
	const torch::Tensor& inv_proj_mats,
	const torch::Tensor& verts_depth,
	const torch::Tensor& faces_intense,

	const torch::Tensor& tets,
	const torch::Tensor& face_tets,
	const torch::Tensor& tet_faces,

	const torch::Tensor& grad_color,
	const torch::Tensor& grad_depth,

	const torch::Tensor& pointBuffer,
	const torch::Tensor& faceBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer
);