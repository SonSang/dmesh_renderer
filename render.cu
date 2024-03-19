#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_renderer/config.h"
#include "cuda_renderer/renderer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const int image_width)
{
    // std::cout<<"=== HI, CUDA is coming!"<<std::endl;
    if (verts.ndimension() != 2 || verts.size(1) != 3) {
        AT_ERROR("verts must have dimensions (num_points, 3)");
    }
    if (faces.ndimension() != 2 || faces.size(1) != 3) {
        AT_ERROR("faces must have dimensions (num_faces, 3)");
    }
    if (verts_color.ndimension() != 2 || verts_color.size(0) != verts.size(0)) {
        AT_ERROR("vert color must have dimensions (num_points, N)");
    }
    if (faces_opacity.ndimension() != 1 || faces_opacity.size(0) != faces.size(0)) {
        AT_ERROR("face opacity must have dimensions (num_faces,)");
    }

    if (mv_mats.ndimension() != 3 || mv_mats.size(1) != 4 || mv_mats.size(2) != 4) {
        AT_ERROR("mv_mats must have dimensions (B, 4, 4)");
    }
    if (proj_mats.ndimension() != 3 || proj_mats.size(1) != 4 || proj_mats.size(2) != 4) {
        AT_ERROR("proj_mats must have dimensions (B, 4, 4)");
    }
    if (inv_mv_mats.ndimension() != 3 || inv_mv_mats.size(1) != 4 || inv_mv_mats.size(2) != 4) {
        AT_ERROR("inv_mv_mats must have dimensions (B, 4, 4)");
    }
    if (inv_proj_mats.ndimension() != 3 || inv_proj_mats.size(1) != 4 || inv_proj_mats.size(2) != 4) {
        AT_ERROR("inv_proj_mats must have dimensions (B, 4, 4)");
    }
    if (verts_depth.ndimension() != 2 || verts_depth.size(1) != verts.size(0)) {
        AT_ERROR("verts_depth must have dimensions (B, num_points,)");
    }
    if (faces_intense.ndimension() != 2 || faces_intense.size(1) != faces.size(0)) {
        AT_ERROR("faces_intense must have dimensions (B, num_faces,)");
    }
    
    const int B = mv_mats.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int H = image_height;
    const int W = image_width;

    auto float_opts = verts.options().dtype(torch::kFloat32);
    torch::Tensor out_color = torch::full({B, NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor out_depth = torch::full({B, 1, H, W}, 0.0, float_opts);
    
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kByte);
    torch::Tensor pointBuffer = torch::empty({0}, options.device(device));
    torch::Tensor faceBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> pointFunc = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> faceFunc = resizeFunctional(faceBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    // std::cout<<"=== Initialized buffers"<<std::endl;
  
    int rendered = 0;
    if(P != 0)
    {
        rendered = CudaRasterizer::Rasterizer::forward(
            pointFunc,
            faceFunc,
            binningFunc,
            imgFunc,
            B, P, F,
            background.contiguous().data<float>(),
            W, H,
            
            verts.contiguous().data<float>(),
            faces.contiguous().data<int>(),
            verts_color.contiguous().data<float>(),
            faces_opacity.contiguous().data<float>(),
            
            mv_mats.contiguous().data<float>(),
            proj_mats.contiguous().data<float>(),
            inv_mv_mats.contiguous().data<float>(),
            inv_proj_mats.contiguous().data<float>(),
            verts_depth.contiguous().data<float>(),
            faces_intense.contiguous().data<float>(),

            out_color.contiguous().data<float>(),
            out_depth.contiguous().data<float>());
    }
    return std::make_tuple(rendered, out_color, out_depth, pointBuffer, faceBuffer, binningBuffer, imgBuffer);
}

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
	const torch::Tensor& imageBuffer) 
{
    const int B = mv_mats.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int H = dL_dout_color.size(2);
    const int W = dL_dout_color.size(3);
    
    torch::Tensor dL_dverts = torch::zeros({P, 3}, verts.options());
    torch::Tensor dL_dvcolor = torch::zeros({P, NUM_CHANNELS}, verts_color.options());
    torch::Tensor dL_dfopacity = torch::zeros({F}, faces_opacity.options());

    torch::Tensor dL_dvdepth = torch::zeros({B, P}, verts_depth.options());
    torch::Tensor dL_dfintense = torch::zeros({B, F}, faces_intense.options());
    
    if(F != 0)
    {  
        CudaRasterizer::Rasterizer::backward(
            B, P, F, R,
            background.contiguous().data<float>(),
            W, H, 
            
            verts.contiguous().data<float>(),
            faces.contiguous().data<int>(),
            verts_color.contiguous().data<float>(),
            faces_opacity.contiguous().data<float>(),
            
            mv_mats.contiguous().data<float>(),
            proj_mats.contiguous().data<float>(),
            inv_mv_mats.contiguous().data<float>(),
            inv_proj_mats.contiguous().data<float>(),
            verts_depth.contiguous().data<float>(),
            faces_intense.contiguous().data<float>(),

            reinterpret_cast<char*>(pointBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(faceBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            
            dL_dout_color.contiguous().data<float>(),
            dL_dout_depth.contiguous().data<float>(),

            dL_dverts.contiguous().data<float>(),
            dL_dvcolor.contiguous().data<float>(),
            dL_dfopacity.contiguous().data<float>(),
            dL_dvdepth.contiguous().data<float>(),
            dL_dfintense.contiguous().data<float>());
    }

    return std::make_tuple(dL_dverts, dL_dvcolor, dL_dfopacity, dL_dvdepth, dL_dfintense);
}

/*
Tet Renderer
*/
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
	const int ray_random_seed)
{
    if (verts.ndimension() != 2 || verts.size(1) != 3) {
        AT_ERROR("verts must have dimensions (num_points, 3)");
    }
    if (faces.ndimension() != 2 || faces.size(1) != 3) {
        AT_ERROR("faces must have dimensions (num_faces, 3)");
    }
    if (verts_color.ndimension() != 2 || verts_color.size(0) != verts.size(0) || verts_color.size(1) != 3) {
        AT_ERROR("vert_color must have dimensions (num_verts, 3)");
    }
    if (faces_opacity.ndimension() != 1 || faces_opacity.size(0) != faces.size(0)) {
        AT_ERROR("face_opacity must have dimensions (num_faces)");
    }

    if (mv_mats.ndimension() != 3 || mv_mats.size(2) != 4 || mv_mats.size(1) != 4) {
        AT_ERROR("mv_mats must have dimensions (batch_size, 4, 4)");
    }
    if (proj_mats.ndimension() != 3 || proj_mats.size(2) != 4 || proj_mats.size(1) != 4) {
        AT_ERROR("proj_mats must have dimensions (batch_size, 4, 4)");
    }
    if (inv_mv_mats.ndimension() != 3 || inv_mv_mats.size(2) != 4 || inv_mv_mats.size(1) != 4) {
        AT_ERROR("inv_mv_mats must have dimensions (batch_size, 4, 4)");
    }
    if (inv_proj_mats.ndimension() != 3 || inv_proj_mats.size(2) != 4 || inv_proj_mats.size(1) != 4) {
        AT_ERROR("inv_proj_mats must have dimensions (batch_size, 4, 4)");
    }
    if (verts_depth.ndimension() != 2 || verts_depth.size(1) != verts.size(0)) {
        AT_ERROR("verts_depth must have dimensions (batch_size, num_verts)");
    }
    if (faces_intense.ndimension() != 2 || faces_intense.size(1) != faces.size(0)) {
        AT_ERROR("faces_intense must have dimensions (batch_size, num_faces)");
    }

    if (tets.ndimension() != 2 || tets.size(1) != 4) {
        AT_ERROR("tets must have dimensions (num_tets, 4)");
    }
    if (face_tets.ndimension() != 2 || face_tets.size(0) != faces.size(0) || face_tets.size(1) != 2) {
        AT_ERROR("face_tets must have dimensions (num_faces, 2)");
    }
    if (tet_faces.ndimension() != 2 || tet_faces.size(0) != tets.size(0) || tet_faces.size(1) != 4) {
        AT_ERROR("tet_faces must have dimensions (num_tets, 4)");
    }

    const int B = mv_mats.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int T = tets.size(0);
    
    const int H = image_height;
    const int W = image_width;

    auto float_opts = verts.options().dtype(torch::kFloat32);
    torch::Tensor out_color = torch::full({B, NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor out_depth = torch::full({B, 1, H, W}, 0.0, float_opts);
    torch::Tensor out_active = torch::full({B, H, W}, 0.0, float_opts);
    
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().dtype(torch::kByte);
    torch::Tensor pointBuffer = torch::empty({0}, options.device(device));
    torch::Tensor faceBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> pointFunc = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> faceFunc = resizeFunctional(faceBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    CudaRenderer::Renderer::forward(
        pointFunc,
        faceFunc,
        binningFunc,
        imgFunc,

        B, P, F, T,
        background.contiguous().data<float>(),
        W, H,
        
        verts.contiguous().data<float>(),
        faces.contiguous().data<int>(),
        verts_color.contiguous().data<float>(),
        faces_opacity.contiguous().data<float>(),

        mv_mats.contiguous().data<float>(),
        proj_mats.contiguous().data<float>(),
        inv_mv_mats.contiguous().data<float>(),
        inv_proj_mats.contiguous().data<float>(),
        verts_depth.contiguous().data<float>(),
        faces_intense.contiguous().data<float>(),

        tets.contiguous().data<int>(),
        face_tets.contiguous().data<int>(),
        tet_faces.contiguous().data<int>(),
        
        ray_random_seed,

        out_color.contiguous().data<float>(),
        out_depth.contiguous().data<float>(),
        out_active.contiguous().data<float>()
    );
    return std::make_tuple(out_color, out_depth, out_active, pointBuffer, faceBuffer, binningBuffer, imgBuffer);
}

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
	const torch::Tensor& imageBuffer)
{
    const int B = mv_mats.size(0);
    const int P = verts.size(0);
    const int F = faces.size(0);
    const int T = tets.size(0);
    
    const int H = grad_color.size(2);   // grad_color = (B, NUM_CHANNELS, H, W)
    const int W = grad_color.size(3);

    auto float_opts = verts.options().dtype(torch::kFloat32);
    torch::Tensor dL_dverts_color = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor dL_dfaces_opacity = torch::full({F}, 0.0, float_opts);
    
    CudaRenderer::Renderer::backward(
        B, P, F, T,
        background.contiguous().data<float>(),
        W, H,
        
        verts.contiguous().data<float>(),
        faces.contiguous().data<int>(),
        verts_color.contiguous().data<float>(),
        faces_opacity.contiguous().data<float>(),

        mv_mats.contiguous().data<float>(),
        proj_mats.contiguous().data<float>(),
        inv_mv_mats.contiguous().data<float>(),
        inv_proj_mats.contiguous().data<float>(),
        verts_depth.contiguous().data<float>(),
        faces_intense.contiguous().data<float>(),

        tets.contiguous().data<int>(),
        face_tets.contiguous().data<int>(),
        tet_faces.contiguous().data<int>(),
        
        reinterpret_cast<char*>(pointBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(faceBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        
        grad_color.contiguous().data<float>(),
        grad_depth.contiguous().data<float>(),
        
        dL_dverts_color.contiguous().data<float>(),
        dL_dfaces_opacity.contiguous().data<float>()
    );

    return std::make_tuple(dL_dverts_color, dL_dfaces_opacity);
}