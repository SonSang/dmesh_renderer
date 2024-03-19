#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include "cuda_math.h"

#define T_EPS		0.0001f

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float pix2Ndc(float v, int S)
{
	return ((v * 2.0 + 1.0) / S) - 1.0;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRectFromTri(const float2 p0, const float2 p1, const float2 p2, uint2& rect_min, uint2& rect_max, dim3 grid) 
{
	rect_min = {
		// should not be grid.x - 1 or grid.y - 1,
		// because it means that the point "really" falls into that tile.
		min(grid.x, max((int)0, (int)(min(min(p0.x, p1.x), p2.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)(min(min(p0.y, p1.y), p2.y) / BLOCK_Y)))
	};
	rect_max = {
		// add 1, because we have to get tile that is strictly larger;
		// in [duplicate_with_keys], we exclude the rect_max, so it should be strictly larger;
		min(grid.x, max((int)0, (int)(max(max(p0.x, p1.x), p2.x) / BLOCK_X) + 1)),
		min(grid.y, max((int)0, (int)(max(max(p0.y, p1.y), p2.y) / BLOCK_Y) + 1))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ bool in_tri(
	const float2& p,		// point
	const float2& p1,		// triangle
	const float2& p2,		
	const float2& p3
)
{
	// https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html

	// change to fixed-point (integer) arithmetic;
	// multiply 16, assuming that we use 16 * 16 subpixels per a pixel;
	float subpixel = 16.0f;
	int px = (int)(p.x * subpixel);
	int py = (int)(p.y * subpixel);
	int x1 = (int)(p1.x * subpixel);
	int y1 = (int)(p1.y * subpixel);
	int x2 = (int)(p2.x * subpixel);
	int y2 = (int)(p2.y * subpixel);
	int x3 = (int)(p3.x * subpixel);
	int y3 = (int)(p3.y * subpixel);

	int area = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
	if (area == 0)
		return false;	// degenerate triangle
	else if (area < 0) {
		// swap, to make vertex order CCW;
		int tx, ty;
		tx = x2;
		ty = y2;
		x2 = x3;
		y2 = y3;
		x3 = tx;
		y3 = ty;
	}
	int cx1, cy1, cx2, cy2, cx3, cy3;
	cx1 = x1 - x2;
	cy1 = y1 - y2;
	cx2 = x2 - x3;
	cy2 = y2 - y3;
	cx3 = x3 - x1;
	cy3 = y3 - y1;

	int px1, py1, px2, py2, px3, py3;
	px1 = px - x1;
	py1 = py - y1;
	px2 = px - x2;
	py2 = py - y2;
	px3 = px - x3;
	py3 = py - y3;

	int s1, s2, s3;
	s1 = cx1 * py1 - cy1 * px1;
	s2 = cx2 * py2 - cy2 * px2;
	s3 = cx3 * py3 - cy3 * px3;

	// filling convention: include edge if the edge is left or up;
	if (cy1 > 0 || (cy1 == 0 && cx1 > 0)) 
		s1 -= 1;
	if (cy2 > 0 || (cy2 == 0 && cx2 > 0))
		s2 -= 1;
	if (cy3 > 0 || (cy3 == 0 && cx3 > 0))
		s3 -= 1;

	return ((s1 < 0) && (s2 < 0) && (s3 < 0));
}

__forceinline__ __device__ float clamp_w(float w, float eps=1e-4)
{
	if (w >= 0 && w < eps)
		return eps;
	else if (w < 0 && w > -eps)
		return -eps;
	else
		return w;
}

__forceinline__ __device__ bool ray_tri_intersection(
	const float3& ray_o,
	const float3& ray_d,
	const float3& p0,
	const float3& p1,
	const float3& p2,
	float3& tuv
)
{
	// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
	bool intersect = false;

	float3 T = ray_o - p0;
	float3 E1 = p1 - p0;
	float3 E2 = p2 - p0;

	float3 P = cross(ray_d, E2);
	float3 Q = cross(T, E1);

	float denom = dot(P, E1);
	if (denom == 0.0f)
		return false;
	float inv_denom = 1.0f / denom;

	tuv.x = dot(Q, E2) * inv_denom;
	tuv.y = dot(P, T) * inv_denom;
	tuv.z = dot(Q, ray_d) * inv_denom;

	// intersect = (tuv.x >= 0.0f && tuv.y >= 0.0f && tuv.z >= 0.0f && tuv.y + tuv.z <= 1.0f);

	return true;
}

__forceinline__ __device__ void ray_tri_intersection_grad(
	const float3& ray_o,
	const float3& ray_d,
	const float3& p0,
	const float3& p1,
	const float3& p2,
	float3& du_dp0,
	float3& du_dp1,
	float3& du_dp2,
	float3& dv_dp0,
	float3& dv_dp1,
	float3& dv_dp2
) {
	float3 T = ray_o - p0;
	float3 E1 = p1 - p0;
	float3 E2 = p2 - p0;

	float denom_sqrt = dot(cross(ray_d, E2), E1);
	float denom = denom_sqrt * denom_sqrt;
	float denom_inv = 1.0f / denom;
	denom = max(denom, 0.0000001f);

	float3 du_dT, du_dE1, du_dE2;
	float3 dv_dT, dv_dE1, dv_dE2;

	float v0 = dot(cross(ray_d, E2), T);
	float v1 = denom_sqrt;
	float v2 = dot(cross(T, E1), E2);

	du_dE1 = (-1 * cross(ray_d, E2) * v0) * denom_inv;
	du_dE2 = (cross(T, ray_d) * v1 - v0 * cross(E1, ray_d)) * denom_inv;
	du_dT = (cross(ray_d, E2) * v1) * denom_inv;

	dv_dE1 = ((cross(E2, T) * v1) - (v2 * cross(ray_d, E2))) * denom_inv;
	dv_dE2 = ((cross(T, E1) * v1) - (v2 * cross(E1, ray_d))) * denom_inv;
	dv_dT = cross(E1, E2) * v1 * denom_inv;

	du_dp0 = -du_dE1 - du_dE2 - du_dT;
	dv_dp0 = -dv_dE1 - dv_dE2 - dv_dT;
	
	du_dp1 = du_dE1;
	dv_dp1 = dv_dE1;

	du_dp2 = du_dE2;
	dv_dp2 = dv_dE2;
}

__forceinline__ __device__ void clamp_bary_uv(float u, float v, float& u_c, float& v_c, int& code)
{
	if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
		u_c = u;
		v_c = v;
		code = 0;
	}
	else if (u <= 0.0f && v <= 0.0f) {
		u_c = 0.0f;
		v_c = 0.0f;
		code = 1;
	}
	else if ((u >= 1.0f && v <= 0.0f) || (v >= 0.0f && v <= u - 1.0f)) {
		u_c = 1.0f;
		v_c = 0.0f;
		code = 2;
	}
	else if ((u <= 0.0f && v >= 1.0f) || (u >= 0.0f && v >= u + 1.0f)) {
		u_c = 0.0f;
		v_c = 1.0f;
		code = 3;
	}
	else if (u <= 0.0f && v <= 1.0f && v >= 0.0f) {
		u_c = 0.0f;
		v_c = v;
		code = 4;
	}
	else if (u <= 1.0f && u >= 0.0f && v <= 0.0f) {
		u_c = u;
		v_c = 0.0f;
		code = 5;
	}
	else {
		u_c = (1.0f + u - v) * 0.5f;
		v_c = (1.0f - u + v) * 0.5f;
		code = 6;
	}
}

__forceinline__ __device__ void clamp_bary_uv_grad(int code, float& duc_du, float& duc_dv, float& dvc_du, float& dvc_dv)
{
	dvc_du = 0.0f;
	duc_dv = 0.0f;
	if (code == 0) {
		duc_du = 1.0f;
		dvc_dv = 1.0f;
	}
	else if (code == 1 || code == 2 || code == 3) {
		duc_du = 0.0f;
		dvc_dv = 0.0f;
	}
	else if (code == 4) {
		duc_du = 0.0f;
		dvc_dv = 1.0f;
	}
	else if (code == 5) {
		duc_du = 1.0f;
		dvc_dv = 0.0f;
	}
	else {
		duc_du = 0.5f;
		dvc_du = -0.5f;
		duc_dv = -0.5f;
		dvc_dv = 0.5f;
	}
}

__forceinline__ __device__ float3 get_vert(const float* verts, int id)
{
	return make_float3(verts[3 * id], verts[3 * id + 1], verts[3 * id + 2]);
}

__forceinline__ __device__ void get_face_vert(
	const float* verts,
	const int* faces,
	int id,
	float3& p0,
	float3& p1,
	float3& p2
)
{
	int v0, v1, v2;
	v0 = faces[3 * id];
	v1 = faces[3 * id + 1];
	v2 = faces[3 * id + 2];
	p0 = get_vert(verts, v0);
	p1 = get_vert(verts, v1);
	p2 = get_vert(verts, v2);
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif