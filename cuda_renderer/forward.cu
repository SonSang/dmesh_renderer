#include "forward.h"
#include "auxiliary.h"
#include "cuda_math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
namespace cg = cooperative_groups;

#include <stdio.h>

namespace TET_FORWARD {
	/*
	============================================================================
	Preprocess Points
	============================================================================
	*/

	// Perform initial steps for each point to rasterization.
	__global__ void preprocessPointCUDA(
		int B,
		int P,
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
		int B,
		int P,
		const float* verts,
		const float* mv_mats,
		const float* proj_mats,
		int W, int H,
		float3* verts_ndc,
		float2* verts_image)
	{
		int BP = B * P;
		preprocessPointCUDA << <(BP + 255) / 256, 256 >> > (
			B,
			P,
			verts,
			mv_mats,
			proj_mats,
			W, H,
			verts_ndc,
			verts_image);
	}

	/*
	============================================================================
	Generate rays
	============================================================================
	*/

	__global__ void setup_curand_kernel(curandState *state, int max_idx, int seed){

		int idx = cg::this_grid().thread_rank();
		if (idx >= max_idx)
			return;
		curand_init(seed, idx, 0, &state[idx]);
	}

	__global__ void generateRaysCUDA(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d,
		curandState* curand_state
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
		if (curand_state) {
			pixf.x = pixel_x - 0.5f + (0.5f * curand_uniform(&curand_state[idx]));
			pixf.y = pixel_y - 0.5f + (0.5f * curand_uniform(&curand_state[idx]));
		}
		else {
			pixf.x = pixel_x + 0.5f;
			pixf.y = pixel_y + 0.5f;
		}

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
		float len = sqrtf(dot(ray_d[idx], ray_d[idx]));
		len = max(len, 0.0001f);	// Avoid division by zero;
		ray_d[idx] = ray_d[idx] / len;
	}

	void generate_rays(
		const float* inv_mv_mats,
		const float* inv_proj_mats,
		const int B, int W, int H,
		float3* ray_o,
		float3* ray_d,
		int random_seed)
	{
		curandState* curand_state = nullptr;
		if (random_seed > 0) {
			// Initialize random number generator;
			cudaMalloc(&curand_state, B * W * H * sizeof(curandState));
			setup_curand_kernel << <(B * W * H + 255) / 256, 256 >> > (curand_state, B * W * H, random_seed);
		}
		generateRaysCUDA << <(B * W * H + 255) / 256, 256 >> > (
			inv_mv_mats,
			inv_proj_mats,
			B, W, H,
			ray_o,
			ray_d,
			curand_state);
		cudaFree(curand_state);
	}

	/*
	============================================================================
	Preprocess faces
	============================================================================
	*/

	// Perform initial steps for each triangle to rasterization.
	__global__ void preprocessFaceCUDA(int B, int P, int F,
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

		min_depths[idx] = (min_z + 1.0f) * 0.5f;
		if (min_depths[idx] < 0.0f) min_depths[idx] = 0.0f;
		if (min_depths[idx] > 1.0f) min_depths[idx] = 1.0f;

		max_depths[idx] = (max_z + 1.0f) * 0.5f;
		if (max_depths[idx] < 0.0f) max_depths[idx] = 0.0f;
		if (max_depths[idx] > 1.0f) max_depths[idx] = 1.0f;
	}

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
		uint32_t* tiles_touched)
	{
		preprocessFaceCUDA << <(B * F + 255) / 256, 256 >> > (
			B, P, F,
			verts,
			verts_ndc,
			verts_image,
			faces,
			mv_mats,
			proj_mats,
			W, H,
			depths,
			min_depths,
			max_depths,
			grid,
			tiles_touched);
	}

	/*
	============================================================================
	First intersecting face and tet
	============================================================================
	*/
	// Find first intersection info.
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	firstIntersectCUDA(
		const float* __restrict__ verts,
		const int* __restrict__ faces,
		const int* __restrict__ tets,
		const int* __restrict__ face_tets,
		const float* __restrict__ faces_min_depth,
		const float* __restrict__ faces_max_depth,
		const uint2* __restrict__ ranges,
		const uint32_t* __restrict__ face_list,
		int B, int F, int W, int H,
		const float3* __restrict__ ray_o,
		const float3* __restrict__ ray_d,
		int* __restrict__ first_face,
		int* __restrict__ first_tet)
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;
		
		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;
		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		uint32_t pix_id = W * pix.y + pix.x;
		auto pix_batch_id = (batch_id * W * H) + pix_id;

		float3 this_ray_o = ray_o[pix_batch_id];
		float3 this_ray_d = ray_d[pix_batch_id];
		first_face[pix_batch_id] = -1;
		first_tet[pix_batch_id] = -1;		// If there is no corresponding info, remain -1;
		
		// Check if this thread is associated with a valid pixel or outside.
		bool inside = pix.x < W && pix.y < H;
		// Done threads can help with fetching, but don't rasterize
		bool done = !inside;

		// Load start/end range of IDs to process in bit sorted list.
		auto tile_id = (batch_id * horizontal_blocks * vertical_blocks) + 
						(block.group_index().y * horizontal_blocks + block.group_index().x);
		uint2 range = ranges[tile_id];
		const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int toDo = range.y - range.x;

		// Allocate storage for batches of collectively fetched data.
		__shared__ int collected_face_id[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_0[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_1[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_2[BLOCK_SIZE];
		__shared__ float collected_face_min_depth[BLOCK_SIZE];
		__shared__ float collected_face_max_depth[BLOCK_SIZE];

		// Initialize helper variables
		float min_T = -1.0f;
		float min_T_max_depth = -1.0f;		// max depth of the face that has the min_T;


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
				int face_batch_id = batch_id * F + face_id;

				collected_face_id[block.thread_rank()] = face_id;

				int vert_id = faces[3 * face_id + 0];
				collected_face_vert_0[block.thread_rank()] = { verts[3 * vert_id + 0], verts[3 * vert_id + 1], verts[3 * vert_id + 2] };
				vert_id = faces[3 * face_id + 1];
				collected_face_vert_1[block.thread_rank()] = { verts[3 * vert_id + 0], verts[3 * vert_id + 1], verts[3 * vert_id + 2] };
				vert_id = faces[3 * face_id + 2];
				collected_face_vert_2[block.thread_rank()] = { verts[3 * vert_id + 0], verts[3 * vert_id + 1], verts[3 * vert_id + 2] };
				
				collected_face_min_depth[block.thread_rank()] = faces_min_depth[face_batch_id];
				collected_face_max_depth[block.thread_rank()] = faces_max_depth[face_batch_id];
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				if (min_T >= 0.0f && collected_face_min_depth[j] > min_T_max_depth) {
					done = true;
					continue;
				}

				// Compute barycentric coordinates and ray parameter (t) at intersection point;
				bool intersect;
				float3 tuv;
				intersect = ray_tri_intersection(
					this_ray_o,
					this_ray_d,
					collected_face_vert_0[j],
					collected_face_vert_1[j],
					collected_face_vert_2[j],
					tuv
				);

				// If no intersection, skip this face;
				if (!intersect)
					continue;

				float curr_T = tuv.x;
				if (min_T < 0.0f || curr_T < min_T) {
					min_T = curr_T;
					min_T_max_depth = collected_face_max_depth[j];

					first_face[pix_batch_id] = collected_face_id[j];
				}
			}
		}

		// Identify the first tet that contains the first face;
		int this_first_face = first_face[pix_batch_id];

		if (this_first_face < 0)
			return;

		for (int i = 0; i < 2; i++) {
			int tet_id = face_tets[2 * this_first_face + i];
			if (tet_id < 0)
				continue;
			
			float3 my_first_face_outward_normal;
			tet_face_outward_normal(
				verts,
				faces,
				tets,
				this_first_face,
				tet_id,
				my_first_face_outward_normal
			);

			float dot_prod = dot(my_first_face_outward_normal, this_ray_d);
			
			if (dot_prod < 0.0f)
				first_tet[pix_batch_id] = tet_id;
		}
	}

	void first_intersect(
		const dim3 batch_grid, dim3 block,
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
		int* first_tet)
	{
		firstIntersectCUDA << <batch_grid, block >> > (
			verts,
			faces,
			tets,
			face_tets,
			faces_min_depth,
			faces_max_depth,
			ranges,
			face_list,
			B, F, W, H,
			ray_o,
			ray_d,
			first_face,
			first_tet);
	}

	/*
	============================================================================
	Main Rendering Kernel using Ray Marching
	============================================================================
	*/

	template <uint32_t CHANNELS>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
		const float* __restrict__ verts,
		const int* __restrict__ faces,
		const float* __restrict__ verts_color,
		const float* __restrict__ faces_opacity,

		const float* __restrict__ mv_mats,
		const float* __restrict__ proj_mats,
		const float* __restrict__ verts_depth,
		const float* __restrict__ faces_intense,

		const int* __restrict__ tets,
		const int* __restrict__ face_tets,
		const int* __restrict__ tet_faces,

		const float3* __restrict__ ray_o,
		const float3* __restrict__ ray_d,
		const int* __restrict__ first_face,
		const int* __restrict__ first_tet,

		int W, int H,
		int P, int F,
		
		const float* __restrict__ bg_color,
		
		float* __restrict__ final_log_T,
		float* __restrict__ final_prev_log_T,
		int* __restrict__ last_face,
		int* __restrict__ last_tet,
		uint32_t* __restrict__ n_contrib,
		bool* __restrict__ is_active,

		float* __restrict__ out_color,
		float* __restrict__ out_depth,
		float* __restrict__ out_active)
	{
		// Identify current tile and associated min/max pixel range.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;

		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;

		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		uint32_t pix_id = W * pix.y + pix.x;
		uint32_t pix_batch_id = (batch_id * W * H) + pix_id;
		
		// Check if this thread is associated with a valid pixel or outside.
		bool inside = pix.x < W && pix.y < H;

		// If outside, do nothing;
		if (!inside)
			return;
		bool done = false;

		// ================
		// Ray marching
		// ================
		float3 this_ray_o = ray_o[pix_batch_id];
		float3 this_ray_d = ray_d[pix_batch_id];
		int this_first_face = first_face[pix_batch_id];
		int this_first_tet = first_tet[pix_batch_id];

		const float* curr_mv = &mv_mats[batch_id * 16];
		const float* curr_proj = &proj_mats[batch_id * 16];

		float rt = 0.0f;
		float iu, iv;
		if (this_first_face == -1 || this_first_tet == -1)
			done = true;
		else {
			// compute [rt], the ray param at the first intersection point;
			float3 tuv, p0, p1, p2;
			get_face_vert(verts, faces, this_first_face, p0, p1, p2);
			ray_tri_intersection(
				this_ray_o,
				this_ray_d,
				p0,
				p1, 
				p2,
				tuv
			);
			rt = tuv.x;
			iu = tuv.y;
			iv = tuv.z;
		}

		// accumulate rendering results before integrating bg;
		float3 C = { 0.0f, 0.0f, 0.0f };
		float D = 0.0f;
		
		// accumulated transmittance;
		float log_T = 0.0f;
		float prev_log_T = 0.0f;
		int this_last_face = -1;
		int this_last_tet = -1;
		bool this_is_active = false;
		uint32_t this_n_contrib = 0;

		// current tet info;
		int curr_face = this_first_face;
		int curr_tet = this_first_tet;
		float curr_rt = rt;
		float curr_iu = iu;
		float curr_iv = iv;

		while(!done) {
			/*
			1. Render using current face
			*/

			// gather colors encoded in vertices (non-batch);
			float3 curr_verts_color_0, curr_verts_color_1, curr_verts_color_2;
			get_face_vert_color(
				verts_color,
				faces,
				curr_face,
				curr_verts_color_0,
				curr_verts_color_1,
				curr_verts_color_2
			);

			// interpolate color;
			float3 curr_face_color = (curr_verts_color_0 + 
										(curr_verts_color_1 - curr_verts_color_0) * curr_iu + 
										(curr_verts_color_2 - curr_verts_color_0) * curr_iv);

			// get opacity of current face (non-batch);
			float curr_face_opacity = faces_opacity[curr_face];

			// Retrieve this face's intensity (batch);
			float curr_face_intense = faces_intense[batch_id * F + curr_face];

			// Update color;
			curr_face_color = curr_face_color * curr_face_intense;
			float tmp_T = expf(log_T);
			C += tmp_T * curr_face_opacity * curr_face_color;
			
			// Update depth;
			float3 curr_point = this_ray_o + (this_ray_d * curr_rt);
			float4 curr_point_ndc = transformPoint4x4(transformPoint4x3(curr_point, curr_mv), curr_proj);
			float curr_point_w = 1.0f / clamp_w(curr_point_ndc.w);
			float curr_point_depth = curr_point_ndc.z * curr_point_w;
			D += tmp_T * curr_face_opacity * curr_point_depth;

			// Update transmittance;
			prev_log_T = log_T;
			if (curr_face_opacity < 1.0f)
				log_T += logf(1.0f - curr_face_opacity);
			else {
				// if opacity is 1.0, log is impossible;
				// make [log_T] very small so that we finish this loop;
				log_T = logf(T_EPS * 0.1f);
			}	

			// if log_T is too small, stop marching;
			if (expf(log_T) < T_EPS) {
				done = true;
				this_is_active = true;
			}

			// update history;
			this_n_contrib++;
			this_last_face = curr_face;
			this_last_tet = curr_tet;
				
			/*
			2. Find next face
			*/
			
			// Compute intersection point between the ray and
			// the remaining 3 faces in the current tet;

			int next_face = -1;
			int next_tet = -1;
			float next_rt, next_iu, next_iv;

			// If there is no current tet to explore, we are done;
			if (curr_tet == -1) {
				this_is_active = true;
				done = true;
			}
			
			if (!done) {
				int curr_tet_faces[4];
				curr_tet_faces[0] = tet_faces[4 * curr_tet + 0];
				curr_tet_faces[1] = tet_faces[4 * curr_tet + 1];
				curr_tet_faces[2] = tet_faces[4 * curr_tet + 2];
				curr_tet_faces[3] = tet_faces[4 * curr_tet + 3];

				int curr_tet_other_faces[3];
				int cnt = 0;
				for (int i = 0; i < 4; i++) {
					if (curr_tet_faces[i] == curr_face)
						continue;
					curr_tet_other_faces[cnt++] = curr_tet_faces[i];
				}

				if (cnt != 3) {
					// it should not happen, but we can't believe numerics...
					done = true;
					// printf("Error case 1\n");
				}

				/*
				Among three other faces in the current tet,
				find the one that intersects with the ray,
				and its outward normal is in the same direction
				as the ray direction;
				We do not use [rt] here, because it is weak to
				numerical errors;
				*/

				// if curr face's outward normal was in the same 
				// direction as the ray, error;
				float3 curr_face_outward_normal;
				tet_face_outward_normal(
					verts,
					faces,
					tets,
					curr_face,
					curr_tet,
					curr_face_outward_normal
				);
				float curr_face_normal_dot_prod = dot(curr_face_outward_normal, this_ray_d);
				if (curr_face_normal_dot_prod >= 0.0f) {
					done = true;
					// printf("Error case 2\n");
				}

				int next_face_cnt = 0;
				for (int i = 0; i < cnt; i++) {
					float3 p0, p1, p2;
					get_face_vert(verts, faces, curr_tet_other_faces[i], p0, p1, p2);
					
					float3 curr_tet_other_face_tuv;
					float3 curr_tet_other_face_outward_normal;
					bool curr_tet_other_face_intersect = ray_tri_intersection(
						this_ray_o,
						this_ray_d,
						p0,
						p1,
						p2,
						curr_tet_other_face_tuv
					);
					tet_face_outward_normal(
						verts,
						faces,
						tets,
						curr_tet_other_faces[i],
						curr_tet,
						curr_tet_other_face_outward_normal
					);
					float curr_tet_other_face_normal_dot_prod = dot(curr_tet_other_face_outward_normal, this_ray_d);

					if (curr_tet_other_face_intersect && curr_tet_other_face_normal_dot_prod > 0.0f) {
						next_face = curr_tet_other_faces[i];
						next_rt = curr_tet_other_face_tuv.x;
						next_iu = curr_tet_other_face_tuv.y;
						next_iv = curr_tet_other_face_tuv.z;
						next_face_cnt++;
					}
				}

				// In edge case, there could be multiple intersecting faces,
				// but generally, there should be only one intersecting face.
				if (next_face_cnt != 1) {
					// it should not happen, but we can't believe numerics...
					done = true;
					// printf("Error case 3\n");
				}
				else {
					for (int i = 0; i < 2; i++) {
						int p_next_tet = face_tets[2 * next_face + i];
						if (p_next_tet == curr_tet || p_next_tet == -1)
							continue;
						next_tet = p_next_tet;
						break;
					}
				}

				curr_face = next_face;
				curr_tet = next_tet;
				curr_rt = next_rt;
				curr_iu = next_iu;
				curr_iv = next_iv;
			}
			
			// Debugging...
			
			// if (block.group_index().z == 3 && block.group_index().x == 7 && block.group_index().y == 7 &&
			// 	block.thread_index().x == 8 && block.thread_index().y == 8) {
			// 	printf("n_contrib: %d\n", this_n_contrib);
			// 	printf("curr_tet: %d\n", curr_tet);
			// 	printf("curr_face: %d\n", curr_face);
			// 	printf("next_face: %d\n", next_face);
			// 	printf("curr_face: %d %d %d\n", faces[3 * curr_face + 0], faces[3 * curr_face + 1], faces[3 * curr_face + 2]);
			// 	printf("curr_face_alpha: %f\n", curr_face_opacity);
			// }
		}

		// Write back results;
		final_log_T[pix_batch_id] = log_T;
		final_prev_log_T[pix_batch_id] = prev_log_T;
		last_face[pix_batch_id] = this_last_face;
		last_tet[pix_batch_id] = this_last_tet;
		n_contrib[pix_batch_id] = this_n_contrib;
		is_active[pix_batch_id] = this_is_active;

		// Integrate background color.
		if (this_is_active) {
			float final_T = expf(log_T);
			out_color[(batch_id * 3 * H * W) + (0 * H * W) + pix_id] = C.x + final_T * bg_color[0];
			out_color[(batch_id * 3 * H * W) + (1 * H * W) + pix_id] = C.y + final_T * bg_color[1];
			out_color[(batch_id * 3 * H * W) + (2 * H * W) + pix_id] = C.z + final_T * bg_color[2];
			out_depth[batch_id * H * W + pix_id] = D + final_T * 1.0f;
			out_active[batch_id * H * W + pix_id] = 1.0f;
		}
		else {
			// if there was error, fill with bg color;
			out_color[(batch_id * 3 * H * W) + (0 * H * W) + pix_id] = bg_color[0];
			out_color[(batch_id * 3 * H * W) + (1 * H * W) + pix_id] = bg_color[1];
			out_color[(batch_id * 3 * H * W) + (2 * H * W) + pix_id] = bg_color[2];
			out_depth[batch_id * H * W + pix_id] = 1.0f;
			out_active[batch_id * H * W + pix_id] = 0.0f;
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
		float* out_active)
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
			
			tets,
			face_tets,
			tet_faces,
			
			ray_o,
			ray_d,
			first_face,
			first_tet,
			
			W, H,
			P, F,
			
			bg_color,
			
			final_log_T,
			final_prev_log_T,
			last_face,
			last_tet,
			n_contrib,
			is_active,

			out_color,
			out_depth,
			out_active);
	}
}