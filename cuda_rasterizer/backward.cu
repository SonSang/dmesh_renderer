#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace TRI_BACKWARD {
	// Backward version of the rendering procedure.
	template <uint32_t CHANNELS>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
		const int B, int P, int F,

		const float* __restrict__ verts,
		const int* __restrict__ faces,
		const float* __restrict__ verts_color,
		const float* __restrict__ faces_opacity,

		const float* __restrict__ mv_mats,
		const float* __restrict__ proj_mats,
		const float* __restrict__ inv_mv_mats,
		const float* __restrict__ inv_proj_mats,

		const float* __restrict__ verts_depth,
		const float* __restrict__ faces_intense,

		const uint2* __restrict__ ranges,
		const float3* __restrict__ ray_o,
		const float3* __restrict__ ray_d,
		const uint32_t* __restrict__ face_list,
		int W, int H,
		const float* __restrict__ bg_color,
		const float2* __restrict__ verts_image,
		const float* __restrict__ final_Ts,
		const float* __restrict__ final_prev_Ts,
		const uint32_t* __restrict__ n_contrib,
		
		const float* __restrict__ dL_dpix_color,
		const float* __restrict__ dL_dpix_depth,

		float* __restrict__ dL_dverts,
		float* __restrict__ dL_dvcolor,
		float* __restrict__ dL_dfopacity,
		float* __restrict__ dL_dvdepth,
		float* __restrict__ dL_dfintense)
	{
		// We rasterize again. Compute necessary block info.
		auto block = cg::this_thread_block();
		auto batch_id = block.group_index().z;

		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;

		uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		float2 pixf = { (float)pix.x, (float)pix.y };

		uint32_t pix_id = W * pix.y + pix.x;
		uint32_t b_pix_id = (batch_id * W * H) + pix_id;
		
		// Check if this thread is associated with a valid pixel or outside.
		bool inside = batch_id < B && pix.x < W && pix.y < H;
		// Done threads can help with fetching, but don't rasterize
		bool done = !inside;

		float3 this_ray_o, this_ray_d;
		if (inside) {
			this_ray_o = ray_o[b_pix_id];
			this_ray_d = ray_d[b_pix_id];
		}

		// Load start/end range of IDs to process in bit sorted list.
		int b_tile_id = (batch_id * horizontal_blocks * vertical_blocks) + 
							(block.group_index().y * horizontal_blocks) + block.group_index().x;
		uint2 range = ranges[b_tile_id];
		const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int toDo = range.y - range.x;

		// Load auxiliary data into shared memory
		__shared__ int collected_face_id[BLOCK_SIZE];

		__shared__ int collected_face_vert_id_0[BLOCK_SIZE];
		__shared__ int collected_face_vert_id_1[BLOCK_SIZE];
		__shared__ int collected_face_vert_id_2[BLOCK_SIZE];

		__shared__ float3 collected_face_vert_0[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_1[BLOCK_SIZE];
		__shared__ float3 collected_face_vert_2[BLOCK_SIZE];

		__shared__ float2 collected_face_vert_image_0[BLOCK_SIZE];
		__shared__ float2 collected_face_vert_image_1[BLOCK_SIZE];
		__shared__ float2 collected_face_vert_image_2[BLOCK_SIZE];

		__shared__ float collected_face_vert_color_0[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_1[BLOCK_SIZE * CHANNELS];
		__shared__ float collected_face_vert_color_2[BLOCK_SIZE * CHANNELS];

		__shared__ float collected_face_vert_depth_0[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_1[BLOCK_SIZE];
		__shared__ float collected_face_vert_depth_2[BLOCK_SIZE];
		
		__shared__ float collected_face_opacity[BLOCK_SIZE];
		__shared__ float collected_face_intense[BLOCK_SIZE];

		// In the forward, we stored the final value for T, the
		// product of all (1 - alpha) factors. 
		const float T_final = inside ? final_Ts[b_pix_id] : 0;
		const float prev_T_final = inside ? final_prev_Ts[b_pix_id] : 0;
		float T = prev_T_final;
		bool T_first_pass = true;

		// We start from the back. The ID of the last contributing
		// face is known from each pixel from the forward.
		uint32_t contributor = toDo;
		const int last_contributor = inside ? n_contrib[b_pix_id] : 0;

		float accum_rec[CHANNELS] = { 0 };
		float accum_recd[1] = { 0 };
		float dL_dpixel_color[CHANNELS];
		float dL_dpixel_depth[1];
		if (inside) {
			for (int i = 0; i < CHANNELS; i++)
				dL_dpixel_color[i] = dL_dpix_color[(batch_id * CHANNELS * H * W) + (i * H * W) + pix_id];
			dL_dpixel_depth[0] = dL_dpix_depth[(batch_id * H * W) + pix_id];
		}
			
		float last_alpha = 0;
		float last_color[CHANNELS] = { 0 };
		float last_depth[1] = { 0 }; 

		// Traverse all Faces
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// Load auxiliary data into shared memory, start in the BACK
			// and load them in reverse order.
			block.sync();
			const int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				const int face_id = face_list[range.y - progress - 1];	// start from BACK
				const int b_face_id = batch_id * F + face_id;

				collected_face_id[block.thread_rank()] = face_id;

				int face_vert_id_0 = faces[3 * face_id + 0];
				int face_vert_id_1 = faces[3 * face_id + 1];
				int face_vert_id_2 = faces[3 * face_id + 2];

				collected_face_vert_id_0[block.thread_rank()] = face_vert_id_0;
				collected_face_vert_id_1[block.thread_rank()] = face_vert_id_1;
				collected_face_vert_id_2[block.thread_rank()] = face_vert_id_2;

				int b_face_vert_id_0 = batch_id * P + face_vert_id_0;
				int b_face_vert_id_1 = batch_id * P + face_vert_id_1;
				int b_face_vert_id_2 = batch_id * P + face_vert_id_2;

				get_face_vert(
					verts,
					faces,
					face_id,
					collected_face_vert_0[block.thread_rank()],
					collected_face_vert_1[block.thread_rank()],
					collected_face_vert_2[block.thread_rank()]
				);

				collected_face_vert_image_0[block.thread_rank()] = verts_image[b_face_vert_id_0];
				collected_face_vert_image_1[block.thread_rank()] = verts_image[b_face_vert_id_1];
				collected_face_vert_image_2[block.thread_rank()] = verts_image[b_face_vert_id_2];

				for (int ch = 0; ch < CHANNELS; ch++)
				{
					collected_face_vert_color_0[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_0 * CHANNELS + ch];
					collected_face_vert_color_1[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_1 * CHANNELS + ch];
					collected_face_vert_color_2[block.thread_rank() * CHANNELS + ch] = verts_color[face_vert_id_2 * CHANNELS + ch];
				}

				collected_face_vert_depth_0[block.thread_rank()] = verts_depth[b_face_vert_id_0];
				collected_face_vert_depth_1[block.thread_rank()] = verts_depth[b_face_vert_id_1];
				collected_face_vert_depth_2[block.thread_rank()] = verts_depth[b_face_vert_id_2];

				collected_face_opacity[block.thread_rank()] = faces_opacity[face_id];
				collected_face_intense[block.thread_rank()] = faces_intense[b_face_id];
			}
			block.sync();

			// Iterate over faces
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				// Keep track of current face ID. Skip, if this one
				// is behind the last contributor for this pixel.
				contributor--;
				if (contributor >= last_contributor)
					continue;

				// Find out if the current face overlaps the current pixel;
				// if not, skip it, as before;
				pixf.x = pix.x + 0.5f;
				pixf.y = pix.y + 0.5f;
				if (!in_tri(pixf, 
						collected_face_vert_image_0[j], 
						collected_face_vert_image_1[j], 
						collected_face_vert_image_2[j]))
					continue;

				// Find intersection point between ray and triangle as forward;
				const float3& v0 = collected_face_vert_0[j];
				const float3& v1 = collected_face_vert_1[j];
				const float3& v2 = collected_face_vert_2[j];
				float3 tuv = { 0, 0, 0 };
				bool not_edge_case = ray_tri_intersection(
					this_ray_o, this_ray_d,
					v0, v1, v2, tuv);

				if (!not_edge_case)
					continue;

				// clamp, because (iu, iv) could be outside of the triangle.
				float iu = tuv.y, iv = tuv.z;
				float iuc, ivc;
				int iclamp_code;
				clamp_bary_uv(iu, iv, iuc, ivc, iclamp_code);	
				float i0 = 1 - iuc - ivc, i1 = iuc, i2 = ivc;
				
				// Find intersection point's color and depth;
				float iC[CHANNELS] = { 0 };
				float iD[1] = { 0 };
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					iC[ch] = (i0 * collected_face_vert_color_0[j * CHANNELS + ch] +
							i1 * collected_face_vert_color_1[j * CHANNELS + ch] +
							i2 * collected_face_vert_color_2[j * CHANNELS + ch])
							* collected_face_intense[j];
				}
				iD[0] = i0 * collected_face_vert_depth_0[j] +
						i1 * collected_face_vert_depth_1[j] +
						i2 * collected_face_vert_depth_2[j];

				// Find intersecting face's opacity;
				float alpha = collected_face_opacity[j];
				
				// Recover previous T
				// at first pass, T is already prev T before the end;
				if (!T_first_pass) {
					// (1.0 - alpha) cannot be 0.0, because if it was,
					// it would have been dealt with in the first pass;
					T = T / (1.f - alpha);
					if (isnan(T) || isinf(T)) {
						printf("[FASTDIFFRAST]: T is nan or inf, which should not happen\n");
					}
				}
				T_first_pass = false;

				/*
				Compute gradients of iC, iD, and alpha w.r.t. loss.
				*/
				float curr_dL_dicolor[CHANNELS] = { 0 };
				float curr_dL_didepth[1] = { 0 };
				float curr_dL_dalpha = 0.0f;
				
				// color
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					const float c = iC[ch];
					// Update last color (to be used in the next iteration)
					accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = c;

					const float dL_dchannel = dL_dpixel_color[ch];
					curr_dL_dicolor[ch] = dL_dchannel * alpha * T;
					curr_dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				}

				// depth;
				for (int ch = 0; ch < 1; ch++) {
					const float c = iD[ch];
					// Update last depth (to be used in the next iteration)
					accum_recd[ch] = last_alpha * last_depth[ch] + (1.f - last_alpha) * accum_recd[ch];
					last_depth[ch] = c;

					const float dL_dchannel = dL_dpixel_depth[ch];
					curr_dL_didepth[ch] = dL_dchannel * alpha * T;
					curr_dL_dalpha += (c - accum_recd[ch]) * dL_dchannel;
				}

				// alpha: additional contribution from the background;
				curr_dL_dalpha *= T;
				// Update last alpha (to be used in the next iteration)
				last_alpha = alpha;

				// Account for fact that alpha also influences how much of
				// the background color is added if nothing left to blend
				float bg_dot_dpixel = 0;
				float bd_dot_dpixel = 0;
				for (int i = 0; i < CHANNELS; i++)
					bg_dot_dpixel += bg_color[i] * dL_dpixel_color[i];
				for (int i = 0; i < 1; i++)
					bd_dot_dpixel += 1.0 * dL_dpixel_depth[i];
				if (alpha == 1.0f) {
					// in this case, (-T_final / (1.f - alpha)) is (-prev_T_final),
					// because when alpha == 1.0, it would have been the last step;
					curr_dL_dalpha += (-prev_T_final) * bg_dot_dpixel;
					curr_dL_dalpha += (-prev_T_final) * bd_dot_dpixel;
				}
				else {
					curr_dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
					curr_dL_dalpha += (-T_final / (1.f - alpha)) * bd_dot_dpixel;
				}

				/*
				Compute gradients of (i0, i1, i2), verts color, faces opacity, faces intense.
				*/
				float curr_dL_di0 = 0, curr_dL_di1 = 0, curr_dL_di2 = 0;
				
				float curr_dL_dvcolor_0[CHANNELS] = { 0 };
				float curr_dL_dvcolor_1[CHANNELS] = { 0 };
				float curr_dL_dvcolor_2[CHANNELS] = { 0 };
				
				float curr_dL_dvdepth_0[1] = { 0 };
				float curr_dL_dvdepth_1[1] = { 0 };
				float curr_dL_dvdepth_2[1] = { 0 };

				float curr_dL_dfopacity = curr_dL_dalpha;
				float curr_dL_dfintense = 0;

				// color
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					curr_dL_di0 += collected_face_vert_color_0[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_di1 += collected_face_vert_color_1[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_di2 += collected_face_vert_color_2[j * CHANNELS + ch] * curr_dL_dicolor[ch] * collected_face_intense[j];

					curr_dL_dvcolor_0[ch] += i0 * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_dvcolor_1[ch] += i1 * curr_dL_dicolor[ch] * collected_face_intense[j];
					curr_dL_dvcolor_2[ch] += i2 * curr_dL_dicolor[ch] * collected_face_intense[j];

					curr_dL_dfintense += (i0 * collected_face_vert_color_0[j * CHANNELS + ch] +
									i1 * collected_face_vert_color_1[j * CHANNELS + ch] +
									i2 * collected_face_vert_color_2[j * CHANNELS + ch]) * curr_dL_dicolor[ch];
				}

				// depth
				curr_dL_di0 += collected_face_vert_depth_0[j] * curr_dL_didepth[0];
				curr_dL_di1 += collected_face_vert_depth_1[j] * curr_dL_didepth[0];
				curr_dL_di2 += collected_face_vert_depth_2[j] * curr_dL_didepth[0];

				curr_dL_dvdepth_0[0] += i0 * curr_dL_didepth[0];
				curr_dL_dvdepth_1[0] += i1 * curr_dL_didepth[0];
				curr_dL_dvdepth_2[0] += i2 * curr_dL_didepth[0];

				/*
				Compute gradients of verts w.r.t. loss.
				*/
				float di0_diuc = -1, di0_divc = -1;
				float di1_diuc = 1, di1_divc = 0;
				float di2_diuc = 0, di2_divc = 1;

				float diuc_diu, diuc_div, divc_diu, divc_div;
				clamp_bary_uv_grad(iclamp_code, diuc_diu, diuc_div, divc_diu, divc_div);

				float di0_diu = di0_diuc * diuc_diu + di0_divc * divc_diu;
				float di0_div = di0_diuc * diuc_div + di0_divc * divc_div;
				float di1_diu = di1_diuc * diuc_diu + di1_divc * divc_diu;
				float di1_div = di1_diuc * diuc_div + di1_divc * divc_div;
				float di2_diu = di2_diuc * diuc_diu + di2_divc * divc_diu;
				float di2_div = di2_diuc * diuc_div + di2_divc * divc_div;

				float curr_dL_diu = curr_dL_di0 * di0_diu + curr_dL_di1 * di1_diu + curr_dL_di2 * di2_diu;
				float curr_dL_div = curr_dL_di0 * di0_div + curr_dL_di1 * di1_div + curr_dL_di2 * di2_div;

				float3 curr_diu_dp0, curr_diu_dp1, curr_diu_dp2;
				float3 curr_div_dp0, curr_div_dp1, curr_div_dp2;
				ray_tri_intersection_grad(
					this_ray_o, 
					this_ray_d,
					v0, v1, v2,
					curr_diu_dp0, curr_diu_dp1, curr_diu_dp2,
					curr_div_dp0, curr_div_dp1, curr_div_dp2);

				float3 curr_dL_dp0 = curr_dL_diu * curr_diu_dp0 + curr_dL_div * curr_div_dp0;
				float3 curr_dL_dp1 = curr_dL_diu * curr_diu_dp1 + curr_dL_div * curr_div_dp1;
				float3 curr_dL_dp2 = curr_dL_diu * curr_diu_dp2 + curr_dL_div * curr_div_dp2;

				/*
				Aggregate gradients.
				*/

				// verts;
				atomicAdd(&(dL_dverts[collected_face_vert_id_0[j] * 3 + 0]), curr_dL_dp0.x);
				atomicAdd(&(dL_dverts[collected_face_vert_id_0[j] * 3 + 1]), curr_dL_dp0.y);
				atomicAdd(&(dL_dverts[collected_face_vert_id_0[j] * 3 + 2]), curr_dL_dp0.z);

				atomicAdd(&(dL_dverts[collected_face_vert_id_1[j] * 3 + 0]), curr_dL_dp1.x);
				atomicAdd(&(dL_dverts[collected_face_vert_id_1[j] * 3 + 1]), curr_dL_dp1.y);
				atomicAdd(&(dL_dverts[collected_face_vert_id_1[j] * 3 + 2]), curr_dL_dp1.z);

				atomicAdd(&(dL_dverts[collected_face_vert_id_2[j] * 3 + 0]), curr_dL_dp2.x);
				atomicAdd(&(dL_dverts[collected_face_vert_id_2[j] * 3 + 1]), curr_dL_dp2.y);
				atomicAdd(&(dL_dverts[collected_face_vert_id_2[j] * 3 + 2]), curr_dL_dp2.z);

				// verts color;
				for (int k = 0; k < 3; k++)
				{
					atomicAdd(&(dL_dvcolor[collected_face_vert_id_0[j] * CHANNELS + k]), curr_dL_dvcolor_0[k]);
					atomicAdd(&(dL_dvcolor[collected_face_vert_id_1[j] * CHANNELS + k]), curr_dL_dvcolor_1[k]);
					atomicAdd(&(dL_dvcolor[collected_face_vert_id_2[j] * CHANNELS + k]), curr_dL_dvcolor_2[k]);
				}

				// verts depth;
				atomicAdd(&(dL_dvdepth[(batch_id * P) + collected_face_vert_id_0[j]]), curr_dL_dvdepth_0[0]);
				atomicAdd(&(dL_dvdepth[(batch_id * P) + collected_face_vert_id_1[j]]), curr_dL_dvdepth_1[0]);
				atomicAdd(&(dL_dvdepth[(batch_id * P) + collected_face_vert_id_2[j]]), curr_dL_dvdepth_2[0]);

				// faces opacity;
				atomicAdd(&(dL_dfopacity[collected_face_id[j]]), curr_dL_dfopacity);

				// faces intense;
				atomicAdd(&(dL_dfintense[(batch_id * F) + collected_face_id[j]]), curr_dL_dfintense);
			}
		}
	}

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
		float* dL_dfintense)
	{
		renderCUDA<NUM_CHANNELS> << <grid, block >> >(
			B, P, F,

			verts, faces, verts_color, faces_opacity,

			mv_mats, proj_mats, inv_mv_mats, inv_proj_mats,
			verts_depth, faces_intense,

			ranges, ray_o, ray_d, face_list,
			W, H,

			bg_color, verts_image, final_Ts, final_prev_Ts, n_contrib,

			dL_dpix_color, dL_dpix_depth,

			dL_dverts, dL_dvcolor, dL_dfopacity, dL_dvdepth, dL_dfintense
		);
	}
}

