#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace TET_BACKWARD {
	__forceinline__ __device__ void compute_dL_dalpha(
		const float final_log_T,
		const float* bg_color,

		const float log_T,
		const float alpha,
		const float last_alpha,

		const float* color,
		const float* last_color,
		const float* accum_rec,
		
		const float depth,
		const float last_depth,
		const float accum_rec_depth,
		
		// incoming gradient;
		const float* dL_dpix_color,
		const float dL_dpix_depth,
		
		// variables that we should update;
		float* n_last_alpha,
		float* n_log_T,
		
		float* n_last_color,
		float* n_accum_rec,
		
		float* n_last_depth,
		float* n_accum_rec_depth,
		
		// outgoing gradient;
		float* dL_dalpha,
		int C
	) {
		float prev_log_T = log_T - logf(1.0f - alpha);
		float prev_T = expf(prev_log_T);
		dL_dalpha[0] = 0.0f;

		// COLOR
		for (int ch = 0; ch < C; ch++)
		{
			const float c = color[ch];
			// Update last color (to be used in the next iteration)
			n_accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
			n_last_color[ch] = color[ch];

			const float dL_dchannel = dL_dpix_color[ch];
			dL_dalpha[0] += (c - n_accum_rec[ch]) * dL_dchannel;
		}

		// DEPTH
		if (true) {
			n_accum_rec_depth[0] = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			n_last_depth[0] = depth;

			const float dL_dchannel = dL_dpix_depth;
			dL_dalpha[0] += (depth - n_accum_rec_depth[0]) * dL_dchannel;
		}

		dL_dalpha[0] *= prev_T;
		n_last_alpha[0] = alpha;
		n_log_T[0] = prev_log_T;

		// Account for fact that alpha also influences how much of
		// the background color is added if nothing left to blend
		float bg_coef = -expf(final_log_T - logf(1.0f - alpha));
		for (int ch = 0; ch < C; ch++) 
		{
			const float dL_dchannel = dL_dpix_color[ch];
			dL_dalpha[0] += bg_coef * (bg_color[ch] * dL_dchannel);
		}
		if (true) {
			const float dL_dchannel = dL_dpix_depth;
			dL_dalpha[0] += bg_coef * (1.0 * dL_dchannel);
		}
	}

	// Backward version of the rendering procedure.
	template <uint32_t C>
	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
	renderCUDA(
		int B, int P, int F,
		int W, int H,
		const float* __restrict__ bg_color,

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
		const int* __restrict__ last_face,
		const int* __restrict__ last_tet,
		const float* __restrict__ final_log_Ts,
		const float* __restrict__ final_prev_log_Ts,
		const bool* __restrict__ is_active_forward,
		const uint32_t* __restrict__ n_contrib,
		
		const float* __restrict__ dL_dpix_color,
		const float* __restrict__ dL_dpix_depth,

		float* __restrict__ dL_dverts_color,
		float* __restrict__ dL_dfaces_opacity,
		bool* __restrict__ is_active_backward)
	{
		// We trace rays again. Compute necessary block info.
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
		bool done = !inside;
		if (!inside)
			return;
			
		// In the forward, we stored the log T value for one step before the end.
		const float this_final_prev_log_T = inside ? final_prev_log_Ts[pix_batch_id] : 0;
		const float this_final_log_T = inside ? final_log_Ts[pix_batch_id] : 0;
		const float final_prev_T = expf(this_final_prev_log_T);
		const float final_T = expf(this_final_log_T);
		
		// We use this, because real final T value could be 0.
		float prev_log_T = this_final_prev_log_T;
		
		// We start from the back. The ID of the last contributing
		// face and tet is known from each pixel from the forward.
		uint32_t this_n_contrib = n_contrib[pix_batch_id];
		int this_last_face = last_face[pix_batch_id];
		int this_last_tet = last_tet[pix_batch_id];
		bool this_is_active_forward = is_active_forward[pix_batch_id];
		is_active_backward[pix_batch_id] = false;
		if (!this_is_active_forward) {
			done = true;		// it was invalid tracing at the foward pass.
			is_active_backward[pix_batch_id] = true;
		}
			
		// We need to stop at the first face and tet, which is also
		// computed at the forward pass.
		int this_first_face = first_face[pix_batch_id];
		int this_first_tet = first_tet[pix_batch_id];

		// this gradient;
		float this_dL_dpix_color[C];
		float this_dL_dpix_depth = 0.0f;
		for (int i = 0; i < C; i++)
			this_dL_dpix_color[i] = dL_dpix_color[(batch_id * 3 * H * W) + (i * H * W) + pix_id];
		this_dL_dpix_depth = dL_dpix_depth[(batch_id * H * W) + pix_id];

		// ================
		// Ray marching
		// ================

		const float* curr_mv = &mv_mats[batch_id * 16];
		const float* curr_proj = &proj_mats[batch_id * 16];
		
		float3 this_ray_o = ray_o[pix_batch_id];
		float3 this_ray_d = ray_d[pix_batch_id];
		float3 tuv, p0, p1, p2;

		float rt = 0.0f;
		float iu, iv;
		if (this_last_face == -1) {
			done = true;
			is_active_backward[pix_batch_id] = true;
		}
		else {
			get_face_vert(verts, faces, this_last_face, p0, p1, p2);
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

		// current face and tet;
		int curr_face = this_last_face;
		int curr_tet = this_last_tet;
		float curr_rt = rt;
		float curr_iu = iu;
		float curr_iv = iv;
		
		float last_alpha = 0.0f;
		float last_color[C] = { 0.0f };
		float accum_rec[C] = { 0.0f };		// accumulated color for computing dL / dalpha

		float last_depth = 0.0f;
		float accum_recd = 0.0f;			// accumulated depth for computing dL / dalpha

		// compute correct [curr_tet], because we have to go backward;
		if (curr_face != -1) {
			for (int i = 0; i < 2; i++) {
				int p_curr_tet = face_tets[2 * curr_face + i];
				if (p_curr_tet == curr_tet)
					continue;
				curr_tet = p_curr_tet;
				break;
			}
		}

		bool first_iter = true;
		while(!done) {

			/*
			1. Retrieve current face's information to compute grads.
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
			float i0 = 1.0f - curr_iu - curr_iv;
			float i1 = curr_iu;
			float i2 = curr_iv;
			float3 curr_face_color = (i0 * curr_verts_color_0) + 
										(i1 * curr_verts_color_1) + 
										(i2 * curr_verts_color_2);

			// get opacity of current face (non-batch);
			float curr_face_opacity = faces_opacity[curr_face];

			// Retrieve this face's intensity (batch);
			float curr_face_intense = faces_intense[batch_id * F + curr_face];
			curr_face_color = curr_face_color * curr_face_intense;
			
			// Compute depth;
			float3 curr_point = this_ray_o + (this_ray_d * curr_rt);
			float4 curr_point_ndc = transformPoint4x4(transformPoint4x3(curr_point, curr_mv), curr_proj);
			float curr_point_w = 1.0f / clamp_w(curr_point_ndc.w);
			float curr_point_depth = curr_point_ndc.z * curr_point_w;

			// Get previous log_T;
			if (!first_iter) {
				// curr_face_opacity is smaller than 1.0f,
				// because it would have been dealt with in the [first_iter].
				prev_log_T = prev_log_T - logf(1.0f - curr_face_opacity);
			}
			first_iter = false;

			float prev_T = expf(prev_log_T);
			if (isnan(prev_T) || isinf(prev_T)) {
				printf("[FTETRENDER]: T is nan or inf, which should not happen\n");
			}

			/*
			2. Compute grads (dL/dcolor, dL/dopacity).
			*/
			float dL_dcurr_face_color[3];
			float dL_dcurr_face_opacity = 0.0f;

			// color
			float t_curr_face_color[3] = { curr_face_color.x, curr_face_color.y, curr_face_color.z };
			for (int ch = 0; ch < 3; ch++)
			{
				const float c = t_curr_face_color[ch];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = this_dL_dpix_color[ch];
				dL_dcurr_face_color[ch] = dL_dchannel * curr_face_opacity * prev_T;
				dL_dcurr_face_opacity += (c - accum_rec[ch]) * dL_dchannel;
			}

			// depth;
			for (int ch = 0; ch < 1; ch++) {
				const float c = curr_point_depth;
				// Update last depth (to be used in the next iteration)
				accum_recd = last_alpha * last_depth + (1.f - last_alpha) * accum_recd;
				last_depth = c;

				const float dL_dchannel = this_dL_dpix_depth;
				// curr_dL_didepth[ch] = dL_dchannel * curr_face_opacity * prev_T;
				dL_dcurr_face_opacity += (c - accum_recd) * dL_dchannel;
			}

			// alpha: additional contribution from the background;
			dL_dcurr_face_opacity *= prev_T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = curr_face_opacity;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			float bd_dot_dpixel = 0;
			for (int i = 0; i < 3; i++)
				bg_dot_dpixel += bg_color[i] * this_dL_dpix_color[i];
			for (int i = 0; i < 1; i++)
				bd_dot_dpixel += 1.0 * this_dL_dpix_depth;
			if (curr_face_opacity == 1.0f) {
				// in this case, (-T_final / (1.f - alpha)) is (-prev_T_final),
				// because when alpha == 1.0, it would have been the last step;
				dL_dcurr_face_opacity += (-final_prev_T) * bg_dot_dpixel;
				dL_dcurr_face_opacity += (-final_prev_T) * bd_dot_dpixel;
			}
			else {
				dL_dcurr_face_opacity += (-final_T / (1.f - curr_face_opacity)) * bg_dot_dpixel;
				dL_dcurr_face_opacity += (-final_T / (1.f - curr_face_opacity)) * bd_dot_dpixel;
			}

			float curr_dL_dvcolor_0[3] = { 0 };
			float curr_dL_dvcolor_1[3] = { 0 };
			float curr_dL_dvcolor_2[3] = { 0 };

			for (int ch = 0; ch < 3; ch++)
			{
				curr_dL_dvcolor_0[ch] += i0 * dL_dcurr_face_color[ch] * curr_face_intense;
				curr_dL_dvcolor_1[ch] += i1 * dL_dcurr_face_color[ch] * curr_face_intense;
				curr_dL_dvcolor_2[ch] += i2 * dL_dcurr_face_color[ch] * curr_face_intense;
			}

			// aggregate;
			for (int i = 0; i < 3; i++) {
				// color;
				atomicAdd(&dL_dverts_color[3 * faces[3 * curr_face + 0] + i], curr_dL_dvcolor_0[i]);
				atomicAdd(&dL_dverts_color[3 * faces[3 * curr_face + 1] + i], curr_dL_dvcolor_1[i]);
				atomicAdd(&dL_dverts_color[3 * faces[3 * curr_face + 2] + i], curr_dL_dvcolor_2[i]);
			}
			// opacity;
			atomicAdd(&dL_dfaces_opacity[curr_face], dL_dcurr_face_opacity);

			this_n_contrib--;
			if (curr_face == this_first_face) {
				done = true;
				is_active_backward[pix_id] = (this_n_contrib == 0);
			}
			
			/*
			3. Prepare next iteration;
			*/

			if (!done) {
				if (curr_tet == -1) {
					done = true;
					is_active_backward[pix_id] = false;
				}
				else {
					int prev_face = -1;
					int prev_tet = -1;
					float prev_rt, prev_iu, prev_iv;

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
					and its outward normal is in the different direction
					as the ray direction;
					We do not use [rt] here, because it is weak to
					numerical errors;
					*/

					// if curr face's outward normal was in the opposite 
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
					if (curr_face_normal_dot_prod <= 0.0f) {
						done = true;
						// printf("Error case 2\n");
					}

					int prev_face_cnt = 0;
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

						if (curr_tet_other_face_intersect && curr_tet_other_face_normal_dot_prod < 0.0f) {
							prev_face = curr_tet_other_faces[i];
							prev_rt = curr_tet_other_face_tuv.x;
							prev_iu = curr_tet_other_face_tuv.y;
							prev_iv = curr_tet_other_face_tuv.z;
							prev_face_cnt++;
						}
					}

					// In edge case, there could be multiple intersecting faces,
					// but generally, there should be only one intersecting face.
					if (prev_face_cnt != 1) {
						// it should not happen, but we can't believe numerics...
						done = true;
						// printf("Error case 3\n");
					}
					else {
						for (int i = 0; i < 2; i++) {
							int p_prev_tet = face_tets[2 * prev_face + i];
							if (p_prev_tet == curr_tet || p_prev_tet == -1)
								continue;
							prev_tet = p_prev_tet;
							break;
						}
					}

					curr_face = prev_face;
					curr_tet = prev_tet;
					curr_rt = prev_rt;
					curr_iu = prev_iu;
					curr_iv = prev_iv;
				}
			}
		}
	}

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
		bool* is_active_backward)
	{
		renderCUDA<NUM_CHANNELS> << <batch_grid, block >> >(
			B, P, F,
			W, H,
			bg_color,

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
			last_face,
			last_tet,
			final_log_T,
			final_prev_log_T,
			is_active_forward,
			n_contrib,
			
			dL_dpix_color,
			dL_dpix_depth,

			dL_dverts_color,
			dL_dfaces_opacity,
			is_active_backward
		);
	}
}
