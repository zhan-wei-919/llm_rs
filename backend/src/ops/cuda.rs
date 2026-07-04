use crate::dtype::Dtype;
use crate::ops::Ops;

pub struct CudaOps;

// dtype 分发已下沉到 C++ 侧,Rust 只需把 D::TAG 作为首参转发,不再有任何指针 cast。
impl<D: Dtype> Ops<D> for CudaOps {
	fn embedding_forward(
		&self,
		out: *mut u8,
		token_ids: *const i32,
		token_table: *const u8,
		pos_table: *const u8,
		b: i32,
		t: i32,
		c: i32,
	) {
		unsafe {
			kernel::cuda::embedding_forward(D::TAG, out, token_ids, token_table, pos_table, b, t, c);
		}
	}

	fn layernorm_forward(
		&self,
		out: *mut u8,
		mean_out: *mut f32,
		rstd_out: *mut f32,
		x: *const u8,
		gamma: *const u8,
		beta: *const u8,
		b: i32,
		t: i32,
		c: i32,
		eps: f32,
	) {
		unsafe {
			kernel::cuda::layernorm_forward(
				D::TAG, out, mean_out, rstd_out, x, gamma, beta, b, t, c, eps,
			);
		}
	}

	fn gemm_forward(
		&self,
		a: *const u8,
		b: *const u8,
		c: *mut u8,
		bias: *const u8,
		alpha: f32,
		beta: f32,
		m: i32,
		n: i32,
		k: i32,
	) {
		unsafe {
			kernel::cuda::gemm_forward(
				D::TAG,
				a,
				b,
				c,
				bias,
				alpha,
				beta,
				m,
				n,
				k,
				std::ptr::null_mut(),
			);
		}
	}

	fn gelu_forward(&self, y: *mut u8, x: *const u8, n: i32) {
		unsafe {
			kernel::cuda::gelu_forward(D::TAG, y, x, n, std::ptr::null_mut());
		}
	}

	fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c: i32) {
		unsafe {
			kernel::cuda::residual_forward(D::TAG, out, x1, x2, b, t, c);
		}
	}

	fn attention_forward(
		&self,
		out: *mut u8,
		att: *mut u8,
		qkv: *const u8,
		b: i32,
		t: i32,
		c: i32,
		nh: i32,
	) {
		unsafe {
			kernel::cuda::attention_forward(D::TAG, out, att, qkv, b, t, c, nh);
		}
	}

	fn crossentropy_forward(
		&self,
		losses: *mut f32,
		probs: *mut u8,
		logits: *const u8,
		targets: *const i32,
		b: i32,
		t: i32,
		v: i32,
	) {
		unsafe {
			kernel::cuda::crossentropy_forward(D::TAG, losses, probs, logits, targets, b, t, v);
		}
	}

	fn transpose_forward(&self, out: *mut u8, input: *const u8, r: i32, c: i32, out_stride: i32) {
		unsafe {
			kernel::cuda::transpose_forward(D::TAG, out, input, r, c, out_stride);
		}
	}

	fn gather_kv_forward(
		&self,
		k_cache: *mut u8,
		v_cache: *mut u8,
		qkv: *const u8,
		t: i32,
		c: i32,
		dst_start: i32,
	) {
		unsafe {
			kernel::cuda::gather_kv_forward(D::TAG, k_cache, v_cache, qkv, t, c, dst_start);
		}
	}

	fn attention_decode_forward(
		&self,
		out: *mut u8,
		qkv: *const u8,
		k_cache: *const u8,
		v_cache: *const u8,
		cur_len: i32,
		c: i32,
		nh: i32,
	) {
		unsafe {
			kernel::cuda::attention_decode_forward(
				D::TAG, out, qkv, k_cache, v_cache, cur_len, c, nh,
			);
		}
	}

	fn rmsnorm_forward(
		&self,
		out: *mut u8,
		x: *const u8,
		gamma: *const u8,
		b: i32,
		t: i32,
		c: i32,
		eps: f32,
	) {
		unsafe {
			kernel::cuda::rmsnorm_forward(D::TAG, out, x, gamma, b, t, c, eps);
		}
	}

	fn rope_forward(
		&self,
		x: *mut u8,
		cos_table: *const f32,
		sin_table: *const f32,
		t: i32,
		n_heads: i32,
		hs: i32,
		pos0: i32,
		max_seq: i32,
	) {
		unsafe {
			kernel::cuda::rope_forward(
				D::TAG, x, cos_table, sin_table, t, n_heads, hs, pos0, max_seq,
			);
		}
	}

	fn gq_attention_prefill_forward(
		&self,
		out: *mut u8,
		q: *const u8,
		k: *const u8,
		v: *const u8,
		b: i32,
		t: i32,
		nh: i32,
		nkv: i32,
		hs: i32,
	) {
		unsafe {
			kernel::cuda::gq_attention_prefill_forward(
				D::TAG, out, q, k, v, b, t, nh, nkv, hs,
			);
		}
	}

	fn gq_attention_decode_forward(
		&self,
		out: *mut u8,
		q: *const u8,
		k_cache: *const u8,
		v_cache: *const u8,
		cur_len: i32,
		nh: i32,
		nkv: i32,
		hs: i32,
	) {
		unsafe {
			kernel::cuda::gq_attention_decode_forward(
				D::TAG, out, q, k_cache, v_cache, cur_len, nh, nkv, hs,
			);
		}
	}

	fn silu_mul_forward(&self, out: *mut u8, gate: *const u8, up: *const u8, n: i32) {
		unsafe {
			kernel::cuda::silu_mul_forward(D::TAG, out, gate, up, n, std::ptr::null_mut());
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::{Backend, F32};

	#[test]
	#[ignore]
	fn test_attention_decode() {
		let backend = Backend::<F32>::cuda();
		let device = &backend.device;
		let ops = &backend.ops;

		for t in [1, 7, 256, 257] {
			let qkv_input: Vec<f32> = (0..t * 3 * 768)
				.map(|i| ((i * 37) % 256) as f32 / 256.0 - 0.5)
				.collect();
			let qkv_ptr: *const u8 = qkv_input.as_ptr() as *const u8;
			let att_out = device.alloc(1 * 12 * t * t * 4);
			let attention_out = device.alloc(1 * t * 768 * 4);
			let attn_qkv = device.alloc(1 * t * 768 * 3 * 4);
			device.copy_from_host_to_device(attn_qkv, qkv_ptr, 1 * t * 768 * 3 * 4);
			ops.attention_forward(attention_out, att_out, attn_qkv, 1, t as i32, 768, 12);
			let mut sample1 = vec![0f32; 1 * t * 768];
			let sample1_ptr: *mut u8 = sample1.as_mut_ptr() as *mut u8;
			device.copy_from_device_to_host(sample1_ptr, attention_out, 1 * t * 768 * 4);

			let k_cache = device.alloc(1 * t * 768 * 4);
			let v_cache = device.alloc(1 * t * 768 * 4);

			let mut sample2 = vec![0f32; 768];
			let sample2_ptr: *mut u8 = sample2.as_mut_ptr() as *mut u8;

			for i in 0..t {
				let row = unsafe { (attn_qkv as *const u8).add(i * 3 * 768 * 4) };
				ops.gather_kv_forward(k_cache, v_cache, row, 1, 768, i as i32);
				ops.attention_decode_forward(
					attention_out,
					row,
					k_cache,
					v_cache,
					(i + 1) as i32,
					768,
					12,
				);
				device.copy_from_device_to_host(sample2_ptr, attention_out, 768 * 4);
				assert_eq!(
					sample2,
					sample1[i * 768..(i + 1) * 768],
					"t={t}, 位置 i={i} 分叉"
				);
			}
		}
	}
}
