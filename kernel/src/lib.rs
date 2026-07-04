#![allow(non_camel_case_types)]

pub mod cuda {
	pub type cudaStream_t = *mut std::ffi::c_void;

	// 每个 op 只导出一个符号,首参 dtype 在 C++ 侧 switch 分发。
	// dtype 契约: F32=0 BF16=1 F16=2,与 backend Dtype::TAG 一致。
	// 参与 dtype 化的数据指针统一声明为 *mut u8 / *const u8,Rust 侧无需任何 cast;
	// 恒为 f32 / i32 的表(cos/sin、mean/rstd、losses、token_ids、targets)保持原类型。

	// ---- Embedding ----
	unsafe extern "C" {
		pub fn embedding_forward(
			dtype: i32,
			out: *mut u8,
			token_ids: *const i32,
			token_table: *const u8,
			pos_table: *const u8,
			B: i32,
			seq_len: i32,
			C: i32,
		);
	}

	// ---- RoPE ----
	unsafe extern "C" {
		pub fn rope_forward(
			dtype: i32,
			x: *mut u8,
			cos_table: *const f32,
			sin_table: *const f32,
			seq_len: i32,
			n_heads: i32,
			HS: i32,
			pos0: i32,
			max_seq: i32,
		);
	}

	// ---- LayerNorm ----
	unsafe extern "C" {
		pub fn layernorm_forward(
			dtype: i32,
			out: *mut u8,
			mean_out: *mut f32,
			rstd_out: *mut f32,
			x: *const u8,
			gamma: *const u8,
			beta: *const u8,
			B: i32,
			seq_len: i32,
			C: i32,
			eps: f32,
		);
	}

	// ---- RMSNorm ----
	unsafe extern "C" {
		pub fn rmsnorm_forward(
			dtype: i32,
			out: *mut u8,
			x: *const u8,
			gamma: *const u8,
			B: i32,
			seq_len: i32,
			C: i32,
			eps: f32,
		);
	}

	// ---- Gemm ----
	unsafe extern "C" {
		pub fn gemm_forward(
			dtype: i32,
			A: *const u8,
			B: *const u8,
			C: *mut u8,
			bias: *const u8,
			alpha: f32,
			beta: f32,
			M: i32,
			N: i32,
			K: i32,
			stream: cudaStream_t,
		);
	}

	// ---- GELU ----
	unsafe extern "C" {
		pub fn gelu_forward(dtype: i32, y: *mut u8, x: *const u8, N: i32, stream: cudaStream_t);
	}

	// ---- SiLU Mul (SwiGLU) ----
	unsafe extern "C" {
		pub fn silu_mul_forward(
			dtype: i32,
			out: *mut u8,
			gate: *const u8,
			up: *const u8,
			N: i32,
			stream: cudaStream_t,
		);
	}

	// ---- Residual ----
	unsafe extern "C" {
		pub fn residual_forward(
			dtype: i32,
			out: *mut u8,
			a: *const u8,
			b: *const u8,
			B: i32,
			seq_len: i32,
			C: i32,
		);
	}

	// ---- Attention ----
	unsafe extern "C" {
		pub fn attention_forward(
			dtype: i32,
			out: *mut u8,
			att: *mut u8,
			qkv: *const u8,
			B: i32,
			seq_len: i32,
			C: i32,
			NH: i32,
		);
	}

	// --- Attention Decode ---
	unsafe extern "C" {
		pub fn attention_decode_forward(
			dtype: i32,
			out: *mut u8,
			qkv: *const u8,
			k_cache: *const u8,
			v_cache: *const u8,
			cur_len: i32,
			c: i32,
			nh: i32,
		);
	}

	// --- GQAttention Prefill ---
	unsafe extern "C" {
		pub fn gq_attention_prefill_forward(
			dtype: i32,
			out: *mut u8,
			q: *const u8,
			k: *const u8,
			v: *const u8,
			b: i32,
			seq_len: i32,
			nh: i32,
			nkv: i32,
			hs: i32,
		);
	}

	// --- GQAttention Decode ---
	unsafe extern "C" {
		pub fn gq_attention_decode_forward(
			dtype: i32,
			out: *mut u8,
			q: *const u8,
			k_cache: *const u8,
			v_cache: *const u8,
			cur_len: i32,
			nh: i32,
			nkv: i32,
			hs: i32,
		);
	}

	// ---- CrossEntropy ----
	unsafe extern "C" {
		pub fn crossentropy_forward(
			dtype: i32,
			losses: *mut f32,
			probs: *mut u8,
			logits: *const u8,
			targets: *const i32,
			B: i32,
			seq_len: i32,
			V: i32,
		);
	}

	// --- transpose ---
	unsafe extern "C" {
		pub fn transpose_forward(
			dtype: i32,
			out: *mut u8,
			input: *const u8,
			R: i32,
			C: i32,
			out_stride: i32,
		);
	}

	// --- gather_kv ---
	unsafe extern "C" {
		pub fn gather_kv_forward(
			dtype: i32,
			k_cache: *mut u8,
			v_cache: *mut u8,
			qkv: *const u8,
			t: i32,
			c: i32,
			dst_start: i32,
		);
	}

	// ---- CUDA Runtime API ----

	unsafe extern "C" {
		pub fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> i32;
		pub fn cudaFree(devPtr: *mut u8) -> i32;
		pub fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: i32) -> i32;
		pub fn cudaMemset(devPtr: *mut u8, value: i32, count: usize) -> i32;
		pub fn cudaDeviceSynchronize() -> i32;
	}

	pub const MEMCPY_HOST_TO_DEVICE: i32 = 1;
	pub const MEMCPY_DEVICE_TO_HOST: i32 = 2;
	pub const MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
}
