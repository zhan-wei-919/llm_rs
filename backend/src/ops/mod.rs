mod cuda;

pub(crate) use cuda::CudaOps;

use crate::dtype::Dtype;

pub trait Ops<D: Dtype> {
	fn embedding_forward(
		&self,
		out: *mut u8,
		token_ids: *const i32,
		token_table: *const u8,
		pos_table: *const u8,
		b: i32,
		t: i32,
		c: i32,
	);

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
	);

	/// C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n] (+ bias)
	///
	/// a: 激活值 也就是传入的x. 高维张量一律摊平视为二维：k = 最后一维，m = 其余维的乘积
	///    如 x 为 [批, 序列, 768]，则 m = 批*序列，k = 768；
	/// b: 权重，行主序 [k, n]，与 safetensors 里 Conv1D 的存储布局一致，原样加载即可用
	/// c: 输出，摊平视为 [m, n]；beta != 0 时会读入旧值累加
	/// bias: 长度 n，按行广播；传 null 则跳过
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
	);

	fn gelu_forward(&self, y: *mut u8, x: *const u8, n: i32);

	fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c: i32);

	fn attention_forward(
		&self,
		out: *mut u8,
		att: *mut u8,
		qkv: *const u8,
		b: i32,
		t: i32,
		c: i32,
		nh: i32,
	);

	fn crossentropy_forward(
		&self,
		losses: *mut f32,
		probs: *mut u8,
		logits: *const u8,
		targets: *const i32,
		b: i32,
		t: i32,
		v: i32,
	);

	fn transpose_forward(&self, out: *mut u8, input: *const u8, r: i32, c: i32, out_stride: i32);

	fn gather_kv_forward(
		&self,
		k_cache: *mut u8,
		v_cache: *mut u8,
		qkv: *const u8,
		t: i32,
		c: i32,
		dst_start: i32,
	);

	fn attention_decode_forward(
		&self,
		out: *mut u8,
		qkv: *const u8,
		k_cache: *const u8,
		v_cache: *const u8,
		cur_len: i32,
		c: i32,
		nh: i32,
	);
}
