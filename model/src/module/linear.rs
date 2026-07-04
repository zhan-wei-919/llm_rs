use backend::Dtype;
use std::sync::Arc;
use tensor::{Arena, Tensor};

pub struct Linear<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
	has_bias: bool,
}

impl<D: Dtype> Linear<D> {
	pub fn new(
		arena: Arc<Arena<D>>,
		prefix: &str,
		in_f: usize,
		out_f: usize,
		b: usize,
		t: usize,
		has_bias: bool,
	) -> Self {
		arena.alloc(format!("{prefix}.weight"), vec![in_f, out_f]);
		arena.alloc(format!("{prefix}.output"), vec![b, t, out_f]);
		if has_bias {arena.alloc(format!("{prefix}.bias"), vec![out_f])}
		Linear {
			arena,
			prefix: prefix.to_string(),
			has_bias,
		}
	}

	pub fn forward(&self, x: &Tensor<D>) {
		let m = (x.shape()[0] * x.shape()[1]) as i32;
		let k = x.shape()[2] as i32;
		let n = self.arena.shape(&format!("{}.weight", self.prefix))[1] as i32;
		let bias = if self.has_bias { self.arena.get(&format!("{}.bias", self.prefix))} else {std::ptr::null()};

		self.arena.backend.ops.gemm_forward(
			x.as_ptr(),
			self.arena.get(&format!("{}.weight", self.prefix)) as *const u8,
			self.arena.get(&format!("{}.output", self.prefix)),
			bias,
			1.0,
			0.0,
			m,
			n,
			k,
		);
	}

	pub fn forward_into(&self, x: &Tensor<D>, dst: &Tensor<D>) {
		let m = (x.shape()[0] * x.shape()[1]) as i32;
		let k = x.shape()[2] as i32;
		let n = self.arena.shape(&format!("{}.weight", self.prefix))[1] as i32;
		let bias = if self.has_bias { self.arena.get(&format!("{}.bias", self.prefix))} else {std::ptr::null()};

		self.arena.backend.ops.gemm_forward(
			x.as_ptr(),
			self.arena.get(&format!("{}.weight", self.prefix)) as *const u8,
			dst.as_mut_ptr(),
			bias,
			1.0,
			0.0,
			m,
			n,
			k,
		);
	}

	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}
