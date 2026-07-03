use backend::Dtype;
use std::sync::Arc;
use tensor::{Arena, Tensor};

pub struct Linear<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Linear<D> {
	pub fn new(
		arena: Arc<Arena<D>>,
		prefix: &str,
		in_f: usize,
		out_f: usize,
		b: usize,
		t: usize,
	) -> Self {
		arena.alloc(format!("{prefix}.weight"), vec![in_f, out_f]);
		arena.alloc(format!("{prefix}.bias"), vec![out_f]);
		arena.alloc(format!("{prefix}.output"), vec![b, t, out_f]);
		Linear {
			arena,
			prefix: prefix.to_string(),
		}
	}

	pub fn forward(&self, x: &Tensor<D>) {
		let m = (x.shape()[0] * x.shape()[1]) as i32;
		let k = x.shape()[2] as i32;
		let n = self.arena.shape(&format!("{}.weight", self.prefix))[1] as i32;
		self.arena.backend.ops.gemm_forward(
			x.as_ptr(),
			self.arena.get(&format!("{}.weight", self.prefix)) as *const u8,
			self.arena.get(&format!("{}.output", self.prefix)),
			self.arena.get(&format!("{}.bias", self.prefix)) as *const u8,
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
