use backend::Dtype;
use std::sync::Arc;
use tensor::{Arena, Tensor};

pub struct Residual<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Residual<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		Residual {
			arena,
			prefix: prefix.to_string(),
		}
	}

	pub fn forward(&self, x1: &Tensor<D>, x2: &Tensor<D>) {
		let b = x1.shape()[0] as i32;
		let t = x1.shape()[1] as i32;
		let c = x1.shape()[2] as i32;

		self.arena.backend.ops.residual_forward(
			//out, x1, x2, b, t, c
			self.arena.get(&format!("{}.output", self.prefix)),
			x1.as_ptr(),
			x2.as_ptr(),
			b,
			t,
			c,
		);
	}

	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}
