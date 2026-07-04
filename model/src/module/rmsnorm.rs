use std::sync::Arc;

use backend::Dtype;
use tensor::{Arena, Tensor};

pub struct RMSNorm<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
	eps: f32,
}

impl<D: Dtype> RMSNorm<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, t_max: usize, hidden_dim: usize, eps: f32) -> Self{
		arena.alloc(format!("{prefix}.weight"), vec![hidden_dim]);
		arena.alloc(format!("{prefix}.output"), vec![1, t_max, hidden_dim]);
		RMSNorm { arena, prefix: prefix.to_string(), eps }
	}

	pub fn forward(&self, x: &Tensor<D>) {
		let b = x.shape()[0];
		let t = x.shape()[1];
		let hidden_dim = x.shape()[2];
		self.arena.backend.ops.rmsnorm_forward( // out, x, gamma, b, t, c, eps
			self.arena.get(&format!("{}.output", self.prefix)),
			x.as_ptr(),
			self.arena.get(&format!("{}.weight", self.prefix)) as *const u8,
			b as i32, t as i32, hidden_dim as i32, self.eps
		);
	}

	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}
