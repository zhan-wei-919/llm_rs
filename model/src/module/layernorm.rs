use backend::{Dtype, Backend};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct LayerNorm<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> LayerNorm<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize) -> Self {
		arena.alloc(format!("{prefix}.gamma"), vec![c]);
		arena.alloc(format!("{prefix}.beta"), vec![c]);
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		arena.alloc(format!("{prefix}.mean_out"), vec![b, t]);
		arena.alloc(format!("{prefix}.rstd_out"), vec![b, t]);
		LayerNorm { arena, prefix: prefix.to_string() }
	}
	
	pub fn forward(&self, x: &Tensor<D>, eps: f32) {
		let b = x.shape()[0] as i32;
		let t = x.shape()[1] as i32;
		let c = x.shape()[2] as i32;
		self.arena.backend.ops.layernorm_forward(
			self.arena.get(&format!("{}.output", self.prefix)), 
			self.arena.get(&format!("{}.mean_out", self.prefix)) as *mut f32,
			self.arena.get(&format!("{}.rstd_out", self.prefix)) as *mut f32,
			x.as_ptr(),
			self.arena.get(&format!("{}.gamma", self.prefix)) as *const u8,
			self.arena.get(&format!("{}.beta", self.prefix)) as *const u8,
			b, t, c, eps
		);
	}
	
	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}