use backend::{Dtype, Backend};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct Gelu<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Gelu<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: String, n: i32) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![n]);
		Gelu { arena, prefix }
	}
	
	pub fn forward(&self, x: &Tensor<D>, n: i32) {
		self.arena.backend.ops.gelu_forward(
			self.arena.get(&format!("{}.output", self.prefix)),
			x.as_ptr(),
			n
		);
	}
}