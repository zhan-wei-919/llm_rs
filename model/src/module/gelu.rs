use backend::{Dtype};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct Gelu<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Gelu<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![b ,t, c]);
		Gelu { arena, prefix: prefix.to_string() }
	}
	
	pub fn forward(&self, x: &Tensor<D>) {
		let n = x.numel() as i32;
		self.arena.backend.ops.gelu_forward(
			self.arena.get(&format!("{}.output", self.prefix)),
			x.as_ptr(),
			n
		);
	}
	
	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}