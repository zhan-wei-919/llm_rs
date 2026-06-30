use backend::{Dtype, Backend};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct Residual<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Residual<D> {
	fn new(arena: Arc<Arena<D>>, prefix: String, b: i32, t: i32, c: i32) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		Residual { arena, prefix }
	}
	
	fn forward(&self, x1: &Tensor<D>, x2: &Tensor<D>) {
		let b = x.shape()[0] as i32;
		let t = x.shape()[1] as i32;
		let c = x.shape()[2] as i32;
		
		self.arena.backend.ops.residual_forward( //out, x1, x2, b, t, c
			self.arena.get(&format!("{}.output", self.prefix)),
			x1.as_ptr(),
			x2.as_ptr(),
			b, t, c
		);
	}
}