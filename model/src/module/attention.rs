use backend::{Backend, Dtype};
use tensor::{Arena, Tensor};
use std::sync::Arc;

pub struct Attention<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Attention<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize, nh: usize) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		arena.alloc(format!("{prefix}.att"), vec![b, nh, t, t]);
		Attention { arena, prefix: prefix.to_string() }
	}
	
	pub fn forward(&self, x: &Tensor<D>) {
		let b = x.shape()[0] as i32;
		let t = x.shape()[1] as i32;
		let c = (x.shape()[2] / 3) as i32;
		let nh = self.arena.shape(&format!("{}.att", self.prefix))[1] as i32;
		self.arena.backend.ops.attention_forward( //out, att, qkv, b, t, c, nh
			self.arena.get(&format!("{}.output", self.prefix)),
			self.arena.get(&format!("{}.att", self.prefix)),
			x.as_ptr(),
			b, t, c, nh
		);
		
	}
	
	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}
