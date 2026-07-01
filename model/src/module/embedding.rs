use backend::{Dtype, Backend};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct Embedding<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Embedding<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize, v: usize) -> Self {
		arena.alloc(format!("{prefix}.token_table"), vec![v, c]);
		arena.alloc(format!("{prefix}.pos_table"), vec![t, c]);
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		Embedding{ arena, prefix: prefix.to_string() }
	}
	
	pub fn forward(&self, token_ids: *const i32) {
		let b = self.arena.shape(&format!("{}.output", self.prefix))[0] as i32;
		let t = self.arena.shape(&format!("{}.output", self.prefix))[1] as i32;
		let c = self.arena.shape(&format!("{}.output", self.prefix))[2] as i32;
		self.arena.backend.ops.embedding_forward(	// out, token_ids, token_table, pos_table, b, t, c
			self.arena.get(&format!("{}.output", self.prefix)),
			token_ids,
			self.arena.get(&format!("{}.token_table", self.prefix)) as *const u8,
			self.arena.get(&format!("{}.pos_table", self.prefix)) as *const u8,
			b, t, c
		);
	}
	
	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}