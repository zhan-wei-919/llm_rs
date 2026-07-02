use backend::{Dtype};
use tensor::{Tensor, Arena};
use std::sync::Arc;

pub struct Embedding<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Embedding<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize, v: usize) -> Self {
		// wte/wpe 是 HF checkpoint 里的顶层名，写死以便哑加载循环按名命中；
		// prefix 只用于激活输出，激活不进文件。
		arena.alloc("wte.weight".to_string(), vec![v, c]);
		arena.alloc("wpe.weight".to_string(), vec![t, c]);
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
			self.arena.get("wte.weight") as *const u8,
			self.arena.get("wpe.weight") as *const u8,
			b, t, c
		);
	}
	
	pub fn forward_decode(&self, token_id: *const i32, pos: usize) {
		let c = self.arena.shape(&format!("{}.output", self.prefix))[2];
		let wpe = unsafe {(self.arena.get("wpe.weight") as *mut u8).add(pos * C * D::SIZE)};
		self.arena.backend.ops.embedding_forward(
			self.arena.get(&format!("{}.output", self.prefix)), 
			token_id, 
			self.arena.get("wte.weight") as *const u8, 
			wpe, 
			1, 1, c
		);
	}
	
	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
	
	pub fn output_decode(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		let c = self.arena.shape(&name)[2];
		Tensor::new(self.arena.get(&name), vec![1, 1, c]);
	}
}