use backend::{Dtype, Backend};
use tensor::{Tensor, Arena};
use std::sync::Arc;
use model::{Embedding, Transformer, Linear, LayerNorm};

pub struct GPT2<D: Dtype> {
	arena: Arc<Arena<D>>,
	embedding: Embedding<D>,
	blocks: Vec<Transformer<D>>,
	ln_f: LayerNorm<D>,
	logits: Linear<D>,
}

impl<D: Dtype> GPT2<D> {
	pub fn new(arena: Arc<Arena<D>>) -> Self {
		let em = Embedding::new(arena.clone(), "em", 1, 1024, 768, 50257);
		let mut blocks = Vec::<Transformer<D>>::new();
		for i in 0..12 {
			blocks.push(
				Transformer::new(arena.clone(), &format!("h.{}", i), 1, 1024, 768, 12)
			);
		}
		let ln = LayerNorm::new(arena.clone(), "ln", 1, 1024, 768);
		let lg = Linear::new(arena.clone(), "lg", 768, 50257, 1, 1024);
		arena.finalize();
		GPT2 { arena, embedding: em, blocks, ln_f: ln, logits: lg }
	}
	
	pub fn forward(&self, token_ids: *const i32) {
		self.embedding.forward(token_ids);
		let mut x = self.embedding.output();
		for block in &self.blocks {
			block.forward(&x, 1e-5);
			x = block.output();
		}
		self.ln_f.forward(&x, 1e-5);
		self.logits.forward(&self.ln_f.output());
	}
	
	pub fn output(&self) -> Tensor<D> {
		self.logits.output()
	}
}
