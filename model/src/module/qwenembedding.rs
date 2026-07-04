use std::sync::Arc;

use backend::Dtype;
use tensor::{Arena, Tensor};

pub struct QwenEmbedding<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
	table: Vec<u8>,		// [vocab, c] 平铺字节,由 load_weight 的 cpu 谓词产出,set_table 注入
	staging: Vec<u8>,	// 本步查表结果的攒批缓冲,每次 forward 复用
	c: usize,
}

impl<D: Dtype> QwenEmbedding<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, t_max: usize, c: usize) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![1, t_max, c]);
		QwenEmbedding { arena, prefix: prefix.to_string(), table: Vec::new(), staging: Vec::new(), c }
	}

	// 构造在权重加载之前发生,表只能事后注入
	pub fn set_table(&mut self, table: Vec<u8>) {
		self.table = table;
	}

	fn row(&self, id: i32) -> &[u8] {
		let w = self.c * D::SIZE;
		&self.table[id as usize * w..(id as usize + 1) * w]
	}

	pub fn forward_prefill(&mut self, token_ids: &[i32]) {
		self.staging.clear();
		let w = self.c * D::SIZE;
		for &id in token_ids {
			// 不走 self.row(): 方法借用整个 self,和 staging 的可变借用冲突;
			// 直接字段访问让借用检查器看到 table/staging 是不相交字段
			self.staging
				.extend_from_slice(&self.table[id as usize * w..(id as usize + 1) * w]);
		}
		self.arena.backend.device.copy_from_host_to_device(
			self.arena.get(&format!("{}.output", self.prefix)),
			self.staging.as_ptr(),
			self.staging.len(),
		);
	}

	pub fn forward_decode(&self, token_id: i32) {
		self.arena.backend.device.copy_from_host_to_device(
			self.arena.get(&format!("{}.output", self.prefix)),
			self.row(token_id).as_ptr(),
			self.c * D::SIZE,
		);
	}

	pub fn prefill_output(&self, t: usize) -> Tensor<D> {
		Tensor::new(self.arena.get(&format!("{}.output", self.prefix)), vec![1, t, self.c])
	}

	pub fn decode_output(&self) -> Tensor<D> {
		Tensor::new(self.arena.get(&format!("{}.output", self.prefix)), vec![1, 1, self.c])
	}
}
