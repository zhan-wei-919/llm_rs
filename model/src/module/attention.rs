use backend::Dtype;
use std::sync::Arc;
use tensor::{Arena, Tensor};

pub struct Attention<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
}

impl<D: Dtype> Attention<D> {
	pub fn new(
		arena: Arc<Arena<D>>,
		prefix: &str,
		b: usize,
		t: usize,
		c: usize,
		nh: usize,
	) -> Self {
		arena.alloc(format!("{prefix}.output"), vec![b, t, c]);
		arena.alloc(format!("{prefix}.att"), vec![b, nh, t, t]);
		arena.alloc(format!("{prefix}.k_cache"), vec![t, c]);
		arena.alloc(format!("{prefix}.v_cache"), vec![t, c]);
		Attention {
			arena,
			prefix: prefix.to_string(),
		}
	}

	pub fn fill_cache(&self, qkv: &Tensor<D>, len: usize, dst_start: usize) {
		let c = (qkv.shape()[2] / 3) as i32;
		self.arena.backend.ops.gather_kv_forward(
			// k_cache, v_cache, qkv, t, c, dst_start
			self.arena.get(&format!("{}.k_cache", self.prefix)),
			self.arena.get(&format!("{}.v_cache", self.prefix)),
			qkv.as_ptr(),
			len as i32,
			c,
			dst_start as i32,
		);
	}

	pub fn forward(&self, x: &Tensor<D>) {
		let b = x.shape()[0] as i32;
		let t = x.shape()[1] as i32;
		let c = (x.shape()[2] / 3) as i32;
		let nh = self.arena.shape(&format!("{}.att", self.prefix))[1] as i32;
		self.arena.backend.ops.attention_forward(
			//out, att, qkv, b, t, c, nh
			self.arena.get(&format!("{}.output", self.prefix)),
			self.arena.get(&format!("{}.att", self.prefix)),
			x.as_ptr(),
			b,
			t,
			c,
			nh,
		);
	}

	pub fn forward_decode(&self, x: &Tensor<D>, pos: usize) {
		let c = (x.shape()[2] / 3) as i32;
		self.fill_cache(x, 1, pos);
		self.arena.backend.ops.attention_decode_forward(
			// out, qkv, k_cache, v_cache, cur_len, c, nh
			self.arena.get(&format!("{}.output", self.prefix)),
			x.as_ptr(),
			self.arena.get(&format!("{}.k_cache", self.prefix)),
			self.arena.get(&format!("{}.v_cache", self.prefix)),
			(pos + 1) as i32,
			c,
			self.arena.shape(&format!("{}.att", self.prefix))[1] as i32,
		);
	}

	pub fn output(&self) -> Tensor<D> {
		let name = format!("{}.output", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name))
	}
}
