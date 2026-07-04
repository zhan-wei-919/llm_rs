use backend::Dtype;
use std::sync::Arc;
use tensor::{Arena, Tensor};

use crate::{Linear, module::rope::Rope};

pub struct GQAttention<D: Dtype> {
	arena: Arc<Arena<D>>,
	prefix: String,
	q_proj: Linear<D>,
	k_proj: Linear<D>,
	v_proj: Linear<D>,
	o_proj: Linear<D>,
	rope: Arc<Rope<D>>,		// 归属于模型
	nh: usize,
	nkv: usize,
	hs: usize,
	t_max: usize,
}

impl<D: Dtype> GQAttention<D> {
	pub fn new(
		arena: Arc<Arena<D>>,
		rope: Arc<Rope<D>>,
		prefix: &str,
		nh: usize,
		nkv: usize,
		hs: usize,
		t_max: usize,
		hidden_size: usize,
	) -> Self {
		let b = 1;		// TODO:暂时不支持多batch推理
		arena.alloc(format!("{prefix}.output"), vec![b, t_max, nh * hs]);
		arena.alloc(format!("{prefix}.k_cache"), vec![b, t_max, nkv * hs]);
		arena.alloc(format!("{prefix}.v_cache"), vec![b, t_max, nkv * hs]);
		let q_proj = Linear::new(arena.clone(), &format!("{prefix}.q_proj"), hidden_size, nh * hs, b, t_max, true);
		let k_proj = Linear::new(arena.clone(), &format!("{prefix}.k_proj"), hidden_size, nkv * hs, b, t_max, true);
		let v_proj = Linear::new(arena.clone(), &format!("{prefix}.v_proj"), hidden_size, nkv * hs, b, t_max, true);
		let o_proj = Linear::new(arena.clone(), &format!("{prefix}.o_proj"), nh * hs, hidden_size, b, t_max, false);
		GQAttention { arena, prefix: prefix.to_string(), q_proj, k_proj, v_proj, o_proj, rope, nh, nkv, hs, t_max }
	}

	pub fn k_cache_rows(&self, start: usize, len: usize) -> Tensor<D> {
		let name = format!("{}.k_cache", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name).to_vec()).view(vec![self.t_max, self.nkv * self.hs]).slice_rows(start, len)
	}

	pub fn v_cache_rows(&self, start: usize, len: usize) -> Tensor<D> {
		let name = format!("{}.v_cache", self.prefix);
		Tensor::new(self.arena.get(&name), self.arena.shape(&name).to_vec()).view(vec![self.t_max, self.nkv * self.hs]).slice_rows(start, len)
	}

	// prefill的输入q应该是[b, t, c], 这里的t是初始化的ids.len()
	pub fn forward_prefill(&self, x_norm: &Tensor<D>) {
		let b = x_norm.shape()[0];
		debug_assert_eq!(b, 1);		// TODO:暂时不支持多batch推理
		let t = x_norm.shape()[1];
		self.q_proj.forward(x_norm);
		self.k_proj.forward_into(x_norm, &self.k_cache_rows(0, t));
		self.v_proj.forward_into(x_norm, &self.v_cache_rows(0, t));
		self.rope.forward(&self.q_proj.output(), self.nh, t, 0);
		self.rope.forward(&self.k_cache_rows(0, t), self.nkv, t, 0);
		self.arena.backend.ops.gq_attention_prefill_forward(	// out, q, k, v, b, t, nh, nkv, hs
			self.arena.get(&format!("{}.output", &self.prefix)),
			self.q_proj.output().as_ptr(),
			self.arena.get(&format!("{}.k_cache", &self.prefix)) as *const u8,
			self.arena.get(&format!("{}.v_cache", &self.prefix)) as *const u8,
			b as i32, t as i32, self.nh as i32, self.nkv as i32, self.hs as i32
		);
		let attn_out = Tensor::new(
        	self.arena.get(&format!("{}.output", self.prefix)),
         	vec![b, t, self.nh * self.hs],
    	);
		self.o_proj.forward(&attn_out);
	}

	pub fn forward_decode(&self, x_norm: &Tensor<D>, pos: usize) {
		let b = x_norm.shape()[0];
		debug_assert_eq!(b, 1);		// TODO:暂时不支持多batch推理
		self.q_proj.forward(x_norm);
		self.k_proj.forward_into(x_norm, &self.k_cache_rows(pos, 1));
		self.v_proj.forward_into(x_norm, &self.v_cache_rows(pos, 1));
		self.rope.forward(&self.q_proj.output(), self.nh, 1, pos);
		self.rope.forward(&self.k_cache_rows(pos, 1), self.nkv, 1, pos);
		self.arena.backend.ops.gq_attention_decode_forward(		// out, q, k_cache, v_cache, cur_len, nh, nkv, hs
			self.arena.get(&format!("{}.output", &self.prefix)),
			self.q_proj.output().as_ptr(),
			self.arena.get(&format!("{}.k_cache", &self.prefix)) as *const u8,
			self.arena.get(&format!("{}.v_cache", &self.prefix)) as *const u8,
			(pos + 1) as i32, self.nh as i32, self.nkv as i32, self.hs as i32
		);
		let attn_out = Tensor::new(
        	self.arena.get(&format!("{}.output", self.prefix)),
         	vec![b, 1, self.nh * self.hs],
    	);
		self.o_proj.forward(&attn_out);
	}

	pub fn output(&self) -> Tensor<D> {
		self.o_proj.output()
	}
}
