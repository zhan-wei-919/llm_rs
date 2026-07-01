use backend::{Dtype, Backend};
use tensor::{Arena, Tensor};
use std::sync::Arc;
use crate::module::linear::Linear;
use crate::module::layernorm::LayerNorm;
use crate::module::gelu::Gelu;
use crate::module::attention::Attention;
use crate::module::residual::Residual;

pub struct Transformer<D: Dtype> {
	arena: Arc<Arena<D>>,
	ln1: LayerNorm<D>,
	attn_qkv: Linear<D>,
	attn: Attention<D>,
	attn_proj: Linear<D>,
	r1: Residual<D>,
	ln2: LayerNorm<D>,
	fc: Linear<D>,
	gelu: Gelu<D>,
	fc_proj: Linear<D>,
	r2: Residual<D>,
}

impl<D: Dtype> Transformer<D> {
	pub fn new(arena: Arc<Arena<D>>, prefix: &str, b: usize, t: usize, c: usize, nh: usize) -> Self {
		let ln1 = LayerNorm::new(arena.clone(), format!("{prefix}.ln_1"), b, t, c);
		let attn_qkv = Linear::new(arena.clone(), format!("{prefix}.attn.c_attn"), c, 3 * c, b, t);
		let attn = Attention::new(arena.clone(), format!("{prefix}.attn"), b, t, c, nh);
		let attn_proj = Linear::new(arena.clone(), format!("{prefix}.attn.c_proj"), c, c, b, t);
		let r1 = Residual::new(arena.clone(), format!("{prefix}.r1"), b, t, c);
		let ln2 = LayerNorm::new(arena.clone(), format!("{prefix}.ln_2"), b, t, c);
		let fc = Linear::new(arena.clone(), format!("{prefix}.mlp.c_fc"), c, 4 * c, b, t);
		let gelu = Gelu::new(arena.clone(), format!("{prefix}.gelu"), b * t * 4 * c);
		let fc_proj = Linear::new(arena.clone(), format!("{prefix}.mlp.c_proj"), 4 * c, c, b, t);
		let r2 = Residual::new(arena.clone(), format!("{prefix}.r2"), b, t, c);
		Transformer { arena, ln1, attn_qkv, attn, attn_proj, r1, ln2, fc, gelu, fc_proj, r2 }
	}

	pub fn forward(&self, x: &Tensor<D>, eps: f32) {
		self.ln1.forward(x, eps);
		self.attn_qkv.forward(&self.ln1.output());
		self.attn.forward(&self.attn_qkv.output());
		self.attn_proj.forward(&self.attn.output());
		self.r1.forward(x, &self.attn_proj.output());
		self.ln2.forward(&self.r1.output(), eps);
		self.fc.forward(&self.ln2.output());
		self.gelu.forward(&self.fc.output());
		self.fc_proj.forward(&self.gelu.output());
		self.r2.forward(&self.r1.output(), &self.fc_proj.output());
	}

	pub fn output(&self) -> Tensor<D> {
		self.r2.output()
	}
}
