use backend::Dtype;
use tensor::{Arena, Tensor};
use std::sync::Arc;

use crate::Linear;

#[allow(non_camel_case_types)]
pub struct SwiGLU_MLP<D: Dtype> {
	arena: Arc<Arena<D>>,
	gate_proj: Linear<D>,
	up_proj: Linear<D>,
	down_proj: Linear<D>,
	proj_dim: usize,
}

impl<D: Dtype> SwiGLU_MLP<D> {
	pub fn new(
		arena: Arc<Arena<D>>,
		prefix: &str,
		hidden_dim: usize,
		proj_dim: usize,
		t_max: usize,
	) -> Self {
		let b = 1;		// TODO:暂时不支持多batch推理
		let gate_proj = Linear::new(arena.clone(), &format!("{prefix}.gate_proj"), hidden_dim, proj_dim, b, t_max, false);
		let up_proj = Linear::new(arena.clone(), &format!("{prefix}.up_proj"), hidden_dim, proj_dim, b, t_max, false);
		let down_proj = Linear::new(arena.clone(), &format!("{prefix}.down_proj"), proj_dim, hidden_dim, b, t_max, false);
		SwiGLU_MLP{arena, gate_proj, up_proj, down_proj, proj_dim}
	}

// x [1536]  ──gate_proj→ [8960] ──silu→ ┐
//                                       ├─ ⊙ 逐元素乘 → [8960] ──down_proj→ [1536]
// x [1536]  ──up_proj──→ [8960] ────────┘

	pub fn forward(&self, x: &Tensor<D>) {
		let b = x.shape()[0];
		let t = x.shape()[1];
		let n = b * t * self.proj_dim;
		self.gate_proj.forward(x);
		self.up_proj.forward(x);
		self.arena.backend.ops.silu_mul_forward(self.gate_proj.output().as_mut_ptr(), self.gate_proj.output().as_ptr(), self.up_proj.output().as_ptr(), n as i32);
		let swi = Tensor::new(self.gate_proj.output().as_mut_ptr(), vec![b, t, self.proj_dim]);
		self.down_proj.forward(&swi);
	}

	pub fn output(&self) -> Tensor<D> {
		self.down_proj.output()
	}
}
