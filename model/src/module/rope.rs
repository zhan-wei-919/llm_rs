use std::sync::Arc;

use backend::Dtype;
use tensor::{Arena, Tensor};

pub struct Rope<D: Dtype> {
	arena: Arc<Arena<D>>,
	max_seq: usize,
	half: usize,
}

impl<D: Dtype> Rope<D> {
	pub fn new(arena: Arc<Arena<D>>, max_seq: usize, hs: usize) -> Self{
		arena.alloc_bytes("rope.cos".into(), vec![max_seq, hs / 2], 4);
		arena.alloc_bytes("rope.sin".into(), vec![max_seq, hs / 2], 4);
		Rope { arena, max_seq, half: hs / 2 }
	}

	pub  fn init_table(&self, theta: f64) {
		let mut cos = vec![0f32; self.max_seq * self.half];
		let mut sin = vec![0f32; self.max_seq * self.half];
		for pos in 0..self.max_seq {
			for i in 0..self.half {
				let freq = theta.powf(-(i as f64) / self.half as f64);
				let ang = pos as f64 * freq;
				cos[pos * self.half + i] = ang.cos() as f32;
				sin[pos * self.half + i] = ang.sin() as f32;
			}
		}
		self.arena.backend.device.copy_from_host_to_device(		// dst, src, size
			self.arena.get("rope.cos"),
			cos.as_ptr() as *const u8,
			self.max_seq * self.half * 4,
		);
		self.arena.backend.device.copy_from_host_to_device(		// dst, src, size
			self.arena.get("rope.sin"),
			sin.as_ptr() as *const u8,
			self.max_seq * self.half * 4,
		);
	}

	pub fn forward(&self, x: &Tensor<D>, nh: usize, t: usize, pos0: usize) {
		let hs = self.half * 2;
		self.arena.backend.ops.rope_forward(
			x.as_mut_ptr(),
			self.arena.get("rope.sin") as *const f32,
			self.arena.get("rope.cos") as *const f32,
			t as i32, nh as i32, hs as i32, pos0 as i32, self.max_seq as i32);
	}
}
