use backend::{Dtype, F32};
use model::{Embedding, LayerNorm, LmHead, Transformer};
use std::sync::Arc;
use tensor::{Arena, Tensor};

const VOCAB: usize = 50257; // 真实词表
const VOCAB_PAD: usize = 50304; // pad 到 128 倍数，满足 gemm 的 N%4 契约
const SEQ_LEN: usize = 1024; // 与 new 里注册槽位的 t 一致
const EOT: i32 = 50256; // <|endoftext|>，生成到它就该闭嘴了

pub struct GPT2<D: Dtype> {
	arena: Arc<Arena<D>>,
	embedding: Embedding<D>,
	blocks: Vec<Transformer<D>>,
	ln_f: LayerNorm<D>,
	logits: LmHead<D>,
}

impl<D: Dtype> GPT2<D> {
	pub fn new(arena: Arc<Arena<D>>) -> Self {
		let em = Embedding::new(arena.clone(), "em", 1, 1024, 768, 50257);
		let mut blocks = Vec::<Transformer<D>>::new();
		for i in 0..12 {
			blocks.push(Transformer::new(
				arena.clone(),
				&format!("h.{}", i),
				1,
				1024,
				768,
				12,
			));
		}
		let ln = LayerNorm::new(arena.clone(), "ln_f", 1, 1024, 768);
		let lm = LmHead::new(arena.clone(), "lg", 768, VOCAB_PAD, 1, 1024);
		arena.finalize();
		GPT2 {
			arena,
			embedding: em,
			blocks,
			ln_f: ln,
			logits: lm,
		}
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

	pub fn load_model(&self, path: &str) {
		// GPT-2 是 Conv1D 布局无需转置,也没有 host 侧 tensor
		self.arena
			.load_weight(path, |name| name.ends_with(".attn.bias"), |_| false, |_| false);

		let lg = self.arena.get("lg.weight");
		let size = VOCAB_PAD * 768 * D::SIZE;
		self.arena.backend.device.memset(lg, 0, size);

		self.arena.backend.ops.transpose_forward(
			lg,
			self.arena.get("wte.weight") as *const u8,
			VOCAB as i32,
			768,
			VOCAB_PAD as i32,
		);

		self.arena.backend.device.synchronize();
	}

	pub fn output(&self) -> Tensor<D> {
		self.logits.output()
	}
}

impl GPT2<F32> {
	fn logits_row(&self, pos: usize) -> Vec<f32> {
		let mut row = vec![0f32; VOCAB];
		let src = unsafe { self.logits.output().as_ptr().add(pos * VOCAB_PAD * 4) };
		self.arena.backend.device.copy_from_device_to_host(
			// dst, src, size字节
			row.as_mut_ptr() as *mut u8,
			src,
			VOCAB * 4,
		);
		row
	}

	fn argmax(xs: &[f32]) -> usize {
		let mut max_val = f32::NEG_INFINITY;
		let mut max_idx = 0;
		for (i, &x) in xs.iter().enumerate() {
			if x > max_val {
				max_idx = i;
				max_val = x;
			}
		}
		max_idx
	}

	fn prefill(&self, dev_ids: *const i32, len: usize) -> i32 {
		self.forward(dev_ids);
		for block in &self.blocks {
			block.fill_cache(len);
		}
		Self::argmax(&self.logits_row(len - 1)) as i32
	}

	fn decode_step(&self, token_ptr: *const i32, pos: usize) -> i32 {
		self.embedding.forward_decode(token_ptr, pos);
		let mut x = self.embedding.output_decode();
		for block in &self.blocks {
			block.forward_decode(&x, pos, 1e-5);
			x = block.output().view(vec![1, 1, 768]);
		}
		self.ln_f.forward(&x, 1e-5);
		self.logits
			.forward(&self.ln_f.output().view(vec![1, 1, 768]));
		Self::argmax(&self.logits_row(0)) as i32
	}

	pub fn inference_cached(&self, prompt_ids: &[i32], max_new_tokens: usize) -> Vec<i32> {
		assert!(prompt_ids.len() + max_new_tokens <= SEQ_LEN);
		let device = &self.arena.backend.device;
		let dev_ids = device.alloc(SEQ_LEN * 4);
		let mut host_ids = vec![0i32; SEQ_LEN];
		host_ids[..prompt_ids.len()].copy_from_slice(prompt_ids);

		let mut ids = prompt_ids.to_vec();
		device.copy_from_host_to_device(dev_ids, host_ids.as_ptr() as *const u8, SEQ_LEN * 4);
		let mut next = self.prefill(dev_ids as *const i32, ids.len());

		for _ in 0..max_new_tokens {
			if next == EOT {
				break;
			};
			let pos = ids.len();
			self.arena.backend.device.copy_from_host_to_device(
				// dst, src, size
				unsafe { dev_ids.add(pos * 4) },
				&next as *const i32 as *const u8,
				4,
			);
			ids.push(next);
			next = self.decode_step(unsafe { (dev_ids as *const i32).add(pos) }, pos);
		}
		device.free(dev_ids);
		ids
	}

	pub fn inference(&self, prompt_ids: &[i32], max_new_tokens: usize) -> Vec<i32> {
		assert!(prompt_ids.len() + max_new_tokens <= SEQ_LEN);
		let device = &self.arena.backend.device;
		let dev_ids = device.alloc(SEQ_LEN * 4);
		let mut host_ids = vec![0i32; SEQ_LEN];
		host_ids[..prompt_ids.len()].copy_from_slice(prompt_ids);

		let mut ids = prompt_ids.to_vec();
		for _ in 0..max_new_tokens {
			device.copy_from_host_to_device(
				// dst, src, size
				dev_ids,
				host_ids.as_ptr() as *const u8,
				SEQ_LEN * 4,
			);
			self.forward(dev_ids as *const i32);
			let next = Self::argmax(&self.logits_row(ids.len() - 1)) as i32;
			if next == EOT {
				break;
			}
			host_ids[ids.len()] = next;
			ids.push(next);
		}
		device.free(dev_ids);
		ids
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use backend::Backend;
	use backend::F32;
	use std::sync::Arc;
	use tensor::Arena;
	use tokenizer::Tokenizer;

	#[test]
	#[ignore]
	fn test_load_gpt2_weight() {
		let backend = Arc::new(Backend::<F32>::cuda());
		let arena = Arc::new(Arena::new(backend));
		let _model = GPT2::new(arena.clone());
		_model.load_model("/home/zhanwei/project/llm_rs/weights/model.safetensors");
		// 转置定义：lg[k][n] == wte[n][k]。lg 行主序 [768, 50304]，wte 行主序 [50257, 768]。
		let read_f32 = |name: &str, row: usize, col: usize, row_stride: usize| -> f32 {
			let mut v = [0f32; 1];
			let src = unsafe { (arena.get(name) as *const u8).add((row * row_stride + col) * 4) };
			arena
				.backend
				.device
				.copy_from_device_to_host(v.as_mut_ptr() as *mut u8, src, 4);
			v[0]
		};

		// 抽两个非平凡位置：一个靠角落，一个在深处
		assert_eq!(
			read_f32("lg.weight", 1, 0, VOCAB_PAD),
			read_f32("wte.weight", 0, 1, 768)
		);
		assert_eq!(
			read_f32("lg.weight", 5, 12345, VOCAB_PAD),
			read_f32("wte.weight", 12345, 5, 768)
		);
		// pad 区必须是 0（同时验证 memset 和 out_stride 没错位）
		assert_eq!(read_f32("lg.weight", 0, 50300, VOCAB_PAD), 0.0);
	}

	#[test]
	#[ignore]
	fn test_qkv() {
		let backend = Arc::new(Backend::<F32>::cuda());
		let arena = Arc::new(Arena::new(backend));
		let model = GPT2::new(arena.clone());
		let read_qkv = |name: &str| -> Vec<f32> {
			let mut res: Vec<f32> = vec![0.0; arena.shape(name).iter().product()];
			let ptr: *mut u8 = res.as_mut_ptr() as *mut u8;
			arena.backend.device.copy_from_device_to_host(
				ptr,
				arena.get(name) as *const u8,
				arena.shape(name).iter().product::<usize>() * 4,
			);
			res
		};
		let input = "hello world";
		let _hit = model.load_model("/home/zhanwei/project/llm_rs/weights/model.safetensors");
		let tok = Tokenizer::new(
			"/home/zhanwei/project/llm_rs/data/merges.txt",
			"/home/zhanwei/project/llm_rs/data/vocab.json",
		);
		let input_ids = tok.encode(input);

		let dev_ids = model.arena.backend.device.alloc(SEQ_LEN * 4);
		let mut host_ids = vec![0i32; SEQ_LEN];
		host_ids[..input_ids.len()].copy_from_slice(&input_ids);
		let mut ids = input_ids.to_vec();
		for _ in 0..5 {
			model.arena.backend.device.copy_from_host_to_device(
				dev_ids,
				host_ids.as_ptr() as *const u8,
				SEQ_LEN * 4,
			);
			model.forward(dev_ids as *const i32);
			let next = GPT2::<F32>::argmax(&model.logits_row(ids.len() - 1)) as i32;
			if next == EOT {
				break;
			}
			host_ids[ids.len()] = next;
			ids.push(next);
		}
		let sample1 = read_qkv("h.0.attn.c_attn.output");
		arena.backend.device.free(dev_ids);

		let dev_ids = model.arena.backend.device.alloc(SEQ_LEN * 4);
		let mut host_ids = vec![0i32; SEQ_LEN];
		host_ids[..input_ids.len()].copy_from_slice(&input_ids);
		let mut ids = input_ids.to_vec();
		for _ in 0..6 {
			model.arena.backend.device.copy_from_host_to_device(
				dev_ids,
				host_ids.as_ptr() as *const u8,
				SEQ_LEN * 4,
			);
			model.forward(dev_ids as *const i32);
			let next = GPT2::<F32>::argmax(&model.logits_row(ids.len() - 1)) as i32;
			if next == EOT {
				break;
			}
			host_ids[ids.len()] = next;
			ids.push(next);
		}
		let sample2 = read_qkv("h.0.attn.c_attn.output");
		arena.backend.device.free(dev_ids);

		let n = 6 * 2304;
		assert_eq!(&sample1[..n], &sample2[..n]);
		assert_ne!(&sample1[n..n + 2304], &sample2[n..n + 2304]);
	}

	#[test]
	#[ignore]
	fn test_cache() {
		let backend = Arc::new(Backend::<F32>::cuda());
		let arena = Arc::new(Arena::new(backend));
		let model = GPT2::new(arena.clone());
		let input = "hello world";
		let _hit = model.load_model("/home/zhanwei/project/llm_rs/weights/model.safetensors");
		let tok = Tokenizer::new(
			"/home/zhanwei/project/llm_rs/data/merges.txt",
			"/home/zhanwei/project/llm_rs/data/vocab.json",
		);
		let input_ids = tok.encode(input);

		let output_ids_naive = model.inference(&input_ids, 16);
		let output_ids_cache = model.inference_cached(&input_ids, 16);

		assert_eq!(output_ids_cache, output_ids_naive);
	}
}
