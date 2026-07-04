use fancy_regex::Regex;
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use unicode_normalization::UnicodeNormalization;

const NONE: usize = usize::MAX;

// R1 蒸馏模型的对话标记。字符里有全角竖线 U+FF5C 和下划线块 U+2581，
// 肉眼和 '|' '_' 无法区分，所以用转义写死，杜绝形近字符手误。
const BOS: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
const EOS: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
const USER: &str = "<\u{FF5C}User\u{FF5C}>";
const ASSISTANT: &str = "<\u{FF5C}Assistant\u{FF5C}>";

pub struct Qwen2Tokenizer {
	merges: HashMap<(String, String), usize>,
	vocab: HashMap<String, i32>,
	id_to_token: HashMap<i32, String>,
	// added tokens 不参与 BPE，编码前先从原文里整体切出来；按长度降序保证最长匹配优先
	specials: Vec<(String, i32)>,
	special_ids: HashMap<i32, String>,
	byte_encoder: [char; 256],
	byte_decoder: HashMap<char, u8>,
	pattern: Regex,
	eos_id: i32,
}

struct Node {
	token: String,
	prev: usize,
	next: usize,
	alive: bool,
}

fn build_byte_encoder() -> [char; 256] {
	let mut table = ['\0'; 256];
	let mut n = 0u32;
	for b in 0..=255u8 {
		table[b as usize] = match b {
			0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => char::from(b),
			_ => {
				let c = char::from_u32(256 + n).unwrap();
				n += 1;
				c
			}
		};
	}
	table
}

impl Qwen2Tokenizer {
	pub fn new(tokenizer_json_path: &str) -> Self {
		let content = fs::read_to_string(tokenizer_json_path).unwrap();
		let tj: Value = serde_json::from_str(&content).unwrap();

		let mut vocab: HashMap<String, i32> = HashMap::new();
		let mut id_to_token: HashMap<i32, String> = HashMap::new();
		for (token, id) in tj["model"]["vocab"].as_object().unwrap() {
			let id = id.as_i64().unwrap() as i32;
			vocab.insert(token.clone(), id);
			id_to_token.insert(id, token.clone());
		}

		// merges 是 "a b" 格式的字符串数组；byte-level 词条内不含裸空格，split_once 安全
		let mut merges = HashMap::new();
		for (i, m) in tj["model"]["merges"].as_array().unwrap().iter().enumerate() {
			let (a, b) = m.as_str().unwrap().split_once(' ').unwrap();
			merges.insert((a.to_string(), b.to_string()), i);
		}

		let mut specials: Vec<(String, i32)> = Vec::new();
		let mut special_ids: HashMap<i32, String> = HashMap::new();
		for at in tj["added_tokens"].as_array().unwrap() {
			let content = at["content"].as_str().unwrap().to_string();
			let id = at["id"].as_i64().unwrap() as i32;
			specials.push((content.clone(), id));
			special_ids.insert(id, content);
		}
		specials.sort_by_key(|(s, _)| Reverse(s.len()));

		// 预切分正则直接取文件里的定义，不手抄；含 lookahead，所以用 fancy_regex
		let pat = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
			.as_str()
			.unwrap();
		let pattern = Regex::new(pat).unwrap();

		let byte_encoder = build_byte_encoder();
		let mut byte_decoder: HashMap<char, u8> = HashMap::new();
		for (id, c) in byte_encoder.iter().enumerate() {
			byte_decoder.insert(*c, id as u8);
		}

		let eos_id = *vocab.get(EOS).unwrap_or(
			specials
				.iter()
				.find(|(s, _)| s == EOS)
				.map(|(_, id)| id)
				.unwrap(),
		);

		Qwen2Tokenizer {
			merges,
			vocab,
			id_to_token,
			specials,
			special_ids,
			byte_encoder,
			byte_decoder,
			pattern,
			eos_id,
		}
	}

	pub fn eos_id(&self) -> i32 {
		self.eos_id
	}

	/// R1 蒸馏模型的对话模板：<BOS><User>{prompt}<Assistant>，模型接着会生成 <think>...
	pub fn encode_chat(&self, user_prompt: &str) -> Vec<i32> {
		self.encode(&format!("{BOS}{USER}{user_prompt}{ASSISTANT}"))
	}

	pub fn encode(&self, text: &str) -> Vec<i32> {
		// tokenizer.json 声明了 NFC normalizer，先归一化再切分
		let text: String = text.nfc().collect();
		let mut ids = Vec::new();
		let mut rest = text.as_str();
		// 先切出 added tokens（它们不参与 BPE），剩余片段走正则 + BPE
		while !rest.is_empty() {
			match self.find_first_special(rest) {
				Some((start, len, id)) => {
					self.encode_text(&rest[..start], &mut ids);
					ids.push(id);
					rest = &rest[start + len..];
				}
				None => {
					self.encode_text(rest, &mut ids);
					break;
				}
			}
		}
		ids
	}

	/// 返回 (字节偏移, 字节长度, id)。specials 已按长度降序，同一位置最长者先命中
	fn find_first_special(&self, text: &str) -> Option<(usize, usize, i32)> {
		let mut best: Option<(usize, usize, i32)> = None;
		for (s, id) in &self.specials {
			if let Some(pos) = text.find(s.as_str()) {
				if best.is_none() || pos < best.unwrap().0 {
					best = Some((pos, s.len(), *id));
				}
			}
		}
		best
	}

	fn encode_text(&self, text: &str, ids: &mut Vec<i32>) {
		for m in self.pattern.find_iter(text) {
			let piece = m.unwrap().as_str();
			let tokens: Vec<String> = piece
				.as_bytes()
				.iter()
				.map(|&b| self.byte_encoder[b as usize].to_string())
				.collect();
			for token in self.bpe(tokens) {
				ids.push(self.vocab[&token]);
			}
		}
	}

	fn bpe(&self, initial_tokens: Vec<String>) -> Vec<String> {
		let n = initial_tokens.len();
		if n <= 1 {
			return initial_tokens;
		}

		let mut nodes: Vec<Node> = initial_tokens
			.into_iter()
			.enumerate()
			.map(|(i, token)| Node {
				token,
				prev: if i == 0 { NONE } else { i - 1 },
				next: if i == n - 1 { NONE } else { i + 1 },
				alive: true,
			})
			.collect();

		let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
		for i in 0..n - 1 {
			if let Some(&priority) = self
				.merges
				.get(&(nodes[i].token.clone(), nodes[i + 1].token.clone()))
			{
				heap.push(Reverse((priority, i)));
			}
		}

		while let Some(Reverse((priority, pos))) = heap.pop() {
			if !nodes[pos].alive {
				continue;
			}
			let right = nodes[pos].next;
			if right == NONE || !nodes[right].alive {
				continue;
			}

			// 懒删除：验证当前 pair 是否还对应这个 priority
			let pair = (nodes[pos].token.clone(), nodes[right].token.clone());
			match self.merges.get(&pair) {
				Some(&p) if p == priority => {}
				_ => continue,
			}

			// 合并：左节点吸收右节点
			nodes[pos].token = format!("{}{}", pair.0, pair.1);
			nodes[right].alive = false;
			let right_next = nodes[right].next;
			nodes[pos].next = right_next;
			if right_next != NONE {
				nodes[right_next].prev = pos;
			}

			if nodes[pos].prev != NONE {
				let left = nodes[pos].prev;
				if let Some(&p) = self
					.merges
					.get(&(nodes[left].token.clone(), nodes[pos].token.clone()))
				{
					heap.push(Reverse((p, left)));
				}
			}

			if nodes[pos].next != NONE {
				let next = nodes[pos].next;
				if let Some(&p) = self
					.merges
					.get(&(nodes[pos].token.clone(), nodes[next].token.clone()))
				{
					heap.push(Reverse((p, pos)));
				}
			}
		}

		nodes
			.into_iter()
			.filter(|n| n.alive)
			.map(|n| n.token)
			.collect()
	}

	pub fn decode(&self, ids: &[i32]) -> String {
		let mut bytes: Vec<u8> = Vec::new();
		for &id in ids {
			// added tokens 的原文不在 byte-unicode 空间里，直接按 UTF-8 还原
			if let Some(s) = self.special_ids.get(&id) {
				bytes.extend_from_slice(s.as_bytes());
				continue;
			}
			for c in self.id_to_token[&id].chars() {
				bytes.push(self.byte_decoder[&c]);
			}
		}
		String::from_utf8_lossy(&bytes).into_owned()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn data_path(name: &str) -> String {
		format!("{}/../data/qwen2/{}", env!("CARGO_MANIFEST_DIR"), name)
	}

	// 金标准由 HF tokenizers 生成（data/qwen2/goldens.json），逐条比对 encode，
	// 并验证 decode(encode(x)) == x 回环
	#[test]
	fn test_against_hf_goldens() {
		let tok = Qwen2Tokenizer::new(&data_path("tokenizer.json"));
		let content = fs::read_to_string(data_path("goldens.json")).unwrap();
		let cases: Vec<(String, Vec<i32>)> = serde_json::from_str::<Vec<Value>>(&content)
			.unwrap()
			.into_iter()
			.map(|v| {
				let text = v["text"].as_str().unwrap().to_string();
				let ids = v["ids"]
					.as_array()
					.unwrap()
					.iter()
					.map(|x| x.as_i64().unwrap() as i32)
					.collect();
				(text, ids)
			})
			.collect();

		assert!(!cases.is_empty());
		for (text, expect) in &cases {
			let got = tok.encode(text);
			assert_eq!(&got, expect, "encode 与 HF 不一致: {text:?}");
			// NFC 归一化是有损的（分解形式会被合成），回环只保证到 nfc(text)
			let nfc: String = text.nfc().collect();
			assert_eq!(tok.decode(&got), nfc, "decode 回环失败: {text:?}");
		}
	}

	#[test]
	fn test_chat_template() {
		let tok = Qwen2Tokenizer::new(&data_path("tokenizer.json"));
		let ids = tok.encode_chat("hi");
		// 模板结构：BOS + User标记 + 内容 + Assistant标记
		assert_eq!(ids[0], 151646);
		assert_eq!(ids[1], 151644);
		assert_eq!(*ids.last().unwrap(), 151645);
		assert_eq!(tok.eos_id(), 151643);
	}
}
