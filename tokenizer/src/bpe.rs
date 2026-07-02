use std::fs;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use regex::Regex;

const NONE: usize = usize::MAX;

pub struct Tokenizer {
    merges: HashMap<(String, String), usize>,
    vocab: HashMap<String, i32>,
    id_to_token: Vec<String>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    pattern: Regex,
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

impl Tokenizer {
    pub fn new(merges_path: &str, vocab_path: &str) -> Self {
        let content = fs::read_to_string(merges_path).unwrap();
        let mut merges = HashMap::new();
        for (i, line) in content.lines().skip(1).filter(|l| !l.is_empty()).enumerate() {
            let (a, b) = line.split_once(' ').unwrap();
            merges.insert((a.to_string(), b.to_string()), i);
        }

        let content = fs::read_to_string(vocab_path).unwrap();
        let vocab: HashMap<String, i32> = serde_json::from_str(&content).unwrap();

        let byte_encoder = build_byte_encoder();

        // GPT-2 预分词：先按这个正则把文本切成片段，再对每个片段做 BPE
        let pattern = Regex::new(
            r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
        ).unwrap();
        
        let mut id_to_token: Vec<String> = vec![String::new(); 50257];
    	for (s, &idx) in vocab.iter() {
    		id_to_token[idx as usize] = s.clone();
    	}
    	
    	let mut byte_decoder: HashMap<char, u8> = HashMap::new();
    	for (id, c) in byte_encoder.iter().enumerate() {
    		byte_decoder.insert(*c, id as u8);
    	}

        Tokenizer { merges, vocab, id_to_token, byte_encoder, byte_decoder, pattern }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut ids = Vec::new();
        for m in self.pattern.find_iter(text) {
            let piece = m.as_str();
            let tokens: Vec<String> = piece.as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[b as usize].to_string())
                .collect();
            for token in self.bpe(tokens) {
                ids.push(self.vocab[&token]);
            }
        }
        ids
    }

    fn bpe(&self, initial_tokens: Vec<String>) -> Vec<String> {
        let n = initial_tokens.len();
        if n <= 1 {
            return initial_tokens;
        }

        let mut nodes: Vec<Node> = initial_tokens.into_iter().enumerate().map(|(i, token)| {
            Node {
                token,
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i == n - 1 { NONE } else { i + 1 },
                alive: true,
            }
        }).collect();

        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&priority) = self.merges.get(&(nodes[i].token.clone(), nodes[i + 1].token.clone())) {
                heap.push(Reverse((priority, i)));
            }
        }

        while let Some(Reverse((priority, pos))) = heap.pop() {
            if !nodes[pos].alive { continue; }
            let right = nodes[pos].next;
            if right == NONE || !nodes[right].alive { continue; }

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

            // 新的左侧 pair
            if nodes[pos].prev != NONE {
                let left = nodes[pos].prev;
                if let Some(&p) = self.merges.get(&(nodes[left].token.clone(), nodes[pos].token.clone())) {
                    heap.push(Reverse((p, left)));
                }
            }

            // 新的右侧 pair
            if nodes[pos].next != NONE {
                let next = nodes[pos].next;
                if let Some(&p) = self.merges.get(&(nodes[pos].token.clone(), nodes[next].token.clone())) {
                    heap.push(Reverse((p, pos)));
                }
            }
        }

        nodes.into_iter().filter(|n| n.alive).map(|n| n.token).collect()
    }
    
    pub fn decode(&self, ids: &[i32]) -> String {
    	let mut bytes: Vec<u8> = Vec::new();
    	for &id in ids.iter() {
    		for c in self.id_to_token[id as usize].chars() {
    			bytes.push(self.byte_decoder[&c]);
    		}
    	}
    	
    	String::from_utf8_lossy(&bytes).into_owned()
    }
}


#[cfg(test)]
mod tests{
	use super::*;
	
	#[test]
	#[ignore]
	fn test_bpe() {
		let tok = Tokenizer::new("/home/zhanwei/project/llm_rs/data/merges.txt", "/home/zhanwei/project/llm_rs/data/vocab.json");
		let ids = tok.encode("hello world");
		println!("ids = {:?}", ids);
		let out = tok.decode(&ids);
		println!("out = {out}");
		
		for s in ["hello world", "你好，世界！", "code 🚀 rocket", "Ġ 本身出现在文本里"] {
			assert_eq!(tok.decode(&tok.encode(s)), s);
		}
	}
}