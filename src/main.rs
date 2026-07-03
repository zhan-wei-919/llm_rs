use backend::Backend;
use llm::gpt2::GPT2;
use std::sync::Arc;
use std::time::Instant;
use tensor::Arena;
use tokenizer::Tokenizer;

fn main() {
	let backend = Arc::new(Backend::cuda());
	let arena = Arc::new(Arena::new(backend));
	let gpt = GPT2::new(arena);
	let hit = gpt.load_model("weights/model.safetensors");
	println!("loaded {} tensors", hit);
	let input = "Alan Turing theorized that computers would one day become";
	let tok = Tokenizer::new("data/merges.txt", "data/vocab.json");
	let input_ids = tok.encode(input);

	let start = Instant::now();

	let output_ids = gpt.inference_cached(&input_ids, 256);
	let n = output_ids.len();

	let elapsed = start.elapsed();
	println!(
		"spend: {:?}, speed: {}",
		elapsed,
		n as f32 / elapsed.as_secs_f32()
	);

	let output = tok.decode(&output_ids);
	println!("{output}");
}
