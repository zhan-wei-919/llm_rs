use backend::Backend;
use tensor::Arena;
use tokenizer::Tokenizer;
use llm::gpt2::GPT2;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let backend = Arc::new(Backend::cuda());
    let arena = Arc::new(Arena::new(backend));
    let gpt = GPT2::new(arena);
    let hit = gpt.load_model("weights/model.safetensors");
    println!("loaded {} tensors", hit);
    let input = "1 + 1 =";
    let tok = Tokenizer::new("data/merges.txt", "data/vocab.json");
    let input_ids = tok.encode(input);
    
    let start = Instant::now();
    
    let output_ids = gpt.inference(&input_ids, 16);
    let n = output_ids.len();
    
    let elapsed = start.elapsed();
    println!("spend: {:?}, speed: {}", elapsed, n as f32 / elapsed.as_secs_f32());
    
    
    let output = tok.decode(&output_ids);
    println!("{output}");
}
