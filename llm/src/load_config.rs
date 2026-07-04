use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Qwen2Config {
    architectures: Vec<String>,
    attention_dropout: f64,
    bos_token_id: u64,
    eos_token_id: u64,
    hidden_act: String,
    hidden_size: u64,
    initializer_range: f64,
    intermediate_size: u64,
    max_position_embeddings: u64,
    max_window_layers: u64,
    model_type: String,
    num_attention_heads: u64,
    num_hidden_layers: u64,
    num_key_value_heads: u64,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: u64,
    tie_word_embeddings: bool,
    torch_dtype: String,
    transformers_version: String,
    use_cache: bool,
    use_mrope: bool,
    use_sliding_window: bool,
    vocab_size: u64,
}
