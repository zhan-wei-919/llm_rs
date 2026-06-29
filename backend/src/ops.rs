use crate::dtype::Dtype;

pub trait Backend<D: Dtype> {
    fn embedding_forward(&self, out: *mut u8, token_ids: *const i32, token_table: *const u8, pos_table: *const u8, b: i32, t: i32, c:i32);
    
    fn layernorm_forward(&self, out: *mut u8, mean_out: *mut f32, rstd_out: *mut f32, x: *const u8, 
    					gamma: *const u8, beta: *const u8, b: i32, t: i32, c: i32, eps: f32);
    
    fn gemm_forward(&self, a: *const u8, b: *const u8, c: *mut u8, bias: *const u8, alpha: f32, beta: f32, m: i32, n:i32, k:i32);
    
    fn gelu(&self, y: *mut u8, x: *const u8, n: i32);
    
    fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c:i32);
    
    fn attention(&self, out: *mut u8, att: *mut u8, qkv: *const u8, b: i32, t: i32, c: i32, nh: i32);
    
    fn crossentropy_forward(&self, losses: *mut f32, probs: *mut u8, logits: *const u8, targets: *const i32, b: i32, t: i32, v: i32);
}