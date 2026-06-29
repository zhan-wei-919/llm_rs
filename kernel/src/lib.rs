#![allow(non_camel_case_types)]

pub mod cuda {
    pub type cudaStream_t = *mut std::ffi::c_void;

    type bf16 = u16;
    type f16 = u16;

    // ---- Embedding ----
    unsafe extern "C" {
        pub fn embedding_forward_f32(
            out: *mut f32,
            token_ids: *const i32,
            token_table: *const f32,
            pos_table: *const f32,
            B: i32,
            seq_len: i32,
            C: i32,
        );
        pub fn embedding_forward_bf16(
            out: *mut bf16,
            token_ids: *const i32,
            token_table: *const bf16,
            pos_table: *const bf16,
            B: i32,
            seq_len: i32,
            C: i32,
        );
        pub fn embedding_forward_f16(
            out: *mut f16,
            token_ids: *const i32,
            token_table: *const f16,
            pos_table: *const f16,
            B: i32,
            seq_len: i32,
            C: i32,
        );
    }

    // ---- LayerNorm ----
    unsafe extern "C" {
        pub fn layernorm_forward_f32(
            out: *mut f32,
            mean_out: *mut f32,
            rstd_out: *mut f32,
            x: *const f32,
            gamma: *const f32,
            beta: *const f32,
            B: i32,
            seq_len: i32,
            C: i32,
            eps: f32,
        );
        pub fn layernorm_forward_bf16(
            out: *mut bf16,
            mean_out: *mut f32,
            rstd_out: *mut f32,
            x: *const bf16,
            gamma: *const bf16,
            beta: *const bf16,
            B: i32,
            seq_len: i32,
            C: i32,
            eps: f32,
        );
        pub fn layernorm_forward_f16(
            out: *mut f16,
            mean_out: *mut f32,
            rstd_out: *mut f32,
            x: *const f16,
            gamma: *const f16,
            beta: *const f16,
            B: i32,
            seq_len: i32,
            C: i32,
            eps: f32,
        );
    }

    // ---- Gemm ----
    unsafe extern "C" {
        pub fn gemm_forward_f32(
            A: *const f32,
            B: *const f32,
            C: *mut f32,
            bias: *const f32,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
        pub fn gemm_forward_bf16(
            A: *const bf16,
            B: *const bf16,
            C: *mut bf16,
            bias: *const bf16,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
        pub fn gemm_forward_bf16_f32(
            A: *const bf16,
            B: *const bf16,
            C: *mut f32,
            bias: *const f32,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
        pub fn gemm_forward_f16(
            A: *const f16,
            B: *const f16,
            C: *mut f16,
            bias: *const f16,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
        pub fn gemm_forward_f16_f32(
            A: *const f16,
            B: *const f16,
            C: *mut f32,
            bias: *const f32,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
        pub fn gemm_forward_i8_i32(
            A: *const i8,
            B: *const i8,
            C: *mut i32,
            bias: *const i32,
            alpha: f32,
            beta: f32,
            M: i32,
            N: i32,
            K: i32,
            stream: cudaStream_t,
        );
    }

    // ---- GELU ----
    unsafe extern "C" {
        pub fn gelu_forward_f32(y: *mut f32, x: *const f32, N: i32, stream: cudaStream_t);
        pub fn gelu_forward_bf16(y: *mut bf16, x: *const bf16, N: i32, stream: cudaStream_t);
        pub fn gelu_forward_f16(y: *mut f16, x: *const f16, N: i32, stream: cudaStream_t);
    }

    // ---- Residual ----
    unsafe extern "C" {
        pub fn residual_forward_f32(
            out: *mut f32,
            a: *const f32,
            b: *const f32,
            B: i32,
            seq_len: i32,
            C: i32,
        );
        pub fn residual_forward_bf16(
            out: *mut bf16,
            a: *const bf16,
            b: *const bf16,
            B: i32,
            seq_len: i32,
            C: i32,
        );
        pub fn residual_forward_f16(
            out: *mut f16,
            a: *const f16,
            b: *const f16,
            B: i32,
            seq_len: i32,
            C: i32,
        );
    }

    // ---- Attention ----
    unsafe extern "C" {
        pub fn attention_forward_f32(
            out: *mut f32,
            att: *mut f32,
            qkv: *const f32,
            B: i32,
            seq_len: i32,
            C: i32,
            NH: i32,
        );
        pub fn attention_forward_bf16(
            out: *mut bf16,
            att: *mut bf16,
            qkv: *const bf16,
            B: i32,
            seq_len: i32,
            C: i32,
            NH: i32,
        );
        pub fn attention_forward_f16(
            out: *mut f16,
            att: *mut f16,
            qkv: *const f16,
            B: i32,
            seq_len: i32,
            C: i32,
            NH: i32,
        );
    }

    // ---- CrossEntropy ----
    unsafe extern "C" {
        pub fn crossentropy_forward_f32(
            losses: *mut f32,
            probs: *mut f32,
            logits: *const f32,
            targets: *const i32,
            B: i32,
            seq_len: i32,
            V: i32,
        );
        pub fn crossentropy_forward_bf16(
            losses: *mut f32,
            probs: *mut bf16,
            logits: *const bf16,
            targets: *const i32,
            B: i32,
            seq_len: i32,
            V: i32,
        );
        pub fn crossentropy_forward_f16(
            losses: *mut f32,
            probs: *mut f16,
            logits: *const f16,
            targets: *const i32,
            B: i32,
            seq_len: i32,
            V: i32,
        );
    }
    
    // ---- CUDA Runtime API ----
    
    unsafe extern "C" {
    	pub fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> i32;
    	pub fn cudaFree(devPtr: *mut u8) -> i32;
    	pub fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: i32) ->32;
    }
    
    pub const MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const MEMCPY_DEVICE_TO_HOST: i32 = 2;
    pub const MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
}
