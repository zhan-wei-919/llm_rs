use crate::dtype::{BF16, F16, F32};
use crate::ops::Ops;


pub struct CudaOps;

impl Ops<F32> for CudaOps {
    fn embedding_forward(
        &self,
        out: *mut u8,
        token_ids: *const i32,
        token_table: *const u8,
        pos_table: *const u8,
        b: i32,
        t: i32,
        c: i32,
    ) {
        unsafe {
            kernel::cuda::embedding_forward_f32(
                out as *mut f32,
                token_ids,
                token_table as *const f32,
                pos_table as *const f32,
                b,
                t,
                c,
            );
        }
    }

    fn layernorm_forward(
        &self,
        out: *mut u8,
        mean_out: *mut f32,
        rstd_out: *mut f32,
        x: *const u8,
        gamma: *const u8,
        beta: *const u8,
        b: i32,
        t: i32,
        c: i32,
        eps: f32,
    ) {
        unsafe {
            kernel::cuda::layernorm_forward_f32(
                out as *mut f32,
                mean_out,
                rstd_out,
                x as *const f32,
                gamma as *const f32,
                beta as *const f32,
                b,
                t,
                c,
                eps,
            );
        }
    }

    fn gelu_forward(&self, y: *mut u8, x: *const u8, n: i32) {
        unsafe {
            kernel::cuda::gelu_forward_f32(y as *mut f32, x as *const f32, n, std::ptr::null_mut());
        }
    }

    fn attention_forward(
        &self,
        out: *mut u8,
        att: *mut u8,
        qkv: *const u8,
        b: i32,
        t: i32,
        c: i32,
        nh: i32,
    ) {
        unsafe {
            kernel::cuda::attention_forward_f32(
                out as *mut f32,
                att as *mut f32,
                qkv as *const f32,
                b,
                t,
                c,
                nh,
            );
        }
    }

    fn crossentropy_forward(
        &self,
        losses: *mut f32,
        probs: *mut u8,
        logits: *const u8,
        targets: *const i32,
        b: i32,
        t: i32,
        v: i32,
    ) {
        unsafe {
            kernel::cuda::crossentropy_forward_f32(
                losses,
                probs as *mut f32,
                logits as *const f32,
                targets,
                b,
                t,
                v,
            );
        }
    }

    fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c: i32) {
        unsafe {
            kernel::cuda::residual_forward_f32(
                out as *mut f32,
                x1 as *const f32,
                x2 as *const f32,
                b,
                t,
                c,
            );
        }
    }

    fn gemm_forward(
        &self,
        a: *const u8,
        b: *const u8,
        c: *mut u8,
        bias: *const u8,
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
    ) {
        unsafe {
            kernel::cuda::gemm_forward_f32(
                a as *const f32,
                b as *const f32,
                c as *mut f32,
                bias as *const f32,
                alpha,
                beta,
                m,
                n,
                k,
                std::ptr::null_mut(),
            );
        }
    }
    
    fn transpose_forward(
    	&self,
    	out: *mut u8,
    	input: *const u8,
    	r: i32,
    	c: i32,
    	out_stride: i32,
    ){
    	unsafe {
    		kernel::cuda::transpose_forward_f32(
    			out as *mut f32,
    			input as *const f32,
    			r,
    			c,
    			out_stride,
    		);
    	}
    }
}

impl Ops<BF16> for CudaOps {
    fn embedding_forward(
        &self,
        out: *mut u8,
        token_ids: *const i32,
        token_table: *const u8,
        pos_table: *const u8,
        b: i32,
        t: i32,
        c: i32,
    ) {
        unsafe {
            kernel::cuda::embedding_forward_bf16(
                out as *mut u16,
                token_ids,
                token_table as *const u16,
                pos_table as *const u16,
                b,
                t,
                c,
            );
        }
    }

    fn layernorm_forward(
        &self,
        out: *mut u8,
        mean_out: *mut f32,
        rstd_out: *mut f32,
        x: *const u8,
        gamma: *const u8,
        beta: *const u8,
        b: i32,
        t: i32,
        c: i32,
        eps: f32,
    ) {
        unsafe {
            kernel::cuda::layernorm_forward_bf16(
                out as *mut u16,
                mean_out,
                rstd_out,
                x as *const u16,
                gamma as *const u16,
                beta as *const u16,
                b,
                t,
                c,
                eps,
            );
        }
    }

    fn gelu_forward(&self, y: *mut u8, x: *const u8, n: i32) {
        unsafe {
            kernel::cuda::gelu_forward_bf16(
                y as *mut u16,
                x as *const u16,
                n,
                std::ptr::null_mut(),
            );
        }
    }

    fn attention_forward(
        &self,
        out: *mut u8,
        att: *mut u8,
        qkv: *const u8,
        b: i32,
        t: i32,
        c: i32,
        nh: i32,
    ) {
        unsafe {
            kernel::cuda::attention_forward_bf16(
                out as *mut u16,
                att as *mut u16,
                qkv as *const u16,
                b,
                t,
                c,
                nh,
            );
        }
    }

    fn crossentropy_forward(
        &self,
        losses: *mut f32,
        probs: *mut u8,
        logits: *const u8,
        targets: *const i32,
        b: i32,
        t: i32,
        v: i32,
    ) {
        unsafe {
            kernel::cuda::crossentropy_forward_bf16(
                losses,
                probs as *mut u16,
                logits as *const u16,
                targets,
                b,
                t,
                v,
            );
        }
    }

    fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c: i32) {
        unsafe {
            kernel::cuda::residual_forward_bf16(
                out as *mut u16,
                x1 as *const u16,
                x2 as *const u16,
                b,
                t,
                c,
            );
        }
    }

    fn gemm_forward(
        &self,
        a: *const u8,
        b: *const u8,
        c: *mut u8,
        bias: *const u8,
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
    ) {
        unsafe {
            kernel::cuda::gemm_forward_bf16(
                a as *const u16,
                b as *const u16,
                c as *mut u16,
                bias as *const u16,
                alpha,
                beta,
                m,
                n,
                k,
                std::ptr::null_mut(),
            );
        }
    }
    
    fn transpose_forward(
    	&self,
    	out: *mut u8,
    	input: *const u8,
    	r: i32,
    	c: i32,
    	out_stride: i32,
    ){
    	unsafe {
    		kernel::cuda::transpose_forward_bf16(
    			out as *mut u16,
    			input as *const u16,
    			r,
    			c,
    			out_stride,
    		);
    	}
    }
}

impl Ops<F16> for CudaOps {
    fn embedding_forward(
        &self,
        out: *mut u8,
        token_ids: *const i32,
        token_table: *const u8,
        pos_table: *const u8,
        b: i32,
        t: i32,
        c: i32,
    ) {
        unsafe {
            kernel::cuda::embedding_forward_f16(
                out as *mut u16,
                token_ids,
                token_table as *const u16,
                pos_table as *const u16,
                b,
                t,
                c,
            );
        }
    }

    fn layernorm_forward(
        &self,
        out: *mut u8,
        mean_out: *mut f32,
        rstd_out: *mut f32,
        x: *const u8,
        gamma: *const u8,
        beta: *const u8,
        b: i32,
        t: i32,
        c: i32,
        eps: f32,
    ) {
        unsafe {
            kernel::cuda::layernorm_forward_f16(
                out as *mut u16,
                mean_out,
                rstd_out,
                x as *const u16,
                gamma as *const u16,
                beta as *const u16,
                b,
                t,
                c,
                eps,
            );
        }
    }

    fn gelu_forward(&self, y: *mut u8, x: *const u8, n: i32) {
        unsafe {
            kernel::cuda::gelu_forward_f16(y as *mut u16, x as *const u16, n, std::ptr::null_mut());
        }
    }

    fn attention_forward(
        &self,
        out: *mut u8,
        att: *mut u8,
        qkv: *const u8,
        b: i32,
        t: i32,
        c: i32,
        nh: i32,
    ) {
        unsafe {
            kernel::cuda::attention_forward_f16(
                out as *mut u16,
                att as *mut u16,
                qkv as *const u16,
                b,
                t,
                c,
                nh,
            );
        }
    }

    fn crossentropy_forward(
        &self,
        losses: *mut f32,
        probs: *mut u8,
        logits: *const u8,
        targets: *const i32,
        b: i32,
        t: i32,
        v: i32,
    ) {
        unsafe {
            kernel::cuda::crossentropy_forward_f16(
                losses,
                probs as *mut u16,
                logits as *const u16,
                targets,
                b,
                t,
                v,
            );
        }
    }

    fn residual_forward(&self, out: *mut u8, x1: *const u8, x2: *const u8, b: i32, t: i32, c: i32) {
        unsafe {
            kernel::cuda::residual_forward_f16(
                out as *mut u16,
                x1 as *const u16,
                x2 as *const u16,
                b,
                t,
                c,
            );
        }
    }

    fn gemm_forward(
        &self,
        a: *const u8,
        b: *const u8,
        c: *mut u8,
        bias: *const u8,
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
    ) {
        unsafe {
            kernel::cuda::gemm_forward_f16(
                a as *const u16,
                b as *const u16,
                c as *mut u16,
                bias as *const u16,
                alpha,
                beta,
                m,
                n,
                k,
                std::ptr::null_mut(),
            );
        }
    }
    
    fn transpose_forward(
    	&self,
    	out: *mut u8,
    	input: *const u8,
    	r: i32,
    	c: i32,
    	out_stride: i32,
    ){
    	unsafe {
    		kernel::cuda::transpose_forward_f16(
    			out as *mut u16,
    			input as *const u16,
    			r,
    			c,
    			out_stride,
    		);
    	}
    }
}

