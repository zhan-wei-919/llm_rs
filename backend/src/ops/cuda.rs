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
    
    fn gather_kv_forward(
    	&self,
    	k_cache: *mut u8,
    	v_cache: *mut u8,
    	qkv: *const u8,
    	t: i32,
    	c: i32,
    	dst_start: i32,
    ) {
    	unsafe {
    		kernel::cuda::gather_kv_forward_f32(
    			k_cache as *mut f32, 
    			v_cache as *mut f32, 
    			qkv as *const f32, 
    			t, c, dst_start
    		);
    	}
    }
    
    fn attention_decode_forward(
    	&self,
    	out: *mut u8,
    	qkv: *const u8,
   		k_cache: *const u8,
   		v_cache: *const u8,
   		cur_len: i32, c: i32, nh: i32
    ){
    	unsafe {
    		kernel::cuda::attention_decode_forward_f32(
    			out as *mut f32,
    			qkv as *const f32,
    			k_cache as *const f32,
    			v_cache as *const f32,
    			cur_len, c, nh
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
    
    fn gather_kv_forward(
    	&self,
    	k_cache: *mut u8,
    	v_cache: *mut u8,
    	qkv: *const u8,
    	t: i32,
    	c: i32,
    	dst_start: i32,
    ) {
    	unsafe {
    		kernel::cuda::gather_kv_forward_bf16(
    			k_cache as *mut u16, 
    			v_cache as *mut u16, 
    			qkv as *const u16, 
    			t, c, dst_start
    		);
    	}
    }
    
    fn attention_decode_forward(
    	&self,
    	out: *mut u8,
    	qkv: *const u8,
   		k_cache: *const u8,
   		v_cache: *const u8,
   		cur_len: i32, c: i32, nh: i32
    ){
    	unsafe {
    		kernel::cuda::attention_decode_forward_bf16(
    			out as *mut u16,
    			qkv as *const u16,
    			k_cache as *const u16,
    			v_cache as *const u16,
    			cur_len, c, nh
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
    
    fn gather_kv_forward(
    	&self,
    	k_cache: *mut u8,
    	v_cache: *mut u8,
    	qkv: *const u8,
    	t: i32,
    	c: i32,
    	dst_start: i32,
    ) {
    	unsafe {
    		kernel::cuda::gather_kv_forward_f16(
    			k_cache as *mut u16, 
    			v_cache as *mut u16, 
    			qkv as *const u16, 
    			t, c, dst_start
    		);
    	}
    }
    
    fn attention_decode_forward(
    	&self,
    	out: *mut u8,
    	qkv: *const u8,
   		k_cache: *const u8,
   		v_cache: *const u8,
   		cur_len: i32, c: i32, nh: i32
    ){
    	unsafe {
    		kernel::cuda::attention_decode_forward_f16(
    			out as *mut u16,
    			qkv as *const u16,
    			k_cache as *const u16,
    			v_cache as *const u16,
    			cur_len, c, nh
    		);
    	}
    }
}


#[cfg(test)]
mod tests{
	use crate::{Backend, F32};
	
	#[test]
	#[ignore]
	fn test_attention_decode() {
		let backend = Backend::<F32>::cuda();
		let device = &backend.device;
		let ops = &backend.ops;
		
		
		for t in [1, 7, 256, 257] {
			let qkv_input: Vec<f32> = (0..t * 3 * 768).map(|i| ((i * 37) % 256) as f32 / 256.0 - 0.5).collect();
			let qkv_ptr: *const u8 = qkv_input.as_ptr() as *const u8;
			let att_out = device.alloc(1*12*t*t*4);
			let attention_out = device.alloc(1*t*768*4);
			let attn_qkv = device.alloc(1*t*768*3*4);
			device.copy_from_host_to_device(attn_qkv, qkv_ptr, 1*t*768*3*4);
			ops.attention_forward(attention_out, att_out, attn_qkv, 1, t as i32, 768, 12);
			let mut sample1 = vec![0f32; 1*t*768];
			let sample1_ptr: *mut u8 = sample1.as_mut_ptr() as *mut u8;
			device.copy_from_device_to_host(sample1_ptr, attention_out, 1*t*768*4);
			
			let k_cache = device.alloc(1*t*768*4);
			let v_cache = device.alloc(1*t*768*4);
			
			let mut sample2 = vec![0f32; 768];
			let sample2_ptr: *mut u8 = sample2.as_mut_ptr() as *mut u8;
			
			for i in 0..t {
				let row = unsafe { (attn_qkv as *const u8).add(i * 3 * 768 * 4) };
				ops.gather_kv_forward(k_cache, v_cache, row, 1, 768, i as i32);
				ops.attention_decode_forward(attention_out, row, k_cache, v_cache, (i + 1) as i32, 768, 12);
				device.copy_from_device_to_host(sample2_ptr, attention_out, 768*4);
    			assert_eq!(sample2, sample1[i * 768..(i + 1) * 768], "t={t}, 位置 i={i} 分叉");
			}

		}
	}
}