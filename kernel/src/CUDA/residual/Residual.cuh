#pragma once
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void residual(T *out, const T *a, const T *b, int C) {		// [B, T, C]
		constexpr int VEC = sizeof(float4) / sizeof(T);
		for (int c = threadIdx.x * VEC; c + VEC <= C; c += blockDim.x * VEC) {
				float4 a_v = *reinterpret_cast<const float4*>(&a[blockIdx.x * C + c]);
				T *a_elems = reinterpret_cast<T*>(&a_v);
				float4 b_v = *reinterpret_cast<const float4*>(&b[blockIdx.x * C + c]);
				T *b_elems = reinterpret_cast<T*>(&b_v);
				float4 out_v;
				T *o_elems = reinterpret_cast<T*>(&out_v);

				for(int j = 0; j < VEC; ++j) {
						float tmp = static_cast<float>(a_elems[j]) + static_cast<float>(b_elems[j]);
						o_elems[j] = static_cast<T>(tmp);
				}
				*reinterpret_cast<float4*>(&out[blockIdx.x * C + c]) = out_v;
		}
		for (int c = C / VEC * VEC + threadIdx.x; c < C; c += blockDim.x) {
				out[blockIdx.x * C + c] = static_cast<T>(static_cast<float>(a[blockIdx.x * C + c]) + static_cast<float>(b[blockIdx.x * C + c]));
		}
}

template<typename T>
void launch_residual_forward(T *out, const T *a, const T *b, int B, int seq_len, int C) {
		int grid = B * seq_len;
		int block = 256;
		residual<<<grid, block>>>(out, a, b, C);
}

// dtype 契约: 0=f32 1=bf16 2=f16,与 backend Dtype::TAG 一致
extern "C" void residual_forward(
		int dtype, void *out, const void *a, const void *b,
		int B, int seq_len, int C
) {
	switch (dtype) {
		case 0: launch_residual_forward((float *)out, (const float *)a, (const float *)b, B, seq_len, C); break;
		case 1: launch_residual_forward((__nv_bfloat16 *)out, (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, B, seq_len, C); break;
		case 2: launch_residual_forward((half *)out, (const half *)a, (const half *)b, B, seq_len, C); break;
		default: assert(false && "unknown dtype");
	}
}
