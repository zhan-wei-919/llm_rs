#pragma once
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

/*
x [1536]  ──gate_proj→ [8960] ──silu→ ┐
                                      ├─ ⊙ 逐元素乘 → [8960] ──down_proj→ [1536]
x [1536]  ──up_proj──→ [8960] ────────┘
 */
template<typename T>
__global__ void silu_mul(
	T	*__restrict__	out,		// [N]
	const T *__restrict__	gate,		// [N]
	const T *__restrict__	up,		// [N]
	int N
){
	const int VEC = sizeof(float4) / sizeof(T);
	int base = (blockDim.x * blockIdx.x + threadIdx.x) * VEC;
	if (base >= N) return;
	if (base + VEC <= N) {
		float4 g4 = *reinterpret_cast<const float4*>(&gate[base]);
		float4 u4 = *reinterpret_cast<const float4*>(&up[base]);
		T *g = reinterpret_cast<T*>(&g4);
		T *u = reinterpret_cast<T*>(&u4);
		float4 o4;
		T *o = reinterpret_cast<T*>(&o4);
		#pragma unroll
		for (int i = 0; i < VEC; ++i) {
			float a = static_cast<float>(g[i]);
			float s = a / (1.0f + expf(-a));
			o[i] = static_cast<T>(s * static_cast<float>(u[i]));
		}
		*reinterpret_cast<float4*>(&out[base]) = o4;
	} else {
		for (int i = base; i < N; ++i) {
			float a = static_cast<float>(gate[i]);
			out[i] = static_cast<T>(a / (1.0f + expf(-a)) * static_cast<float>(up[i]));
		}
	}
}

template<typename T>
void launch_silu_mul(T *out, const T *gate, const T *up, int N, cudaStream_t s = nullptr) {
    constexpr int VEC = sizeof(float4) / sizeof(T);
    int threads = 256;
    int blocks = (N + VEC * threads - 1) / (VEC * threads);
    silu_mul<<<blocks, threads, 0, s>>>(out, gate, up, N);
}

// dtype 契约: 0=f32 1=bf16 2=f16,与 backend Dtype::TAG 一致
extern "C" void silu_mul_forward(int dtype, void *out, const void *gate, const void *up, int N, cudaStream_t s) {
	switch (dtype) {
		case 0: launch_silu_mul((float *)out, (const float *)gate, (const float *)up, N, s); break;
		case 1: launch_silu_mul((__nv_bfloat16 *)out, (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up, N, s); break;
		case 2: launch_silu_mul((half *)out, (const half *)gate, (const half *)up, N, s); break;
		default: assert(false && "unknown dtype");
	}
}
