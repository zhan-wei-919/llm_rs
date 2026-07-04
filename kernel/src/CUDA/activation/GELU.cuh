#pragma once
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void GELU(T *y, const T *x, int N) {
		const int VEC = sizeof(float4) / sizeof(T);
		int base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
		if (base >= N) return;

		if (base + VEC <= N) {
				float4 in = *reinterpret_cast<const float4*>(&x[base]);
				T *v = reinterpret_cast<T*>(&in);
				#pragma unroll
				for (int i = 0; i < VEC; ++i) {
						float xc = static_cast<float>(v[i]);
						float inner = 0.7978845608f * (xc + 0.044715f * xc*xc*xc);
						v[i] = static_cast<T>(0.5f * xc * (1.0f + tanhf(inner)));
				}
				*reinterpret_cast<float4*>(&y[base]) = in;
		} else {
				for (int i = base; i < N; ++i) {
						float xc = static_cast<float>(x[i]);
						float inner = 0.7978845608f * (xc + 0.044715f * xc*xc*xc);
                        y[i] = static_cast<T>(0.5f * xc * (1.0f + tanhf(inner)));
				}
		}
}

template<typename T>
void launch_GELU_forward(T *y, const T *x, int N, cudaStream_t s = nullptr) {
		constexpr int VEC = sizeof(float4) / sizeof(T);
		int threads = 256;
		int blocks = (N + VEC * threads - 1) / (VEC * threads);
		GELU<<<blocks, threads, 0, s>>>(y, x, N);
}

// dtype 契约: 0=f32 1=bf16 2=f16,与 backend Dtype::TAG 一致
extern "C" void gelu_forward(int dtype, void *y, const void *x, int N, cudaStream_t s) {
	switch (dtype) {
		case 0: launch_GELU_forward((float *)y, (const float *)x, N, s); break;
		case 1: launch_GELU_forward((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x, N, s); break;
		case 2: launch_GELU_forward((half *)y, (const half *)x, N, s); break;
		default: assert(false && "unknown dtype");
	}
}
