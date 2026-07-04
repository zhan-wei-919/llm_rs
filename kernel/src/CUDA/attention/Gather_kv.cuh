#pragma once
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void gather_kv(T *k_cache, T *v_cache, T const *qkv, int t, int c,
                          int dst_start) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= t * c)
		return;
	int i = idx / c;
	int j = idx % c;
	k_cache[(dst_start + i) * c + j] = qkv[i * 3 * c + c + j];
	v_cache[(dst_start + i) * c + j] = qkv[i * 3 * c + 2 * c + j];
}

template <typename T>
void launch_gather_kv_forward(T *k_cache, T *v_cache, T const *qkv, int t,
                              int c, int dst_start) {
	int grid = (t * c + 255) / 256;
	int block = 256;
	gather_kv<T><<<grid, block>>>(k_cache, v_cache, qkv, t, c, dst_start);
}

// dtype 契约: 0=f32 1=bf16 2=f16,与 backend Dtype::TAG 一致
extern "C" void gather_kv_forward(
		int dtype, void *k_cache, void *v_cache, const void *qkv,
		int t, int c, int dst_start
) {
	switch (dtype) {
		case 0: launch_gather_kv_forward((float *)k_cache, (float *)v_cache, (const float *)qkv, t, c, dst_start); break;
		case 1: launch_gather_kv_forward((__nv_bfloat16 *)k_cache, (__nv_bfloat16 *)v_cache, (const __nv_bfloat16 *)qkv, t, c, dst_start); break;
		case 2: launch_gather_kv_forward((half *)k_cache, (half *)v_cache, (const half *)qkv, t, c, dst_start); break;
		default: assert(false && "unknown dtype");
	}
}
