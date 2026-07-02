#pragma once

template<typename T>
__global__ void gather_kv (
		T		*k_cache,
		T		*v_cache,
		T const *qkv,
		int t, int c, int dst_start
) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= t * c) return;
		int i = idx / c;
		int j = idx % c;
		k_cache[(dst_start + i) * c + j] = qkv[i * 3 * c + c + j];
		v_cache[(dst_start + i) * c + j] = qkv[i * 3 * c + 2 * c + j];
}

template<typename T>
void launch_gather_kv_forward (
		T		*k_cache,
		T		*v_cache,
		T const *qkv,
		int t, int c, int dst_start
) {
		int grid = (t * c + 255) / 256;
		int block = 256;
		gather_kv<T><<<grid, block>>>(k_cache, v_cache, qkv, t, c, dst_start);
}

#define GATHER_KV_FORWARD(name, InT)                                          \
extern "C" void gather_kv_forward_##name(                                     \
		InT *k_cache, InT *v_cache, const InT *qkv, int t, int c, int dst_start   \
) {                                                                           \
		launch_gather_kv_forward(k_cache, v_cache, qkv, t, c, dst_start);                  \
}

GATHER_KV_FORWARD(bf16, __nv_bfloat16)
GATHER_KV_FORWARD(f16,  half)
GATHER_KV_FORWARD(f32,  float)

#undef GATHER_KV_FORWARD