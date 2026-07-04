#include <cassert>
#include "embedding/add_pos_embedding.cuh"
#include "LayerNorm/LayerNorm.cuh"
#include "Gemm/Gemm.cuh"
#include "Gemm/Gemm_f32.cuh"
#include "activation/GELU.cuh"
#include "residual/Residual.cuh"
#include "attention/Attention.cuh"
#include "loss/CrossEntropy.cuh"
#include "transpose/Transpose.cuh"
#include "attention/Gather_kv.cuh"
#include "attention/Attention_decode.cuh"
#include "LayerNorm/RMSNorm.cuh"
#include "embedding/RoPE.cuh"
#include "attention/GQAttention_prefill.cuh"
#include "attention/GQAttention_decode.cuh"
#include "activation/SwiGLU.cuh"

// gemm 独占一个 extern 分发符号:f32 走 Gemm_f32 的独立 launcher,
// bf16/f16 走 Gemm.cuh 的 tensor-core launcher,两条路径只有在本 TU 同时可见
// dtype 契约: 0=f32 1=bf16 2=f16,与 backend Dtype::TAG 一致
extern "C" void gemm_forward(
		int dtype, const void *A, const void *B, void *C, const void *bias,
		float alpha, float beta, int M, int N, int K, cudaStream_t s
) {
	switch (dtype) {
		case 0: launch_gemm_f32_forward((const float *)A, (const float *)B, (float *)C, (const float *)bias, alpha, beta, M, N, K, s); break;
		case 1: launch_Gemm_forward<GemmConfig<__nv_bfloat16, __nv_bfloat16>>((const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (__nv_bfloat16 *)C, (const __nv_bfloat16 *)bias, alpha, beta, M, N, K, s); break;
		case 2: launch_Gemm_forward<GemmConfig<half, half>>((const half *)A, (const half *)B, (half *)C, (const half *)bias, alpha, beta, M, N, K, s); break;
		default: assert(false && "unknown dtype");
	}
}