#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "Config.h"
#include "GemmLoader.cuh"

template<typename Config>
__global__ void Gemm (
		const typename Config::In	*__restrict__ A,
		const typename Config::In	*__restrict__ B,
		typename Config::Out		*__restrict__ C,
		const typename Config::Out	*__restrict__ bias,
		const float	alpha, const float beta,
		int M, int N, int K
) {
		using namespace nvcuda;
		int tid = threadIdx.x;
		int block_row = blockIdx.y * Config::BM, block_col = blockIdx.x * Config::BN;
		int warp_id = threadIdx.x / 32, warp_row = warp_id / Config::WARPS_N, warp_col = warp_id % Config::WARPS_N;
		int wm0 = warp_row * Config::WM, wn0 = warp_col * Config::WN;
		
		__shared__ typename Config::In shared_a[2][Config::BM][Config::BK + Config::PAD], shared_b[2][Config::BK][Config::BN + Config::PAD];
		
		wmma::fragment<wmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::Acc> c_frag[Config::FM][Config::FN];
		for (int m = 0; m < Config::FM; ++m) {
				for (int n = 0; n < Config::FN; ++n) {
						wmma::fill_fragment(c_frag[m][n], 0);
				}
		}
		
		TileLoader<Config> loader {
				A, B, tid,
				block_row, block_col,
				M, N, K
		};
		
		int cur = 0; float4 ra[Config::A_LPT], rb[Config::B_LPT];
		loader.gmem_to_reg(0, ra, rb);
		loader.reg_to_smem(0, ra, rb, shared_a, shared_b);
		__syncthreads();
		
		for (int i = 0; i < K; i += Config::BK) {
				int next_i = i + Config::BK;
				float4 next_a[Config::A_LPT], next_b[Config::B_LPT];
				if (next_i < K) loader.gmem_to_reg(next_i, next_a, next_b);
				
				wmma::fragment<wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> a_frag[Config::FM];
				wmma::fragment<wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, typename Config::In, wmma::row_major> b_frag[Config::FN];
				
				for (int k = 0; k < Config::BK; k += Config::WMMA_K) {
						for (int m = 0; m < Config::FM; ++m) {
								wmma::load_matrix_sync(a_frag[m], &shared_a[cur][wm0 + m * Config::WMMA_M][k], Config::BK + Config::PAD);
						}
						for (int n = 0; n < Config::FN; ++n) {
								wmma::load_matrix_sync(b_frag[n], &shared_b[cur][k][wn0 + n * Config::WMMA_N], Config::BN + Config::PAD);
						}
						for (int m = 0; m < Config::FM; ++m) {
								for (int n = 0; n < Config::FN; ++n) {
										wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
								}
						}
				}
				
				if (next_i < K) {
						__syncthreads();
						loader.reg_to_smem(cur ^ 1, next_a, next_b, shared_a, shared_b);
						__syncthreads();
						cur ^= 1;
				}
		}
		__syncthreads();
		
		auto *smem_base = reinterpret_cast<typename Config::Acc*>(&shared_a[0][0][0]);
		typename Config::Acc *warp_c = smem_base + warp_id * (Config::WMMA_M * Config::WMMA_N);
		
		int lane = tid &31;
		
		for (int m = 0; m < Config::FM; ++m) {
				for (int n = 0; n < Config::FN; ++n) {
						wmma::store_matrix_sync(warp_c, c_frag[m][n], Config::WMMA_N, wmma::mem_row_major);
						__syncwarp();
						int tile_row = block_row + wm0 + m * Config::WMMA_M;
						int tile_col = block_col + wn0 + n * Config::WMMA_N;
						
						for (int e = lane; e < Config::WMMA_M * Config::WMMA_N; e += 32) {
								int r = e / Config::WMMA_N, c = e % Config::WMMA_N;
								int gr = tile_row + r, gc = tile_col + c;
								if (gr < M && gc < N) {
										int idx = gr * N + gc;
										typename  Config::Acc v = alpha * warp_c[r * Config::WMMA_N + c];
										if (beta != 0) v += beta * static_cast<typename Config::Acc>(C[idx]);
										if (bias)	   v += static_cast<typename Config::Acc>(bias[gc]);
										C[idx] = static_cast<typename Config::Out>(v); 
								}
						}
						__syncwarp();
				}
		}	
}

template<typename Config>
void launch_Gemm_forward(
		const typename Config::In *A, 
		const typename Config::In *B,
		typename Config::Out *C,
		const typename Config::Out *bias,
		float alpha, float beta, int M, int N, int K, cudaStream_t s
) {
		dim3 block(Config::THREADS);
		dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
		Gemm<Config><<<grid, block, 0, s>>>(A, B, C, bias, alpha, beta, M, N, K);
}

#define GEMM_FORWARD(name, InT, OutT)                                         \
extern "C" void gemm_forward_##name(                                          \
		const InT *A, const InT *B, OutT *C, const OutT *bias,               \
		float alpha, float beta, int M, int N, int K, cudaStream_t s          \
) {                                                                           \
		launch_Gemm_forward<GemmConfig<InT, OutT>>(                           \
				A, B, C, bias, alpha, beta, M, N, K, s);                      \
}

GEMM_FORWARD(bf16,      __nv_bfloat16, __nv_bfloat16)
GEMM_FORWARD(bf16_f32,  __nv_bfloat16, float)
GEMM_FORWARD(f16,       half,          half)
GEMM_FORWARD(f16_f32,   half,          float)
GEMM_FORWARD(i8_i32,    signed char,   int)

#undef GEMM_FORWARD