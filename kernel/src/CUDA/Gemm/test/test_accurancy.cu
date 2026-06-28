#include "../Gemm.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

void cpu_gemm(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
              int M, int N, int K) {
	//for (int m = 0; m < M; ++m) {
	//	for (int n = 0; n < N; ++n) {
	//		float sum = 0.0f;
	//		for (int k = 0; k < K; ++k) {
	//			sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[k * N + n]);
	//		}
	//		C[m * N + n] = static_cast<__nv_bfloat16>(sum);
	//	}
	//}
}

int main() {
	int M = 12288, N = 12288, K = 12288;
	size_t sA = sizeof(__nv_bfloat16) * M * K;
	size_t sB = sizeof(__nv_bfloat16) * K * N;
	size_t sC = sizeof(__nv_bfloat16) * M * N;

	__nv_bfloat16 *a_host = (__nv_bfloat16*)malloc(sA);
	__nv_bfloat16 *b_host = (__nv_bfloat16*)malloc(sB);
	__nv_bfloat16 *c_cpu  = (__nv_bfloat16*)malloc(sC);
	__nv_bfloat16 *c_gpu  = (__nv_bfloat16*)malloc(sC);

	memset(a_host, 0x3f, sA);
	memset(b_host, 0x3f, sB);

	cpu_gemm(a_host, b_host, c_cpu, M, N, K);

	__nv_bfloat16 *a_device, *b_device, *c_device;
	CUDA_CHECK(cudaMalloc(&a_device, sA));
	CUDA_CHECK(cudaMalloc(&b_device, sB));
	CUDA_CHECK(cudaMalloc(&c_device, sC));
	CUDA_CHECK(cudaMemset(c_device, 0, sC));

	CUDA_CHECK(cudaMemcpy(a_device, a_host, sA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(b_device, b_host, sB, cudaMemcpyHostToDevice));

	using Cfg = GemmConfig<__nv_bfloat16, __nv_bfloat16>;
	cudaStream_t stream = nullptr;

	float ms = 0.0f;
	TIME_MS(ms, 10, 50,
		launch_Gemm_forward<Cfg>(a_device, b_device, c_device, nullptr, 1.0f, 0.0f, M, N, K, stream));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(c_gpu, c_device, sC, cudaMemcpyDeviceToHost));
	printf("time: %.3f ms\n", ms);

	//if (matricesEqual(c_gpu, c_cpu, M * N, 1e-1f))
	//	printf("SUCCESS\n");
	//else
	//	printf("WRONG\n");
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetStream(handle, stream);
	float alpha = 1.0f, beta = 0.0f;

	CUDA_CHECK(cudaMemset(c_device, 0, sC));
	float ms2 = 0.0f;
	TIME_MS(ms2, 10, 50, cublasGemmEx(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		N, M, K,
		&alpha,
		b_device, CUDA_R_16BF, N,
		a_device, CUDA_R_16BF, K,
		&beta,
		c_device, CUDA_R_16BF, N,
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT
	));
	printf("cuBLAS time: %.3f ms\n", ms2);
	
	printf("%f", ms2/ms);

	CUDA_CHECK(cudaMemcpy(c_gpu, c_device, sC, cudaMemcpyDeviceToHost));
	//if (matricesEqual(c_gpu, c_cpu, M * N, 1e-1f))
	//	printf("cuBLAS: SUCCESS\n");
	//else
	//	printf("cuBLAS: WRONG\n");

	cublasDestroy(handle);

	cudaFree(a_device); cudaFree(b_device); cudaFree(c_device);
	free(a_host); free(b_host); free(c_cpu); free(c_gpu);
}
