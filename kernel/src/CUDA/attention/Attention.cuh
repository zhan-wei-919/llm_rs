#include "../reduce/Reduce.cuh"
#include <cmath>
#include <cstddef>

template<typename T>
__global__ void attention(
		T		*__restrict__		out,		// [B, T, C]
		T		*__restrict__		att,		// [B, NH, T, T]
		const T *__restrict__		qkv,		// [B, T, 3C]
		int B, int seq_len, int C, int NH
) {
        int i = blockIdx.x % seq_len;                                 // 第几个query行
        int h = (blockIdx.x / seq_len) % NH;                          // 第几个head
        int b = (blockIdx.x) / (seq_len * NH);                        // 第几个batch

        const int HS = C / NH;
        const float scale = rsqrtf((float)HS);          // 1/√HS
        const T *q_i = qkv + (long)(b * seq_len + i) * 3 * C + 0 * C + h * HS;

        __shared__ float scores[1024];
        
        // 计算 scores[i][j] = q[i]*k[j]
        for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        		const T *k_j = qkv + (long)(b * seq_len + i) * 3 * C + 1 * C + h * HS;
        		float acc = 0.0f;
        		for (int d = 0; d < HS; ++d) {
        				acc += static_cast<float>(q_i[d]) * static_cast<float>(k_j[d]);
        		}
        		scores[j] = s * scale;
        }
        __syncthreads();
        
        // 计算 softmax
        T local_max = -INFINITY;
        for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        		local_max = device_max(local_max, scores[j]);
        }
        float row_max = static_cast<float>(block_max(local_max));
        float local_sum = 0.0f;
        for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        		float e = expf(scores[j] - row_max);
        		scores[j] = e;
        		local_sum += e;
        }
        float Z = block_sum(local_sum);
        float inv_z = 1.0f / Z;
        __syncthreads();
        
        // 计算得分
        T *out_i = out + (size_t)(b * T + i) * C + h * HS;
        for (int d = threadIdx.x; d < HS; d += blockDim.x) {
        		float acc =0.0f;
        		for (int j = 0; j <= i; ++j) {
        				float tmp = scores[j] * inv_z;
        				const T *v_j = qkv + (size_t)(b * T + j) * 3 * C + 2 * C + h + HS;
        				acc += static_cast<float>(v_j[d]) * tmp;
        		}
        		out_i[d] = static_cast<T>(acc);
        }
        
        // 写回
        
        T *att_i = att + ((size_t)(b * T + h) * T + i) * T;
        for (int j = threadIdx.x; j < T; j += blockDim.x) {
        		float tmp = (j <= i)? scores[j] : 0.0f;
        		att_i[j] = static_cast<T>(tmp);
        }
        
		
}

template<typename T>
void launch_attention_forward(
		T		*__restrict__		out,		// [B, T, C]
		T		*__restrict__		att,		// [B, NH, T, T]
		const T *__restrict__		qkv,		// [B, T, 3C]
		int B, int T, int C, int NH
) {
		dim3 grid = B * NH * T;
		dim3 block = 256;
		attention<<<grid, block>>>(out, att, qkv, B, T, C, NH);
}