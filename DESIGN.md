# llm_rs 架构设计

## 整体分层

```
kernel    CUDA FFI，声明所有 CUDA kernel 和 runtime API
backend   硬件抽象，Device（显存操作）+ Ops（计算操作）+ Backend（统一入口）
tensor    Tensor（显存 view）+ Arena（显存池）
model     通用 module（Linear, LayerNorm, GELU, Residual, Attention, Embedding）+ 具体模型（GPT2, ...）
```

依赖方向：kernel ← backend ← tensor ← model

## Arena 显存池

核心思想：所有 GPU 显存由 Arena 统一管理，一次 cudaMalloc，避免频繁小分配。

### 两阶段设计

阶段一（记账）：Arena 创建后不分配显存。module 创建时调用 arena.alloc(size)，Arena 只累加 offset，返回偏移量。

阶段二（分配）：所有 module 声明完毕后，调用 arena.finalize()，此时 Arena 知道总量，执行一次 cudaMalloc。

### Tensor 存偏移量

Tensor 不直接存 GPU 指针，而是存 Arena 引用 + 偏移量：

```rust
struct Tensor<D: Dtype> {
    arena: Arc<Arena<D>>,
    offset: usize,
    shape: Vec<usize>,
}
```

finalize() 之后，Tensor 通过 arena.resolve(offset) 获取真实 GPU 指针。

### 使用流程

```rust
let arena = Arena::new(backend);
let linear = Linear::new(&arena, c, 3*c, b, t);   // 记账
let gelu = GELU::new(&arena, b, t, c);             // 记账
arena.finalize();                                    // 一次 malloc
linear.forward(&x);                                 // 正常使用
```

## Module 设计

每个 op 都是一个 module，统一持有权重（可选）和输出 buffer：

```
Linear      weight, bias, output
LayerNorm   gamma, beta, output
Embedding   wte, wpe, output          (forward 直接调 backend.ops)
GELU        output                    (无权重)
Residual    output                    (无权重)
Attention   output, att_scores        (无权重，att_scores 是中间结果)
```

每个 module 有两个方法：
- new(arena, ...) — 从 Arena 分配空间，构造 module
- forward(x) — 读输入，结果写入自己的 output

### 链式调用

每个 module 的 .output 是下一个 module 的输入：

```rust
self.ln1.forward(x);
self.attn_qkv.forward(&self.ln1.output);
self.attention.forward(&self.attn_qkv.output);
self.attn_proj.forward(&self.attention.output);
self.residual1.forward(x, &self.attn_proj.output);
```

## Tensor 方法

Tensor 上实现所有计算方法，module 的 forward 调用这些方法：

- gemm(rhs, bias, alpha, beta, out) — 矩阵乘法
- gelu(out) — 激活函数
- residual(rhs, out) — 残差连接
- attention(att, nh, out) — 自注意力
- layernorm(gamma, beta, mean, rstd, eps, out) — 层归一化

embedding 特殊（输入是 i32），由 Embedding module 直接调 backend.ops。

## 模型层

通用 module 放 model/src/module/，具体模型各自一个文件：

```
model/src/
  module/
    linear.rs
    layernorm.rs
    gelu.rs
    residual.rs
    attention.rs
    embedding.rs
  gpt2.rs
```

GPT2 组合通用 module，定义自己的结构和 forward。
未来 LLaMA、Qwen 等复用同一套 module，只需写各自的结构文件。

## 数据流（GPT2 推理）

```
token_ids
  → Embedding → [B, T, C]
  → N x TransformerBlock:
      → LayerNorm → Linear(QKV) → Attention → Linear(Proj) → Residual
      → LayerNorm → Linear(FC) → GELU → Linear(Proj) → Residual
  → LayerNorm
  → Linear(logits) → [B, T, V]
  → 采样 → next token
```

## 待实现

- Arena 两阶段（记账 + finalize）
- Tensor 改为存偏移量
- 各 module 的 new 和 forward
- GPT2 struct + forward
- 权重加载（读 checkpoint，拷贝到 Arena）
- Tokenizer
- 生成循环（forward → 采样 → 重复）
