# Step 4: Transformer Block - 完整的 Transformer 解码器块

本步骤实现完整的 Transformer 解码器块，包括 LayerNorm、FFN 和残差连接。

## 目标

1. 实现 Layer Normalization
2. 实现 Feed-Forward Network (FFN)
3. 组装完整的 Transformer Block

## 知识铺垫

### Layer Normalization

Layer Normalization 对每个样本的特征维度进行归一化：

```
y = γ * (x - μ) / √(σ² + ε) + β

其中:
- μ: 均值 (沿特征维度)
- σ²: 方差 (沿特征维度)
- γ, β: 可学习的缩放和偏移参数
- ε: 防止除零的小常数
```

**与 Batch Normalization 的区别：**

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch 维度 | 特征维度 |
| 依赖 batch size | 是 | 否 |
| 适用场景 | CNN | Transformer |

### Feed-Forward Network (FFN)

两层全连接网络，中间有激活函数：

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
```

通常 `ffn_dim = 4 * hidden_dim`，形成"扩展-压缩"结构：

```
[hidden_dim] -> [ffn_dim] -> [hidden_dim]
    64      ->    256    ->     64
```

### GELU 激活函数

GELU (Gaussian Error Linear Unit) 是 Transformer 中常用的激活函数：

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

比 ReLU 更平滑，在负值区域有非零梯度。

### Transformer Block 结构

使用 Pre-LN (Pre-LayerNorm) 结构：

```
┌─────────────────────────────────────┐
│             Input (x)               │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  LayerNorm   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  Attention   │
        └──────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
    │         ┌───────────┘
    │         │
    ▼         ▼
   (+) ◄──── Add (残差连接)
    │
    ▼
┌──────────────┐
│  LayerNorm   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     FFN      │
└──────┬───────┘
       │
    ┌──┴──────────┐
    │             │
    │    ┌────────┘
    │    │
    ▼    ▼
   (+) ◄──── Add (残差连接)
    │
    ▼
┌──────────────┐
│    Output    │
└──────────────┘
```

**Pre-LN vs Post-LN:**
- Pre-LN: 在子层之前归一化 → 训练更稳定
- Post-LN: 在子层之后归一化 → 原始论文结构

### 残差连接

残差连接让梯度更容易流动：

```
output = x + F(x)

梯度: ∂output/∂x = 1 + ∂F(x)/∂x
```

即使 F(x) 的梯度很小，总梯度至少为 1。

## 文件结构

```
step4/
├── Makefile
├── README.md
├── src/
│   ├── tensor.h / tensor.c
│   ├── math_ops.h / math_ops.c
│   ├── tokenizer.h / tokenizer.c
│   ├── embedding.h / embedding.c
│   ├── attention.h / attention.c
│   ├── layernorm.h / layernorm.c    # 新增
│   ├── ffn.h / ffn.c                # 新增
│   ├── transformer.h / transformer.c # 新增
│   └── main.c
```

## 核心 API

### LayerNorm

```c
LayerNorm* layernorm_create(int hidden_dim, float eps);
void layernorm_free(LayerNorm* ln);
void layernorm_forward(LayerNorm* ln, Tensor* input, Tensor* output);
```

### FFN

```c
FFN* ffn_create(int hidden_dim, int ffn_dim);
void ffn_free(FFN* ffn);
void ffn_forward(FFN* ffn, Tensor* input, FFNCache* cache, Tensor* output);
```

### TransformerBlock

```c
TransformerBlock* transformer_block_create(int hidden_dim, int num_heads, int ffn_dim);
void transformer_block_free(TransformerBlock* block);
void transformer_block_init(TransformerBlock* block, float std);
void transformer_block_forward(
    TransformerBlock* block,
    Tensor* input,
    Tensor* mask,
    TransformerCache* cache,
    Tensor* output
);
```

## 使用示例

```c
#include "transformer.h"

int main() {
    int hidden_dim = 64;
    int num_heads = 4;
    int ffn_dim = 256;
    int seq_len = 10;

    // 创建 Transformer Block
    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    transformer_block_init(block, 0.02f);

    // 创建缓存
    TransformerCache* cache = transformer_cache_create(seq_len, hidden_dim, num_heads, ffn_dim);

    // 输入输出
    int shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);

    // 因果掩码
    Tensor* mask = create_causal_mask(seq_len);

    // 前向传播
    transformer_block_forward(block, input, mask, cache, output);

    tensor_free(input);
    tensor_free(output);
    tensor_free(mask);
    transformer_cache_free(cache);
    transformer_block_free(block);
    return 0;
}
```

## 参数数量计算

```
hidden_dim = 64, num_heads = 4, ffn_dim = 256

LayerNorm 1:
  gamma: 64
  beta: 64
  小计: 128

Attention:
  W_q: 64 × 64 = 4,096
  W_k: 64 × 64 = 4,096
  W_v: 64 × 64 = 4,096
  W_o: 64 × 64 = 4,096
  小计: 16,384

LayerNorm 2:
  gamma: 64
  beta: 64
  小计: 128

FFN:
  W1: 64 × 256 = 16,384
  b1: 256
  W2: 256 × 64 = 16,384
  b2: 64
  小计: 33,088

总计: 128 + 16,384 + 128 + 33,088 = 49,728 参数/层
```

## 编译和运行

```bash
make        # 编译
make test   # 运行测试
make clean  # 清理
```

## 预期输出

```
========================================
      miniLLM Step4 测试套件
      Transformer Block 测试
========================================

=== 测试: LayerNorm 创建 ===
[PASS] 创建 LayerNorm 成功
...

=== 测试: LayerNorm 前向传播 ===
[PASS] 1D LayerNorm 输出均值接近 0
[PASS] 1D LayerNorm 输出方差接近 1
...

=== 测试: TransformerBlock 前向传播 ===
[PASS] 输出 shape[0] 正确
[PASS] TransformerBlock 输出不全为零
[PASS] 输出与输入不同
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] LayerNorm 输出均值接近 0，方差接近 1
- [ ] FFN 正确实现扩展-压缩结构
- [ ] 残差连接有效（小权重时输出接近输入）
- [ ] 支持不同序列长度
- [ ] 无内存泄漏

## 下一步

完成 Step 4 后，继续 Step 5 组装完整的 GPT 模型。
