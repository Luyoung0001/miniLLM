# Step 3: Attention - 多头自注意力机制

本步骤实现 Transformer 的核心组件：多头自注意力机制 (Multi-Head Self-Attention)。

## 目标

1. 实现多头自注意力层
2. 实现因果掩码 (Causal Mask)
3. 支持可变序列长度

## 知识铺垫

### 注意力机制的直觉

注意力机制让模型能够"关注"输入序列中的相关部分：

```
输入: "The cat sat on the mat"
当处理 "sat" 时，模型可能更关注 "cat"（谁在坐）
```

### 自注意力公式

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

- **Q (Query)**: 当前位置想要查询什么
- **K (Key)**: 其他位置提供的索引
- **V (Value)**: 其他位置的实际内容
- **d_k**: Key 的维度，用于缩放防止梯度消失

### 多头注意力

将注意力分成多个"头"，每个头关注不同的特征：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o

其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
```

**为什么用多头？**
- 不同的头可以关注不同的特征
- 一个头关注语法，另一个关注语义
- 增加模型的表达能力

### 因果掩码

在语言模型中，生成位置 i 的 token 时，只能看到位置 0 到 i-1：

```
掩码矩阵 (4x4):
[0,    -∞,   -∞,   -∞  ]
[0,     0,   -∞,   -∞  ]
[0,     0,    0,   -∞  ]
[0,     0,    0,    0  ]
```

添加到 attention scores 后，softmax 会将 -∞ 位置的权重变为 0。

### 计算流程

```
输入: x [seq_len, hidden_dim]

1. 线性投影:
   Q = x @ W_q  [seq_len, hidden_dim]
   K = x @ W_k  [seq_len, hidden_dim]
   V = x @ W_v  [seq_len, hidden_dim]

2. 分成多头:
   Q, K, V -> [num_heads, seq_len, head_dim]

3. 计算注意力分数:
   scores = Q @ K^T / sqrt(head_dim)  [num_heads, seq_len, seq_len]

4. 应用掩码:
   scores = scores + mask

5. Softmax:
   attn = softmax(scores)  [num_heads, seq_len, seq_len]

6. 加权求和:
   out = attn @ V  [num_heads, seq_len, head_dim]

7. 合并多头:
   out = concat(heads)  [seq_len, hidden_dim]

8. 输出投影:
   output = out @ W_o  [seq_len, hidden_dim]
```

### 缩放因子

为什么除以 sqrt(d_k)？

当 d_k 很大时，Q @ K^T 的值会很大，导致 softmax 输出接近 one-hot，梯度变小。
除以 sqrt(d_k) 保持方差稳定。

## 文件结构

```
step3/
├── Makefile
├── README.md
├── src/
│   ├── tensor.h / tensor.c       # (来自 step0)
│   ├── math_ops.h / math_ops.c   # (来自 step0)
│   ├── tokenizer.h / tokenizer.c # (来自 step1)
│   ├── embedding.h / embedding.c # (来自 step2)
│   ├── attention.h               # 注意力层声明
│   ├── attention.c               # 注意力层实现
│   └── main.c                    # 测试程序
```

## 核心 API

### 创建和销毁

```c
// 创建多头注意力层
MultiHeadAttention* attention_create(int hidden_dim, int num_heads);

// 释放
void attention_free(MultiHeadAttention* attn);

// 创建计算缓存
AttentionCache* attention_cache_create(int seq_len, int hidden_dim, int num_heads);

// 释放缓存
void attention_cache_free(AttentionCache* cache);
```

### 初始化和掩码

```c
// 随机初始化权重
void attention_init_random(MultiHeadAttention* attn, float std);

// 创建因果掩码
Tensor* create_causal_mask(int seq_len);
```

### 前向传播

```c
// 多头注意力前向传播
void attention_forward(
    MultiHeadAttention* attn,
    Tensor* input,          // [seq_len, hidden_dim]
    Tensor* mask,           // [seq_len, seq_len] 或 NULL
    AttentionCache* cache,
    Tensor* output          // [seq_len, hidden_dim]
);
```

## 使用示例

```c
#include "attention.h"

int main() {
    int hidden_dim = 64;
    int num_heads = 4;
    int seq_len = 10;

    // 创建注意力层
    MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
    attention_init_random(attn, 0.02f);

    // 创建缓存
    AttentionCache* cache = attention_cache_create(seq_len, hidden_dim, num_heads);

    // 创建输入和输出
    int shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);

    // 创建因果掩码
    Tensor* mask = create_causal_mask(seq_len);

    // 前向传播
    attention_forward(attn, input, mask, cache, output);

    // 获取注意力权重 (用于可视化)
    Tensor* weights = attention_get_weights(cache);

    tensor_free(input);
    tensor_free(output);
    tensor_free(mask);
    attention_cache_free(cache);
    attention_free(attn);
    return 0;
}
```

## 参数数量计算

```
hidden_dim = 64, num_heads = 4

W_q: 64 × 64 = 4,096
W_k: 64 × 64 = 4,096
W_v: 64 × 64 = 4,096
W_o: 64 × 64 = 4,096
------------------------
总计:          16,384 参数
```

## 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Q, K, V 投影 | O(seq_len × hidden_dim²) | O(seq_len × hidden_dim) |
| 注意力分数 | O(seq_len² × hidden_dim) | O(seq_len²) |
| Softmax | O(seq_len²) | O(1) |
| 输出投影 | O(seq_len × hidden_dim²) | O(seq_len × hidden_dim) |

**注意**: 注意力的 O(seq_len²) 复杂度是 Transformer 的主要瓶颈！

## 编译和运行

```bash
# 编译
make

# 运行测试
make test

# 清理
make clean
```

## 预期输出

```
========================================
      miniLLM Step3 测试套件
       Attention 注意力测试
========================================

=== 测试: Attention 创建 ===
[PASS] 创建 Attention 成功
[PASS] hidden_dim 正确
...

=== 测试: 因果掩码 ===
[PASS] mask[0,0] = 0
[PASS] mask[0,1] = -inf
...

=== 测试: 注意力权重 ===
[PASS] 所有注意力权重行和为 1
[PASS] 因果掩码生效 (上三角为 0)
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 注意力层创建和释放正确
- [ ] hidden_dim 必须能被 num_heads 整除
- [ ] 因果掩码正确（下三角为 0，上三角为 -inf）
- [ ] 注意力权重每行和为 1
- [ ] 因果掩码使上三角权重为 0
- [ ] 支持不同序列长度
- [ ] 无内存泄漏

## 下一步

完成 Step 3 后，继续 Step 4 实现完整的 Transformer Block。
