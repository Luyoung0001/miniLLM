# Step 2: Embedding - 词嵌入层

本步骤实现词嵌入层，将离散的 token ID 映射到连续的向量空间。

## 目标

1. 实现 Token Embedding (词嵌入)
2. 实现 Position Embedding (位置编码)
3. 支持正弦位置编码和可学习位置编码

## 知识铺垫

### 什么是词嵌入？

词嵌入将离散的 token（整数 ID）映射到连续的向量空间：

```
token_id: 42  →  embedding: [0.1, -0.3, 0.5, ..., 0.2]  (hidden_dim 维)
```

本质上是一个**查表操作**：从 embedding 矩阵中取出对应行。

```
embedding_matrix: [vocab_size, hidden_dim]
token_embedding = embedding_matrix[token_id]
```

### 为什么需要位置编码？

Transformer 本身没有位置信息（与 RNN 不同），需要显式注入位置信息。

**方式 1: 正弦位置编码 (Sinusoidal)**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

特点：
- 固定不可学习
- 可泛化到训练时未见过的长度
- 相对位置信息通过三角函数性质编码

**方式 2: 可学习位置编码 (Learned)**

```
position_embedding: [max_seq_len, hidden_dim]  # 可训练参数
```

特点：
- 可训练
- 无法泛化到超过 max_seq_len 的长度

### Embedding 计算流程

```
输入: token_ids = [t0, t1, t2, ...]
      位置:       [0,  1,  2,  ...]

token_emb = lookup(token_embedding_matrix, token_ids)
pos_emb   = position_embedding[0:seq_len]

output = token_emb + pos_emb   # 逐元素相加
```

### 正弦位置编码可视化

```
位置\维度  0(sin)  1(cos)  2(sin)  3(cos)  ...
   0        0       1       0       1
   1       sin     cos     sin     cos      ...
   2       sin     cos     sin     cos      ...
  ...
```

- 低频维度变化慢，捕获长距离位置
- 高频维度变化快，捕获短距离位置

## 文件结构

```
step2/
├── Makefile
├── README.md
├── src/
│   ├── tensor.h / tensor.c       # (来自 step0)
│   ├── math_ops.h / math_ops.c   # (来自 step0)
│   ├── tokenizer.h / tokenizer.c # (来自 step1)
│   ├── embedding.h               # 嵌入层声明
│   ├── embedding.c               # 嵌入层实现
│   └── main.c                    # 测试程序
```

## 核心 API

### 创建和销毁

```c
// 创建 Embedding 层
Embedding* embedding_create(int vocab_size, int hidden_dim, int max_seq_len);

// 释放
void embedding_free(Embedding* emb);
```

### 初始化

```c
// 随机初始化 token embedding (正态分布)
void embedding_init_random(Embedding* emb, float std);

// 正弦位置编码
void embedding_init_sinusoidal_position(Embedding* emb);

// 可学习位置编码 (随机初始化)
void embedding_init_learned_position(Embedding* emb, float std);
```

### 前向传播

```c
// 完整前向传播: output = token_emb + pos_emb
void embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

// 仅 token embedding
void embedding_get_token(Embedding* emb, int* token_ids, int seq_len, Tensor* output);

// 仅 position embedding
void embedding_get_position(Embedding* emb, int seq_len, Tensor* output);
```

## 使用示例

```c
#include "embedding.h"
#include "tokenizer.h"

int main() {
    // 配置
    int vocab_size = 260;
    int hidden_dim = 64;
    int max_seq_len = 128;

    // 创建 embedding
    Embedding* emb = embedding_create(vocab_size, hidden_dim, max_seq_len);
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    // 输入 token IDs
    int token_ids[] = {76, 109, 112, 112, 115};  // "Hello"
    int seq_len = 5;

    // 创建输出张量
    int shape[] = {seq_len, hidden_dim};
    Tensor* output = tensor_zeros(2, shape);

    // 前向传播
    embedding_forward(emb, token_ids, seq_len, output);
    // output: [5, 64]

    tensor_free(output);
    embedding_free(emb);
    return 0;
}
```

## 参数数量计算

```
vocab_size = 260
hidden_dim = 64
max_seq_len = 128

token_embedding:    260 × 64 = 16,640
position_embedding: 128 × 64 =  8,192
-----------------------------------------
总计:                         24,832 参数
```

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
      miniLLM Step2 测试套件
         Embedding 测试
========================================

=== 测试: Embedding 创建 ===
[PASS] 创建 Embedding 成功
[PASS] vocab_size 正确
...

=== 测试: 正弦位置编码 ===
[PASS] pos=0, dim=0: sin(0) = 0
[PASS] pos=0, dim=1: cos(0) = 1
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] Embedding 创建和释放正确
- [ ] Token embedding 随机初始化，均值接近 0，标准差接近指定值
- [ ] 正弦位置编码值在 [-1, 1] 范围内
- [ ] 相同 token 的嵌入向量相同
- [ ] 加位置编码后，相同 token 在不同位置的嵌入不同
- [ ] 与 Tokenizer 正确集成
- [ ] 无内存泄漏

## 下一步

完成 Step 2 后，继续 Step 3 实现注意力机制 (Attention)。
