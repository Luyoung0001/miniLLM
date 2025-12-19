# miniLLM - 从零实现迷你语言模型

一个纯 C 语言实现的教学用语言模型项目，从零开始逐步构建完整的 GPT 架构。

## 项目简介

miniLLM 是一个教学项目，旨在帮助理解大语言模型 (LLM) 的核心原理。通过 12 个递进的步骤，从最基础的张量运算开始，逐步实现完整的 Transformer 架构。

**特点**：
- 纯 C 语言实现，无外部依赖
- 每个步骤独立可编译运行
- 包含完整的前向传播和反向传播
- 支持训练、推理、对话和 HTTP API

## 项目结构

```
miniLLM/
├── step0/   - 张量基础
├── step1/   - 分词器
├── step2/   - 词嵌入
├── step3/   - 注意力机制
├── step4/   - Transformer 层
├── step5/   - GPT 模型
├── step6/   - 训练循环
├── step7/   - 文本生成
├── step8/   - 对话系统
├── step9/   - HTTP 服务器
├── step10/  - KV Cache 优化
└── step11/  - BPE 分词器
```

## 各步骤详解

### Step 0 - 张量基础
实现基础的张量数据结构和数学运算。

```c
Tensor* t = tensor_create_2d(rows, cols);
tensor_matmul(A, B, C);  // C = A @ B
tensor_add(A, B, C);     // C = A + B
```

**核心文件**: `tensor.c`, `math_ops.c`

---

### Step 1 - 分词器 (Tokenizer)
实现字节级分词器，将文本转换为 token ID 序列。

```c
Tokenizer* tok = tokenizer_create();
int* ids = tokenizer_encode(tok, "hello world", &len);
char* text = tokenizer_decode(tok, ids, len);
```

**词汇表大小**: 256 (ASCII 字符)

---

### Step 2 - 词嵌入 (Embedding)
实现 Token Embedding 和 Position Embedding。

```c
Embedding* emb = embedding_create(vocab_size, hidden_dim, max_seq_len);
embedding_forward(emb, token_ids, seq_len, output);
```

**输出**: `[seq_len, hidden_dim]` 的嵌入向量

---

### Step 3 - 注意力机制 (Attention)
实现 Scaled Dot-Product Attention 和 Multi-Head Attention。

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

```c
MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
attention_forward(attn, input, seq_len, output);
```

---

### Step 4 - Transformer 层
组合 Multi-Head Attention、Feed-Forward Network 和 Layer Normalization。

```
TransformerBlock(x) = x + Attention(LayerNorm(x))
                    = x + FFN(LayerNorm(x))
```

```c
TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
transformer_block_forward(block, input, seq_len, output);
```

---

### Step 5 - GPT 模型
组装完整的 GPT 模型架构。

```
GPT(tokens) = LM_Head(TransformerBlocks(Embedding(tokens)))
```

```c
GPTModel* model = model_create(config);
model_forward(model, input_ids, seq_len, cache, logits);
model_save(model, "model.bin");
model_load("model.bin");
```

---

### Step 6 - 训练循环
实现反向传播、梯度计算和 Adam 优化器。

```c
// 前向传播
model_forward_with_cache(model, input_ids, seq_len, cache, logits);

// 计算损失和梯度
float loss = cross_entropy_loss_with_grad(logits, targets, seq_len, vocab_size, grad_logits);

// 反向传播
model_backward(model, input_ids, seq_len, grad_logits, cache, grads);

// 参数更新
adam_step(optimizer, model, grads);
```

---

### Step 7 - 文本生成
实现自回归文本生成，支持多种采样策略。

```c
GenerateConfig gen_config = {
    .max_new_tokens = 50,
    .temperature = 0.8,
    .top_k = 40,
    .top_p = 0.9
};
char* output = generate(model, tokenizer, "Hello", &gen_config);
```

**采样方法**:
- 贪婪解码 (temperature=0)
- 温度采样
- Top-K 采样
- Top-P (Nucleus) 采样

---

### Step 8 - 对话系统
实现多轮对话管理和上下文维护。

```c
Conversation* conv = conversation_create();
conversation_set_system(conv, "You are a helpful assistant.");
conversation_add_message(conv, "user", "Hello!");
char* response = chat_generate(model, tokenizer, conv, &config);
```

---

### Step 9 - HTTP 服务器
实现 REST API，支持远程调用。

```bash
./step9 --serve --port 8080
```

**API 端点**:
- `POST /chat` - 对话生成
- `POST /generate` - 文本补全
- `GET /health` - 健康检查

---

### Step 10 - KV Cache 优化
实现 Key-Value Cache，大幅提升推理速度。

**原理**:
```
无 Cache: 每次生成重新计算所有 K/V → O(N²)
有 Cache: 缓存已计算的 K/V，只计算新位置 → O(N)
```

```c
GPTCache* cache = model_cache_create(model, max_seq_len);
model_forward(model, tokens, seq_len, cache, logits);  // 自动利用 cache
```

---

### Step 11 - BPE 分词器
实现 Byte Pair Encoding 分词，提升训练效率。

**原理**:
```
字节级: "hello world" → 11 tokens (每个字符)
BPE:    "hello world" → 3 tokens  ["hello", " ", "world"]
压缩比: ~3-7x
```

```c
BPETokenizer* tok = bpe_tokenizer_create();
bpe_train(tok, text, num_merges);
int* ids = bpe_encode(tok, "hello world", &len);
```

## 快速开始

### 编译运行最终版本

```bash
cd step11
make all

# 训练模型
./train_bpe train_data.txt 200

# 交互对话
./chat_bpe

# 效果展示
./demo_bpe
```

### 训练数据格式

纯文本文件，每行一个句子：

```text
hello how are you
I am fine thank you
good morning
good morning to you
```

## 模型配置

### 默认配置 (~280K 参数)

```c
ModelConfig config = {
    .vocab_size   = 601,    // BPE 词汇表大小
    .hidden_dim   = 64,     // 隐藏层维度
    .num_heads    = 4,      // 注意力头数
    .num_layers   = 4,      // Transformer 层数
    .ffn_dim      = 256,    // FFN 中间层维度
    .max_seq_len  = 128     // 最大序列长度
};
```

### 大配置 (~1M 参数)

```c
ModelConfig config = {
    .vocab_size   = 601,
    .hidden_dim   = 128,
    .num_heads    = 4,
    .num_layers   = 4,
    .ffn_dim      = 512,
    .max_seq_len  = 256
};
```

## 架构图

```
输入文本: "hello world"
    │
    ▼
┌─────────────────┐
│   BPE Tokenizer │  → [token_ids]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token Embedding │  → [seq_len, hidden_dim]
│  + Position Emb │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Transformer x N │
│ ┌─────────────┐ │
│ │ LayerNorm   │ │
│ │ Multi-Head  │ │
│ │ Attention   │ │──→ KV Cache
│ │ + Residual  │ │
│ ├─────────────┤ │
│ │ LayerNorm   │ │
│ │ FFN         │ │
│ │ + Residual  │ │
│ └─────────────┘ │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Final LayerNorm│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    LM Head      │  → [seq_len, vocab_size]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Softmax/Sample  │  → next_token
└─────────────────┘
```

## 学习收获

通过这个项目，你将理解：

1. **张量运算** - 矩阵乘法、广播、内存布局
2. **分词原理** - 字节级分词 vs BPE 子词分词
3. **嵌入层** - 如何将离散 token 映射到连续向量空间
4. **注意力机制** - Q/K/V 的含义，为什么需要缩放
5. **Transformer** - 残差连接、层归一化的作用
6. **训练循环** - 交叉熵损失、反向传播、Adam 优化
7. **生成策略** - 温度、Top-K、Top-P 采样的区别
8. **推理优化** - KV Cache 如何将 O(N²) 降到 O(N)

## 局限性

这是一个教学项目，存在以下局限：

| 方面 | miniLLM | 真实 LLM (GPT-2) |
|------|---------|------------------|
| 参数量 | ~1M | 117M - 1.5B |
| 训练数据 | KB 级 | GB - TB 级 |
| 训练时间 | 分钟 | 天 - 周 |
| 硬件 | CPU | GPU 集群 |
| 生成质量 | 简单模式 | 流畅对话 |

## 扩展方向

如果想进一步学习，可以尝试：

1. **添加 CUDA 支持** - 实现 GPU 加速
2. **实现更多架构** - Llama、Mistral 等
3. **量化推理** - INT8/INT4 量化
4. **模型并行** - 张量并行、流水线并行
5. **RLHF** - 人类反馈强化学习

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
- [llm.c](https://github.com/karpathy/llm.c) - Andrej Karpathy 的 LLM 实现

## 许可证

MIT License - 仅供学习和教学使用。

---

*Build your own LLM, understand the magic behind it.*
