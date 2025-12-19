# miniLLM Step11 - BPE Tokenizer

在 Step10 KV Cache 的基础上，实现 BPE (Byte Pair Encoding) 分词器，大幅提升训练效率和生成质量。

## 本步骤新增内容

### BPE 分词器
相比字节级分词 (1字符=1token)，BPE 通过合并高频字符对构建子词词汇表：

```
字节级: "hello world" -> 11 tokens (每个字符一个)
BPE:    "hello world" -> 3 tokens  ["hello", " ", "world"]
压缩比: ~3.7x
```

**优势**:
- 更高效的词汇利用率
- 模型更容易学习词级模式
- 支持未知词的子词分解

## 快速开始

### 1. 编译所有程序

```bash
make all
```

### 2. 首次训练

```bash
# 使用默认数据训练
make train

# 或指定数据和轮数
./train_bpe train_better.txt 200
```

训练会：
1. 从训练数据构建 BPE 词汇表
2. 创建并训练 GPT 模型
3. 保存 `model_bpe.bin` 和 `bpe_vocab.txt`

### 3. 继续训练

```bash
# 在现有模型上继续训练
./continue_train train_better.txt 100 0.0005
```

参数: `<数据文件> [epochs] [learning_rate]`

### 4. 交互对话

```bash
make chat
# 或
./chat_bpe
```

**对话命令**:
| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/temp <值>` | 设置温度 (0-2, 默认0.8) |
| `/len <值>` | 设置最大生成长度 |
| `/topk <值>` | 设置 Top-K 采样 |
| `/quit` | 退出 |

### 5. 查看生成效果

```bash
make demo
```

## 文件说明

### 源代码
| 文件 | 说明 |
|------|------|
| `src/bpe_tokenizer.c/h` | BPE 分词器实现 |
| `src/model.c/h` | GPT 模型 (继承自 step10) |
| `src/kv_cache.c/h` | KV Cache (继承自 step10) |

### 程序
| 文件 | 说明 |
|------|------|
| `train_bpe.c` | 首次训练脚本 |
| `continue_train.c` | 继续训练脚本 |
| `chat_bpe.c` | 交互对话程序 |
| `demo_bpe.c` | 效果展示程序 |
| `test_bpe.c` | BPE 分词器测试 |

### 生成文件
| 文件 | 说明 |
|------|------|
| `model_bpe.bin` | 训练后的模型权重 |
| `bpe_vocab.txt` | BPE 词汇表 |

## 训练数据格式

纯文本文件，每行一个句子：

```text
hello how are you
I am fine thank you
the cat sat on the mat
good morning friend
```

## 模型配置

默认配置 (约 234K 参数):
```c
vocab_size:   动态 (基于 BPE 训练)
hidden_dim:   64
num_heads:    4
num_layers:   4
ffn_dim:      256
max_seq_len:  80
```

## BPE API

```c
#include "src/bpe_tokenizer.h"

// 创建分词器
BPETokenizer* tok = bpe_tokenizer_create();

// 训练 BPE (从文本构建词汇表)
bpe_train(tok, "训练文本...", 500);  // 500次合并

// 保存/加载词汇表
bpe_save_vocab(tok, "vocab.txt");
bpe_load_vocab(tok, "vocab.txt");

// 编码文本 -> token IDs
int len;
int* ids = bpe_encode(tok, "hello world", &len, 0, 0);

// 解码 token IDs -> 文本
char* text = bpe_decode(tok, ids, len);

// 释放
free(ids);
free(text);
bpe_tokenizer_free(tok);
```

## 训练建议

| 数据量 | 推荐 epochs | 预期效果 |
|--------|-------------|----------|
| 1-5 KB | 200-500 | 学习简单模式 |
| 10-50 KB | 100-200 | 基本连贯性 |
| 100+ KB | 50-100 | 较好的生成质量 |

**调参建议**:
- `learning_rate`: 首次训练 0.001，继续训练 0.0005
- `temperature`: 0.7-0.9 比较平衡
- `top_k`: 20-50 防止生成低概率词

## 与 Step10 的区别

| 特性 | Step10 | Step11 |
|------|--------|--------|
| 分词器 | 字节级 | BPE |
| 压缩率 | 1x | 3-7x |
| 训练效率 | 低 | 高 |
| 生成质量 | 基础 | 较好 |
| 词汇表 | 固定 256 | 动态构建 |

## 运行测试

```bash
make test
```

## 许可

MIT License - 仅供学习和教学使用。
