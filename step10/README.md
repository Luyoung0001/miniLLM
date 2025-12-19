# miniLLM Step10 - KV Cache

在 Step9 HTTP Server 的基础上，实现 KV Cache 推理优化，大幅提升文本生成速度。

## 本步骤新增内容

### KV Cache 原理

在 Transformer 推理时，每生成一个新 token 都需要对整个序列重新计算 Attention。KV Cache 通过缓存已计算的 Key 和 Value 避免重复计算：

**无 KV Cache**: 每次生成都重新计算所有位置的 K/V
```
Step 1: 计算 K1, V1
Step 2: 计算 K1, K2, V1, V2
Step 3: 计算 K1, K2, K3, V1, V2, V3
...
时间复杂度: O(N²)
```

**有 KV Cache**: 只计算新位置，复用之前的 K/V
```
Step 1: 计算并缓存 K1, V1
Step 2: 计算 K2, V2，复用缓存的 K1, V1
Step 3: 计算 K3, V3，复用缓存的 K1-2, V1-2
...
时间复杂度: O(N)
```

## 快速开始

### 1. 编译

```bash
make clean && make
```

### 2. 运行测试

```bash
make test
```

### 3. 启动 HTTP 服务

```bash
make serve
# 或指定端口
make serve-port PORT=8080
```

## 文件说明

### 新增源文件
| 文件 | 说明 |
|------|------|
| `src/kv_cache.c` | KV Cache 实现 |
| `src/kv_cache.h` | KV Cache 头文件 |

### 继承的核心文件
| 文件 | 说明 |
|------|------|
| `src/model.c/h` | GPT 模型 |
| `src/attention.c/h` | 多头注意力（已集成 KV Cache） |
| `src/transformer.c/h` | Transformer 层 |
| `src/generate.c/h` | 文本生成（使用 KV Cache） |

## KV Cache API

```c
#include "src/kv_cache.h"
#include "src/model.h"

// 创建 KV Cache
GPTModel* model = model_load("model.bin");
GPTCache* cache = model_cache_create(model, max_seq_len);

// 使用 cache 进行推理
Tensor* logits = tensor_create_2d(seq_len, vocab_size);
model_forward(model, input_ids, seq_len, cache, logits);

// 释放
model_cache_free(cache);
tensor_free(logits);
```

## 模型配置

默认配置 (约 234K 参数):
```c
vocab_size:   256
hidden_dim:   64
num_heads:    4
num_layers:   4
ffn_dim:      256
max_seq_len:  128
```

## 性能提升

| 序列长度 | 无 Cache | 有 Cache | 加速比 |
|----------|----------|----------|--------|
| 32 tokens | 32 次完整计算 | 32 次增量计算 | ~16x |
| 64 tokens | 64 次完整计算 | 64 次增量计算 | ~32x |
| 128 tokens | 128 次完整计算 | 128 次增量计算 | ~64x |

## 下一步

Step11 实现 BPE 分词器，提升训练效率和生成质量。

## 许可

MIT License - 仅供学习和教学使用。
