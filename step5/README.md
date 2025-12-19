# Step 5: Complete GPT Model - 完整的 GPT 模型

本步骤将之前实现的所有组件组装成完整的 GPT 模型。

## 目标

1. 实现模型配置管理
2. 组装完整的 GPT 模型
3. 实现模型的保存和加载

## 知识铺垫

### GPT 模型结构

GPT (Generative Pre-trained Transformer) 是一个解码器-only 的 Transformer 模型：

```
Input Token IDs
      │
      ▼
┌─────────────────────────────┐
│  Token Embedding            │  [vocab_size, hidden_dim]
│  + Position Embedding       │  [max_seq_len, hidden_dim]
└─────────────┬───────────────┘
              │
              ▼
     ┌────────────────┐
     │ Transformer    │ ◄─── N 层
     │    Block       │
     └────────┬───────┘
              │
              ▼
┌─────────────────────────────┐
│    Final LayerNorm          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    LM Head                  │  [hidden_dim, vocab_size]
│  (Output Projection)        │
└─────────────┬───────────────┘
              │
              ▼
         Output Logits
      [seq_len, vocab_size]
```

### 模型配置

```c
typedef struct {
    int vocab_size;     // 词汇表大小 (默认 260)
    int hidden_dim;     // 隐藏层维度 (默认 64)
    int num_heads;      // 注意力头数 (默认 4)
    int num_layers;     // Transformer 层数 (默认 4)
    int ffn_dim;        // FFN 中间层维度 (默认 256)
    int max_seq_len;    // 最大序列长度 (默认 128)
} ModelConfig;
```

### 前向传播流程

```
1. Embedding: input_ids → hidden [seq_len, hidden_dim]
   hidden = token_embed[input_ids] + pos_embed[0:seq_len]

2. Transformer Layers (×N):
   for each layer:
       hidden = transformer_block(hidden, mask)

3. Final LayerNorm:
   hidden = layernorm(hidden)

4. LM Head:
   logits = hidden @ lm_head  → [seq_len, vocab_size]
```

### 参数数量计算

```
使用默认配置 (vocab_size=260, hidden_dim=64, num_heads=4, num_layers=4, ffn_dim=256, max_seq_len=128):

Embedding:
  token_embedding: 260 × 64 = 16,640
  position_embedding: 128 × 64 = 8,192
  小计: 24,832

Transformer Layers (×4):
  每层: 49,728 (来自 step4)
  总计: 198,912

Final LayerNorm:
  gamma: 64
  beta: 64
  小计: 128

LM Head:
  64 × 260 = 16,640

总参数: 24,832 + 198,912 + 128 + 16,640 = 240,512 参数
内存占用: 240,512 × 4 bytes ≈ 939 KB
```

## 文件结构

```
step5/
├── Makefile
├── README.md
├── src/
│   ├── tensor.h / tensor.c
│   ├── math_ops.h / math_ops.c
│   ├── tokenizer.h / tokenizer.c
│   ├── embedding.h / embedding.c
│   ├── attention.h / attention.c
│   ├── layernorm.h / layernorm.c
│   ├── ffn.h / ffn.c
│   ├── transformer.h / transformer.c
│   ├── config.h           # 新增: 模型配置
│   ├── model.h / model.c  # 新增: 完整模型
│   └── main.c
```

## 核心 API

### ModelConfig

```c
// 预定义配置
ModelConfig default_config(void);  // ~240K 参数
ModelConfig tiny_config(void);     // 用于测试
ModelConfig medium_config(void);   // ~1M 参数

// 配置操作
void config_print(ModelConfig* config);
int config_validate(ModelConfig* config);
int config_save(ModelConfig* config, const char* path);
int config_load(ModelConfig* config, const char* path);
```

### GPTModel

```c
// 创建和销毁
GPTModel* model_create(ModelConfig config);
void model_free(GPTModel* model);

// 初始化
void model_init_random(GPTModel* model, float std);

// 前向传播
void model_forward(
    GPTModel* model,
    int* input_ids,
    int seq_len,
    GPTCache* cache,
    Tensor* logits
);

// 获取最后一个位置的 logits (用于生成)
void model_get_last_logits(Tensor* logits, int seq_len, Tensor* last_logits);

// 保存和加载
int model_save(GPTModel* model, const char* path);
GPTModel* model_load(const char* path);

// 工具函数
int model_num_params(GPTModel* model);
void model_print_info(GPTModel* model);
size_t model_memory_size(GPTModel* model);
```

### GPTCache

```c
GPTCache* model_cache_create(GPTModel* model, int seq_len);
void model_cache_free(GPTCache* cache);
int model_cache_resize(GPTCache* cache, GPTModel* model, int new_seq_len);
```

## 使用示例

```c
#include "model.h"
#include "tokenizer.h"

int main() {
    // 创建 tokenizer 和模型
    Tokenizer* tok = tokenizer_create();
    ModelConfig config = default_config();
    config.vocab_size = tok->vocab_size;

    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    // 打印模型信息
    model_print_info(model);

    // 编码输入
    const char* text = "Hello";
    int len;
    int* ids = tokenizer_encode(tok, text, &len, 0, 0);

    // 前向传播
    int logits_shape[] = {len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    model_forward(model, ids, len, NULL, logits);

    // 获取下一个 token 的预测
    int last_shape[] = {config.vocab_size};
    Tensor* last_logits = tensor_zeros(1, last_shape);
    model_get_last_logits(logits, len, last_logits);

    int predicted_id = tensor_argmax(last_logits);
    printf("Predicted next token: %d ('%s')\n",
           predicted_id, tokenizer_decode_id(tok, predicted_id));

    // 保存模型
    model_save(model, "model.bin");

    // 清理
    free(ids);
    tensor_free(logits);
    tensor_free(last_logits);
    model_free(model);
    tokenizer_free(tok);

    return 0;
}
```

## 模型文件格式

```
┌─────────────────────────────────────┐
│ Magic Number (4 bytes): "MLLM"      │
│ Version (4 bytes): 1                │
├─────────────────────────────────────┤
│ ModelConfig (sizeof(ModelConfig))   │
├─────────────────────────────────────┤
│ Token Embedding                     │
│ Position Embedding                  │
├─────────────────────────────────────┤
│ Transformer Layer 0:                │
│   - LayerNorm1 (gamma, beta)        │
│   - Attention (W_q, W_k, W_v, W_o)  │
│   - LayerNorm2 (gamma, beta)        │
│   - FFN (W1, b1, W2, b2)            │
├─────────────────────────────────────┤
│ ... (more layers)                   │
├─────────────────────────────────────┤
│ Final LayerNorm (gamma, beta)       │
├─────────────────────────────────────┤
│ LM Head                             │
└─────────────────────────────────────┘
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
      miniLLM Step5 测试套件
       完整 GPT 模型测试
========================================

=== 测试: Config 配置 ===
[PASS] 默认 vocab_size = 260
[PASS] 默认 hidden_dim = 64
...

=== 测试: Model 创建 ===
[PASS] 创建模型成功
[PASS] embedding 已创建
...

=== 测试: Model 前向传播 ===
[PASS] logits shape[0] 正确
[PASS] logits 不全为零
[PASS] softmax 后概率和为 1
...

=== 测试: Model 保存和加载 ===
[PASS] 保存模型成功
[PASS] 加载模型成功
[PASS] 保存/加载后输出相同
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 模型配置正确验证和保存/加载
- [ ] 前向传播输出正确的 shape
- [ ] softmax 后概率分布和为 1
- [ ] 模型保存/加载后输出一致
- [ ] 与 Tokenizer 正确集成
- [ ] 支持不同配置 (tiny, default, medium)
- [ ] 无内存泄漏

## 下一步

完成 Step 5 后，继续 Step 6 实现训练功能（损失计算、反向传播、优化器）。
