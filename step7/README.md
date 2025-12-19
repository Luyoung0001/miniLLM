# Step 7: Generation - 文本生成实现

本步骤实现完整的文本生成功能，包括多种采样策略。

## 目标

1. 实现贪婪解码 (Greedy Decoding)
2. 实现温度采样 (Temperature Sampling)
3. 实现 Top-K 采样
4. 实现 Top-P (Nucleus) 采样
5. 实现流式生成

## 知识铺垫

### 采样策略

语言模型生成文本时，需要从概率分布中选择下一个 token：

```
1. 贪婪解码 (Greedy):
   next_token = argmax(logits)
   简单但容易产生重复

2. 温度采样 (Temperature):
   scaled_logits = logits / temperature
   probs = softmax(scaled_logits)
   - temperature < 1: 更确定性 (分布更尖)
   - temperature > 1: 更随机 (分布更平)

3. Top-K 采样:
   保留概率最高的 K 个 token
   将其他 token 的概率设为 0
   重新归一化后采样

4. Top-P (Nucleus) 采样:
   按概率降序排列 tokens
   保留累积概率达到 P 的最小集合
   比 Top-K 更动态 (自动调整候选数量)
```

### 自回归生成

```
tokens = encode(prompt)
for i in range(max_new_tokens):
    logits = model_forward(tokens)
    next_token = sample(logits[-1])  # 取最后位置
    if next_token == EOS:
        break
    tokens.append(next_token)
return decode(tokens)
```

## 文件结构

```
step7/
├── Makefile
├── README.md
├── src/
│   ├── ... (继承 step6)
│   ├── generate.h / generate.c   # 文本生成
│   └── main.c
```

## 核心 API

### GenerateConfig

```c
typedef struct {
    float temperature;      // 温度参数 (默认 1.0)
                           // < 1: 更确定性  > 1: 更随机
    int top_k;             // Top-K 采样 (默认 40, 0 表示禁用)
    float top_p;           // Top-P 采样 (默认 0.9, 1.0 表示禁用)
    int max_new_tokens;    // 最大生成 token 数 (默认 100)
    int do_sample;         // 1: 采样, 0: 贪婪解码
    int seed;              // 随机种子 (-1 表示使用当前时间)
} GenerateConfig;

// 预设配置
GenerateConfig default_generate_config(void);  // 标准采样
GenerateConfig greedy_config(void);            // 贪婪解码
GenerateConfig creative_config(void);          // 创意生成 (高温度)
```

### 生成函数

```c
// 简单生成 (返回文本)
char* generate(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
);

// 完整生成 (返回详细结果)
GenerateResult* generate_full(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
);

// 流式生成 (每个 token 调用回调)
typedef int (*GenerateCallback)(int token_id, const char* token_str, void* user_data);

int generate_stream(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config,
    GenerateCallback callback,
    void* user_data
);
```

### 采样函数

```c
// 贪婪采样
int sample_greedy(Tensor* logits);

// 温度采样
int sample_temperature(Tensor* logits, float temperature);

// Top-K 采样
int sample_top_k(Tensor* logits, int k, float temperature);

// Top-P 采样
int sample_top_p(Tensor* logits, float p, float temperature);

// 组合采样 (Top-K + Top-P + Temperature)
int sample_combined(Tensor* logits, GenerateConfig* config);
```

## 使用示例

### 基本生成

```c
#include "generate.h"

int main() {
    // 创建模型
    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    // 创建 tokenizer
    Tokenizer* tok = tokenizer_create();

    // 生成配置
    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 50;
    gen_cfg.temperature = 0.8f;
    gen_cfg.seed = 42;

    // 生成文本
    char* text = generate(model, tok, "Hello", &gen_cfg);
    printf("Generated: %s\n", text);

    free(text);
    tokenizer_free(tok);
    model_free(model);
    return 0;
}
```

### 流式生成

```c
int print_token(int token_id, const char* token_str, void* user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", token_str);
    fflush(stdout);
    return 0;  // 继续生成
}

int main() {
    // ... 创建模型和 tokenizer ...

    GenerateConfig cfg = default_generate_config();
    cfg.max_new_tokens = 100;

    printf("Generating: ");
    generate_stream(model, tok, "Once upon a time", &cfg, print_token, NULL);
    printf("\n");

    // ... 清理 ...
}
```

### 不同采样策略

```c
// 贪婪解码 - 最确定性
GenerateConfig greedy = greedy_config();
char* text1 = generate(model, tok, prompt, &greedy);

// 低温度 - 较确定
GenerateConfig low_temp = default_generate_config();
low_temp.temperature = 0.3f;
char* text2 = generate(model, tok, prompt, &low_temp);

// 高温度 - 更随机
GenerateConfig high_temp = creative_config();
char* text3 = generate(model, tok, prompt, &high_temp);

// Top-K 采样
GenerateConfig topk = default_generate_config();
topk.top_k = 10;  // 只从 top 10 中采样
char* text4 = generate(model, tok, prompt, &topk);

// Top-P 采样
GenerateConfig topp = default_generate_config();
topp.top_p = 0.9f;  // 从累积概率 90% 的集合中采样
char* text5 = generate(model, tok, prompt, &topp);
```

## 采样策略比较

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| Greedy | 最确定，可能重复 | 翻译、摘要 |
| Temperature < 1 | 较确定，减少随机性 | 代码生成 |
| Temperature > 1 | 更随机，更有创意 | 创意写作 |
| Top-K | 限制候选数量 | 通用生成 |
| Top-P | 动态调整候选集 | 对话、故事 |

## 编译和运行

```bash
make        # 编译
make test   # 运行测试
make clean  # 清理
```

## 预期输出

```
========================================
      miniLLM Step7 测试套件
        文本生成测试
========================================

=== 测试: 贪婪采样 ===
[PASS] 贪婪采样选择最大 logit
[PASS] 贪婪采样选择新的最大 logit

=== 测试: 温度采样 ===
  高温度 (T=2.0) 分布: 15 18 25 22 20
  低温度 (T=0.1) 分布: 0 0 100 0 0
[PASS] 高温度分布相对均匀
[PASS] 低温度集中在最大 logit

=== 测试: Top-K 采样 ===
  Top-K (k=3) 分布: 0 0 0 0 0 0 0 28 35 37
[PASS] Top-K 只采样前 K 个

=== 测试: Top-P 采样 ===
  Top-P (p=0.9) 分布: 58 32 8 2 0
[PASS] Top-P 主要采样高概率 tokens

=== 测试: 基本文本生成 ===
[PASS] 生成文本成功
[PASS] 生成文本长度 >= prompt
  Prompt: "Hello"
  Generated: "Hello..."

=== 测试: 贪婪生成 ===
[PASS] 两次生成都成功
[PASS] 贪婪生成是确定性的

=== 测试: 流式生成 ===
[PASS] 流式生成返回 > 0
[PASS] 回调被调用 5 次

=== 测试: 训练后生成 ===
  Trained pattern: "abcabcabc..."
  Generated: "abcabcabc..."
[PASS] 训练后能生成文本

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 贪婪采样正确选择最大概率 token
- [ ] 温度采样正确调整分布
- [ ] Top-K 采样只保留前 K 个
- [ ] Top-P 采样正确计算累积概率
- [ ] 生成函数正确产生文本
- [ ] 流式生成正确调用回调
- [ ] 训练后的模型能生成有意义的文本
- [ ] 无内存泄漏

## 下一步

完成 Step 7 后，继续 Step 8 实现对话系统（多轮对话、上下文管理）。
