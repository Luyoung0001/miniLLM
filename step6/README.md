# Step 6: Training - 训练功能实现

本步骤实现完整的训练流程，包括损失计算、反向传播和优化器。

## 目标

1. 实现交叉熵损失函数
2. 实现反向传播
3. 实现 Adam 优化器
4. 实现数据加载器
5. 实现训练循环

## 知识铺垫

### 交叉熵损失

语言模型使用交叉熵损失：

```
L = -sum(log(softmax(logits)[target]))

对于 softmax + cross entropy 的梯度有简洁形式:
grad = softmax(logits) - one_hot(target)
```

### 反向传播

反向传播通过链式法则计算每个参数的梯度：

```
dL/dW = dL/dy * dy/dW

对于线性层 y = x @ W:
  dL/dW = x^T @ dL/dy
  dL/dx = dL/dy @ W^T

对于 LayerNorm:
  需要保存前向传播的均值和方差

对于 GELU:
  GELU'(x) = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * ...
```

### Adam 优化器

Adam 结合了动量和自适应学习率：

```
m = β1 * m + (1 - β1) * g          # 一阶矩 (动量)
v = β2 * v + (1 - β2) * g²         # 二阶矩 (自适应)
m_hat = m / (1 - β1^t)             # 偏差校正
v_hat = v / (1 - β2^t)
θ = θ - lr * m_hat / (√v_hat + ε)  # 参数更新

默认参数:
- β1 = 0.9
- β2 = 0.999
- ε = 1e-8
```

### 学习率调度

常用的学习率调度策略：

```
1. 线性预热 (Warmup):
   lr = lr_max * step / warmup_steps  (step < warmup_steps)

2. 余弦退火 (Cosine Annealing):
   lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

## 文件结构

```
step6/
├── Makefile
├── README.md
├── src/
│   ├── ... (继承 step5)
│   ├── loss.h / loss.c           # 损失函数
│   ├── backward.h / backward.c   # 反向传播
│   ├── optimizer.h / optimizer.c # 优化器
│   ├── dataloader.h / dataloader.c # 数据加载
│   ├── train.h / train.c         # 训练循环
│   └── main.c
└── data/
    └── train.txt                 # 训练数据
```

## 核心 API

### Loss

```c
// 计算交叉熵损失
float cross_entropy_loss(
    Tensor* logits,     // [seq_len, vocab_size]
    int* targets,       // [seq_len]
    int seq_len,
    int vocab_size
);

// 计算损失和梯度
float cross_entropy_loss_with_grad(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size,
    Tensor* grad_logits  // 输出梯度
);
```

### Gradients

```c
// 梯度结构 - 存储所有参数的梯度
typedef struct {
    Tensor* d_token_embedding;
    Tensor* d_position_embedding;
    // ... 所有层的梯度
    Tensor* d_lm_head;
} Gradients;

Gradients* gradients_create(GPTModel* model);
void gradients_free(Gradients* grads);
void gradients_zero(Gradients* grads);
float gradients_clip(Gradients* grads, float max_norm);
```

### Optimizer

```c
// Adam 优化器
typedef struct {
    float lr, beta1, beta2, eps, weight_decay;
    Gradients* m;  // 一阶矩
    Gradients* v;  // 二阶矩
    int t;         // 时间步
} AdamOptimizer;

AdamOptimizer* adam_create(GPTModel* model, float lr);
void adam_step(AdamOptimizer* opt, GPTModel* model, Gradients* grads);

// 学习率调度
float cosine_lr(float lr_max, float lr_min, int step, int total_steps);
float warmup_lr(float lr_max, int step, int warmup_steps);
```

### DataLoader

```c
typedef struct {
    int* tokens;        // 编码后的数据
    int num_tokens;
    int seq_len;
    int batch_size;
    // ...
} DataLoader;

DataLoader* dataloader_create(const char* filepath, Tokenizer* tok, int seq_len, int batch_size);
int dataloader_next_batch(DataLoader* dl, int* input_ids, int* targets);
void dataloader_reset(DataLoader* dl);
void dataloader_shuffle(DataLoader* dl);
```

### Train

```c
typedef struct {
    int num_epochs;
    float learning_rate;
    int batch_size;
    int seq_len;
    float grad_clip;
    // ...
} TrainConfig;

float train(GPTModel* model, Tokenizer* tokenizer, TrainConfig* config);
float train_step(GPTModel* model, int* input_ids, int* targets, ...);
float evaluate(GPTModel* model, DataLoader* dataloader);
```

## 使用示例

```c
#include "train.h"

int main() {
    // 创建模型
    ModelConfig config = default_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    // 创建 tokenizer
    Tokenizer* tok = tokenizer_create();

    // 训练配置
    TrainConfig train_cfg = default_train_config();
    train_cfg.num_epochs = 10;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.batch_size = 4;
    train_cfg.seq_len = 32;
    train_cfg.data_path = "data/train.txt";
    train_cfg.save_path = "model.bin";

    // 开始训练
    float final_loss = train(model, tok, &train_cfg);
    printf("Training complete! Final loss: %.4f\n", final_loss);

    model_free(model);
    tokenizer_free(tok);
    return 0;
}
```

## 训练数据格式

简单的文本文件，每行是一段文本：

```
Hello world! This is a simple example.
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating.
```

## 训练流程

```
for epoch in range(num_epochs):
    dataloader_reset(dl)
    dataloader_shuffle(dl)

    while dataloader_next_batch(dl, input_ids, targets):
        // 前向传播
        model_forward_with_cache(model, input_ids, seq_len, cache, logits)

        // 计算损失和梯度
        loss = cross_entropy_loss_with_grad(logits, targets, grad_logits)

        // 反向传播
        gradients_zero(grads)
        model_backward(model, input_ids, seq_len, grad_logits, cache, grads)

        // 梯度裁剪
        gradients_clip(grads, max_norm)

        // 参数更新
        adam_step(optimizer, model, grads)

        // 学习率调度
        optimizer->lr = warmup_cosine_lr(...)
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
      miniLLM Step6 测试套件
        训练功能测试
========================================

=== 测试: Cross Entropy Loss ===
[PASS] 损失值 > 0
[PASS] 两种方式损失相同
...

=== 测试: Gradients 结构 ===
[PASS] 创建梯度成功
[PASS] 裁剪后范数 <= 1
...

=== 测试: Adam Optimizer ===
[PASS] 创建 Adam 优化器成功
[PASS] 参数更新后改变
...

=== 测试: 训练损失下降 ===
  Initial loss: 5.5xxx
  Final loss: 4.xxxx
  Reduction: xx.xx%
[PASS] 训练后损失下降

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 交叉熵损失计算正确
- [ ] 梯度形状和数值正确
- [ ] Adam 优化器正确更新参数
- [ ] 学习率调度工作正常
- [ ] 数据加载器正确生成批次
- [ ] 训练过程中损失下降
- [ ] 无内存泄漏

## 下一步

完成 Step 6 后，继续 Step 7 实现文本生成功能（采样策略：贪婪、Top-K、Top-P）。
