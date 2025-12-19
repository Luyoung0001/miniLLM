#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "model.h"
#include "backward.h"

/**
 * Adam 优化器
 *
 * 参数更新规则:
 * m = β1 * m + (1 - β1) * g          (一阶矩估计)
 * v = β2 * v + (1 - β2) * g²         (二阶矩估计)
 * m_hat = m / (1 - β1^t)             (偏差校正)
 * v_hat = v / (1 - β2^t)
 * θ = θ - lr * m_hat / (√v_hat + ε)
 *
 * 带权重衰减 (AdamW):
 * θ = θ - lr * weight_decay * θ
 */

typedef struct {
    float lr;           // 学习率
    float beta1;        // 一阶矩衰减率 (默认 0.9)
    float beta2;        // 二阶矩衰减率 (默认 0.999)
    float eps;          // 防止除零 (默认 1e-8)
    float weight_decay; // 权重衰减 (默认 0.01)

    // Adam 状态 (一阶和二阶矩)
    Gradients* m;       // 一阶矩
    Gradients* v;       // 二阶矩

    int t;              // 时间步 (用于偏差校正)
} AdamOptimizer;

/**
 * SGD 优化器 (带动量)
 */
typedef struct {
    float lr;           // 学习率
    float momentum;     // 动量 (默认 0.9)
    float weight_decay; // 权重衰减

    Gradients* velocity; // 速度 (动量)
} SGDOptimizer;

// ============ Adam 优化器 ============

/**
 * 创建 Adam 优化器
 * @param model 模型 (用于获取参数结构)
 * @param lr 学习率
 * @return 优化器实例
 */
AdamOptimizer* adam_create(GPTModel* model, float lr);

/**
 * 创建带完整参数的 Adam 优化器
 */
AdamOptimizer* adam_create_full(
    GPTModel* model,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

/**
 * 释放 Adam 优化器
 */
void adam_free(AdamOptimizer* opt);

/**
 * Adam 优化步骤
 * @param opt 优化器
 * @param model 模型
 * @param grads 梯度
 */
void adam_step(AdamOptimizer* opt, GPTModel* model, Gradients* grads);

/**
 * 重置优化器状态
 */
void adam_reset(AdamOptimizer* opt);

// ============ SGD 优化器 ============

/**
 * 创建 SGD 优化器
 */
SGDOptimizer* sgd_create(GPTModel* model, float lr, float momentum);

/**
 * 释放 SGD 优化器
 */
void sgd_free(SGDOptimizer* opt);

/**
 * SGD 优化步骤
 */
void sgd_step(SGDOptimizer* opt, GPTModel* model, Gradients* grads);

// ============ 学习率调度器 ============

/**
 * 余弦退火学习率
 * lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
 */
float cosine_lr(float lr_max, float lr_min, int step, int total_steps);

/**
 * 线性预热学习率
 * lr = lr_max * min(1, step / warmup_steps)
 */
float warmup_lr(float lr_max, int step, int warmup_steps);

/**
 * 带预热的余弦退火
 */
float warmup_cosine_lr(
    float lr_max,
    float lr_min,
    int step,
    int warmup_steps,
    int total_steps
);

#endif // OPTIMIZER_H
