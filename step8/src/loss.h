#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

/**
 * 交叉熵损失函数
 *
 * 用于语言模型训练:
 * L = -sum(log(softmax(logits)[target]))
 *
 * 对于 softmax 后的梯度:
 * grad = softmax(logits) - one_hot(target)
 */

/**
 * 计算交叉熵损失
 * @param logits 模型输出 [seq_len, vocab_size]
 * @param targets 目标 token IDs [seq_len]
 * @param seq_len 序列长度
 * @param vocab_size 词汇表大小
 * @return 平均损失值
 */
float cross_entropy_loss(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size
);

/**
 * 计算交叉熵损失并返回梯度
 * @param logits 模型输出 [seq_len, vocab_size]
 * @param targets 目标 token IDs [seq_len]
 * @param seq_len 序列长度
 * @param vocab_size 词汇表大小
 * @param grad_logits 输出的 logits 梯度 [seq_len, vocab_size]
 * @return 平均损失值
 */
float cross_entropy_loss_with_grad(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size,
    Tensor* grad_logits
);

/**
 * 只计算损失的梯度 (不返回损失值，用于批量计算)
 */
void cross_entropy_grad(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size,
    Tensor* grad_logits
);

/**
 * 对单个位置计算 softmax 和交叉熵梯度
 * 梯度 = softmax(logits) - one_hot(target)
 */
void softmax_cross_entropy_grad(
    float* logits,      // [vocab_size]
    int target,         // 目标 token ID
    int vocab_size,
    float* grad         // 输出梯度 [vocab_size]
);

#endif // LOSS_H
