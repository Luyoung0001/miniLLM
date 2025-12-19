#include "loss.h"
#include <math.h>
#include <float.h>

void softmax_cross_entropy_grad(
    float* logits,
    int target,
    int vocab_size,
    float* grad
) {
    // 1. 找到最大值 (数值稳定性)
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // 2. 计算 exp(logits - max) 和 sum
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        grad[i] = expf(logits[i] - max_val);
        sum += grad[i];
    }

    // 3. softmax = exp(x) / sum
    // 4. grad = softmax - one_hot(target)
    for (int i = 0; i < vocab_size; i++) {
        grad[i] = grad[i] / sum;
        if (i == target) {
            grad[i] -= 1.0f;
        }
    }
}

float cross_entropy_loss(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size
) {
    if (logits == NULL || targets == NULL) return 0.0f;

    float total_loss = 0.0f;

    for (int s = 0; s < seq_len; s++) {
        float* pos_logits = &logits->data[s * vocab_size];
        int target = targets[s];

        // 找最大值
        float max_val = pos_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (pos_logits[i] > max_val) {
                max_val = pos_logits[i];
            }
        }

        // 计算 log_sum_exp
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(pos_logits[i] - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);

        // 交叉熵: -log(softmax[target]) = -logits[target] + log_sum_exp
        float loss = -pos_logits[target] + log_sum_exp;
        total_loss += loss;
    }

    return total_loss / seq_len;
}

void cross_entropy_grad(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size,
    Tensor* grad_logits
) {
    if (logits == NULL || targets == NULL || grad_logits == NULL) return;

    for (int s = 0; s < seq_len; s++) {
        float* pos_logits = &logits->data[s * vocab_size];
        float* pos_grad = &grad_logits->data[s * vocab_size];
        int target = targets[s];

        softmax_cross_entropy_grad(pos_logits, target, vocab_size, pos_grad);

        // 除以 seq_len 得到平均梯度
        for (int i = 0; i < vocab_size; i++) {
            pos_grad[i] /= seq_len;
        }
    }
}

float cross_entropy_loss_with_grad(
    Tensor* logits,
    int* targets,
    int seq_len,
    int vocab_size,
    Tensor* grad_logits
) {
    if (logits == NULL || targets == NULL) return 0.0f;

    float total_loss = 0.0f;

    for (int s = 0; s < seq_len; s++) {
        float* pos_logits = &logits->data[s * vocab_size];
        float* pos_grad = (grad_logits != NULL) ? &grad_logits->data[s * vocab_size] : NULL;
        int target = targets[s];

        // 找最大值
        float max_val = pos_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (pos_logits[i] > max_val) {
                max_val = pos_logits[i];
            }
        }

        // 计算 exp 和 sum
        float sum_exp = 0.0f;
        float exp_vals[vocab_size];  // VLA
        for (int i = 0; i < vocab_size; i++) {
            exp_vals[i] = expf(pos_logits[i] - max_val);
            sum_exp += exp_vals[i];
        }

        // 损失: -log(softmax[target])
        float log_sum_exp = max_val + logf(sum_exp);
        float loss = -pos_logits[target] + log_sum_exp;
        total_loss += loss;

        // 梯度: softmax - one_hot(target)
        if (pos_grad != NULL) {
            for (int i = 0; i < vocab_size; i++) {
                pos_grad[i] = exp_vals[i] / sum_exp;
                if (i == target) {
                    pos_grad[i] -= 1.0f;
                }
                // 除以 seq_len 得到平均梯度
                pos_grad[i] /= seq_len;
            }
        }
    }

    return total_loss / seq_len;
}
