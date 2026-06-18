#ifndef BACKWARD_H
#define BACKWARD_H

#include "tensor.h"
#include "model.h"

/**
 * 模型梯度结构
 * 存储所有可训练参数的梯度
 */
typedef struct {
    // Embedding 梯度
    Tensor* d_token_embedding;      // [vocab_size, hidden_dim]
    Tensor* d_position_embedding;   // [max_seq_len, hidden_dim]

    // Transformer 层梯度 (每层)
    struct {
        // LayerNorm1
        Tensor* d_ln1_gamma;        // [hidden_dim]
        Tensor* d_ln1_beta;         // [hidden_dim]

        // Attention
        Tensor* d_W_q;              // [hidden_dim, hidden_dim]
        Tensor* d_W_k;              // [hidden_dim, hidden_dim]
        Tensor* d_W_v;              // [hidden_dim, hidden_dim]
        Tensor* d_W_o;              // [hidden_dim, hidden_dim]

        // LayerNorm2
        Tensor* d_ln2_gamma;        // [hidden_dim]
        Tensor* d_ln2_beta;         // [hidden_dim]

        // FFN
        Tensor* d_W1;               // [hidden_dim, ffn_dim]
        Tensor* d_b1;               // [ffn_dim]
        Tensor* d_W2;               // [ffn_dim, hidden_dim]
        Tensor* d_b2;               // [hidden_dim]
    }* layers;

    // Final LayerNorm 梯度
    Tensor* d_final_ln_gamma;       // [hidden_dim]
    Tensor* d_final_ln_beta;        // [hidden_dim]

    // LM Head 梯度
    Tensor* d_lm_head;              // [hidden_dim, vocab_size]

    int num_layers;
} Gradients;

/**
 * 反向传播中间缓存
 */
typedef struct {
    // 前向传播保存的激活值
    Tensor* embed_output;           // embedding 输出
    Tensor** layer_inputs;          // 每层输入
    Tensor** ln1_outputs;           // ln1 输出
    Tensor** attn_outputs;          // attention 输出
    Tensor** ln2_outputs;           // ln2 输出
    Tensor** ffn_hidden;            // ffn 中间层
    Tensor* final_ln_input;         // final ln 输入
    Tensor* final_ln_output;        // final ln 输出

    int num_layers;
    int seq_len;
    int hidden_dim;
} BackwardCache;

// ============ Gradients 操作 ============

/**
 * 创建梯度结构
 */
Gradients* gradients_create(GPTModel* model);

/**
 * 释放梯度结构
 */
void gradients_free(Gradients* grads);

/**
 * 将所有梯度清零
 */
void gradients_zero(Gradients* grads);

/**
 * 梯度裁剪 (防止梯度爆炸)
 * @param max_norm 最大梯度范数
 * @return 实际的梯度范数
 */
float gradients_clip(Gradients* grads, float max_norm);

/**
 * 计算梯度的 L2 范数
 */
float gradients_norm(Gradients* grads);

// ============ BackwardCache 操作 ============

/**
 * 创建反向传播缓存
 */
BackwardCache* backward_cache_create(GPTModel* model, int seq_len);

/**
 * 释放反向传播缓存
 */
void backward_cache_free(BackwardCache* cache);

// ============ 前向传播 (带缓存) ============

/**
 * 带梯度缓存的前向传播
 * 保存中间激活值用于反向传播
 */
void model_forward_with_cache(
    GPTModel* model,
    int* input_ids,
    int seq_len,
    BackwardCache* cache,
    Tensor* logits
);

// ============ 反向传播 ============

/**
 * 完整模型反向传播
 * @param model 模型
 * @param input_ids 输入 token IDs
 * @param seq_len 序列长度
 * @param grad_logits 来自损失函数的梯度 [seq_len, vocab_size]
 * @param cache 前向传播保存的缓存
 * @param grads 输出的参数梯度
 */
void model_backward(
    GPTModel* model,
    int* input_ids,
    int seq_len,
    Tensor* grad_logits,
    BackwardCache* cache,
    Gradients* grads
);

// ============ 各模块反向传播 ============

/**
 * 线性层反向传播
 * y = x @ W
 * dL/dW = x^T @ dL/dy
 * dL/dx = dL/dy @ W^T
 */
void linear_backward(
    Tensor* grad_output,    // [seq_len, out_dim]
    Tensor* input,          // [seq_len, in_dim]
    Tensor* weight,         // [in_dim, out_dim]
    Tensor* grad_weight,    // [in_dim, out_dim]
    Tensor* grad_input      // [seq_len, in_dim], 可为 NULL
);

/**
 * LayerNorm 反向传播
 */
void layernorm_backward(
    Tensor* grad_output,    // [seq_len, hidden_dim]
    Tensor* input,          // [seq_len, hidden_dim]
    Tensor* gamma,          // [hidden_dim]
    Tensor* grad_gamma,     // [hidden_dim]
    Tensor* grad_beta,      // [hidden_dim]
    Tensor* grad_input,     // [seq_len, hidden_dim]
    float eps
);

/**
 * GELU 反向传播
 */
void gelu_backward(
    Tensor* grad_output,    // [seq_len, dim]
    Tensor* input,          // [seq_len, dim]
    Tensor* grad_input      // [seq_len, dim]
);

/**
 * Softmax 反向传播 (用于 attention)
 */
void softmax_backward(
    Tensor* grad_output,    // [seq_len, seq_len]
    Tensor* output,         // softmax 输出 [seq_len, seq_len]
    Tensor* grad_input      // [seq_len, seq_len]
);

/**
 * Embedding 反向传播
 * 只更新出现的 token 的梯度
 */
void embedding_backward(
    Tensor* grad_output,        // [seq_len, hidden_dim]
    int* input_ids,             // [seq_len]
    int seq_len,
    Tensor* grad_token_emb,     // [vocab_size, hidden_dim]
    Tensor* grad_pos_emb        // [max_seq_len, hidden_dim]
);

#endif // BACKWARD_H
