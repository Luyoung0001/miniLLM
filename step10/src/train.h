#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"
#include "tokenizer.h"
#include "dataloader.h"
#include "optimizer.h"
#include "backward.h"

/**
 * 训练配置
 */
typedef struct {
    int num_epochs;         // 训练轮数
    float learning_rate;    // 初始学习率
    float min_lr;           // 最小学习率 (用于余弦退火)
    int warmup_steps;       // 预热步数
    int batch_size;         // 批大小
    int seq_len;            // 序列长度

    int log_interval;       // 每多少步打印一次
    int eval_interval;      // 每多少步评估一次
    int save_interval;      // 每多少步保存一次

    float grad_clip;        // 梯度裁剪阈值 (0 表示不裁剪)

    const char* data_path;  // 训练数据路径
    const char* eval_path;  // 验证数据路径 (可为 NULL)
    const char* save_path;  // 模型保存路径
} TrainConfig;

/**
 * 训练状态
 */
typedef struct {
    int current_epoch;
    int current_step;
    int total_steps;
    float current_loss;
    float avg_loss;
    float learning_rate;
    float grad_norm;
} TrainState;

/**
 * 创建默认训练配置
 */
TrainConfig default_train_config(void);

/**
 * 训练模型
 * @param model 模型
 * @param tokenizer tokenizer
 * @param config 训练配置
 * @return 最终损失值
 */
float train(GPTModel* model, Tokenizer* tokenizer, TrainConfig* config);

/**
 * 单步训练
 * @param model 模型
 * @param input_ids 输入 [batch_size, seq_len]
 * @param targets 目标 [batch_size, seq_len]
 * @param batch_size 实际批大小
 * @param seq_len 序列长度
 * @param optimizer 优化器
 * @param grads 梯度
 * @param cache 反向传播缓存
 * @param grad_clip 梯度裁剪阈值
 * @return 损失值
 */
float train_step(
    GPTModel* model,
    int* input_ids,
    int* targets,
    int batch_size,
    int seq_len,
    AdamOptimizer* optimizer,
    Gradients* grads,
    BackwardCache* cache,
    float grad_clip
);

/**
 * 评估模型
 * @param model 模型
 * @param dataloader 验证数据加载器
 * @return 平均损失值
 */
float evaluate(GPTModel* model, DataLoader* dataloader);

/**
 * 计算困惑度 (perplexity)
 */
float perplexity(float loss);

/**
 * 打印训练进度
 */
void print_train_progress(TrainState* state);

#endif // TRAIN_H
