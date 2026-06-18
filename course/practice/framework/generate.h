#ifndef GENERATE_H
#define GENERATE_H

#include "model.h"
#include "tokenizer.h"

/**
 * 文本生成模块
 *
 * 支持多种采样策略:
 * 1. 贪婪解码 (Greedy): 总是选择概率最高的 token
 * 2. 温度采样 (Temperature): 调整概率分布的"锐度"
 * 3. Top-K 采样: 只从概率最高的 K 个 token 中采样
 * 4. Top-P (Nucleus) 采样: 从累积概率达到 P 的最小集合中采样
 */

/**
 * 生成配置
 */
typedef struct {
    float temperature;      // 温度参数 (默认 1.0)
                           // < 1: 更确定性  > 1: 更随机
    int top_k;             // Top-K 采样 (默认 40, 0 表示禁用)
    float top_p;           // Top-P 采样 (默认 0.9, 1.0 表示禁用)
    int max_new_tokens;    // 最大生成 token 数 (默认 100)
    int do_sample;         // 1: 采样, 0: 贪婪解码
    int seed;              // 随机种子 (-1 表示使用当前时间)
} GenerateConfig;

/**
 * 生成结果
 */
typedef struct {
    int* token_ids;        // 生成的 token IDs (包含 prompt)
    int num_tokens;        // 总 token 数
    int prompt_len;        // prompt 长度
    int generated_len;     // 生成的 token 数
    char* text;            // 解码后的文本
} GenerateResult;

// ============ 配置 ============

/**
 * 创建默认生成配置
 */
GenerateConfig default_generate_config(void);

/**
 * 创建贪婪解码配置
 */
GenerateConfig greedy_config(void);

/**
 * 创建创意生成配置 (高温度、高随机性)
 */
GenerateConfig creative_config(void);

// ============ 生成函数 ============

/**
 * 生成文本
 * @param model 模型
 * @param tokenizer tokenizer
 * @param prompt 输入提示
 * @param config 生成配置
 * @return 生成的文本 (需要 free)
 */
char* generate(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
);

/**
 * 生成文本 (返回详细结果)
 */
GenerateResult* generate_full(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
);

/**
 * 释放生成结果
 */
void generate_result_free(GenerateResult* result);

/**
 * 流式生成 (每生成一个 token 调用回调)
 * @param callback 回调函数，返回 0 继续生成，非 0 停止
 */
typedef int (*GenerateCallback)(int token_id, const char* token_str, void* user_data);

int generate_stream(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config,
    GenerateCallback callback,
    void* user_data
);

// ============ 采样函数 ============

/**
 * 贪婪采样 (选择最大概率的 token)
 */
int sample_greedy(Tensor* logits);

/**
 * 温度采样
 * @param logits [vocab_size]
 * @param temperature 温度参数
 * @return 采样的 token ID
 */
int sample_temperature(Tensor* logits, float temperature);

/**
 * Top-K 采样
 * @param logits [vocab_size]
 * @param k 保留的最高概率 token 数
 * @param temperature 温度参数
 * @return 采样的 token ID
 */
int sample_top_k(Tensor* logits, int k, float temperature);

/**
 * Top-P (Nucleus) 采样
 * @param logits [vocab_size]
 * @param p 累积概率阈值
 * @param temperature 温度参数
 * @return 采样的 token ID
 */
int sample_top_p(Tensor* logits, float p, float temperature);

/**
 * 组合采样 (Top-K + Top-P + Temperature)
 */
int sample_combined(Tensor* logits, GenerateConfig* config);

// ============ 辅助函数 ============

/**
 * 应用温度到 logits
 * logits = logits / temperature
 */
void apply_temperature(Tensor* logits, float temperature);

/**
 * 应用 Top-K 过滤
 * 将 logits 中非 top-k 的位置设为 -inf
 */
void apply_top_k(Tensor* logits, int k);

/**
 * 应用 Top-P 过滤
 * 将累积概率超过 p 的位置设为 -inf
 */
void apply_top_p(Tensor* logits, float p);

/**
 * 从概率分布中采样
 * @param probs 概率分布 [vocab_size]
 * @return 采样的索引
 */
int sample_from_probs(Tensor* probs);

/**
 * 设置随机种子
 */
void set_random_seed(int seed);

#endif // GENERATE_H
