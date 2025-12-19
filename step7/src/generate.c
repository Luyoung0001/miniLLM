#include "generate.h"
#include "math_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// ============ 配置 ============

GenerateConfig default_generate_config(void) {
    GenerateConfig config;
    config.temperature = 1.0f;
    config.top_k = 40;
    config.top_p = 0.9f;
    config.max_new_tokens = 100;
    config.do_sample = 1;
    config.seed = -1;
    return config;
}

GenerateConfig greedy_config(void) {
    GenerateConfig config;
    config.temperature = 1.0f;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.max_new_tokens = 100;
    config.do_sample = 0;
    config.seed = -1;
    return config;
}

GenerateConfig creative_config(void) {
    GenerateConfig config;
    config.temperature = 1.2f;
    config.top_k = 50;
    config.top_p = 0.95f;
    config.max_new_tokens = 200;
    config.do_sample = 1;
    config.seed = -1;
    return config;
}

// ============ 辅助函数 ============

void set_random_seed(int seed) {
    if (seed < 0) {
        srand((unsigned int)time(NULL));
    } else {
        srand((unsigned int)seed);
    }
}

static float rand_uniform(void) {
    return (float)rand() / (float)RAND_MAX;
}

void apply_temperature(Tensor* logits, float temperature) {
    if (logits == NULL || temperature <= 0.0f) return;

    for (int i = 0; i < logits->size; i++) {
        logits->data[i] /= temperature;
    }
}

void apply_top_k(Tensor* logits, int k) {
    if (logits == NULL || k <= 0 || k >= logits->size) return;

    int vocab_size = logits->size;

    // 找到第 k 大的值
    // 简单实现: 复制并排序
    float* sorted = (float*)malloc(vocab_size * sizeof(float));
    memcpy(sorted, logits->data, vocab_size * sizeof(float));

    // 降序排序 (简单冒泡，对于小 vocab_size 足够)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (sorted[j] > sorted[i]) {
                float temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }

    float threshold = sorted[k - 1];
    free(sorted);

    // 将小于阈值的设为 -inf
    for (int i = 0; i < vocab_size; i++) {
        if (logits->data[i] < threshold) {
            logits->data[i] = -INFINITY;
        }
    }
}

void apply_top_p(Tensor* logits, float p) {
    if (logits == NULL || p >= 1.0f) return;

    int vocab_size = logits->size;

    // 先计算 softmax
    int shape[] = {vocab_size};
    Tensor* probs = tensor_zeros(1, shape);
    softmax(probs, logits);

    // 创建索引数组并按概率降序排列
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }

    // 按概率降序排序索引
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs->data[indices[j]] > probs->data[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    // 计算累积概率，找到截止点
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs->data[indices[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // 将不在 top-p 集合中的设为 -inf
    for (int i = cutoff; i < vocab_size; i++) {
        logits->data[indices[i]] = -INFINITY;
    }

    free(indices);
    tensor_free(probs);
}

int sample_from_probs(Tensor* probs) {
    if (probs == NULL) return 0;

    float r = rand_uniform();
    float cumsum = 0.0f;

    for (int i = 0; i < probs->size; i++) {
        cumsum += probs->data[i];
        if (r < cumsum) {
            return i;
        }
    }

    // 返回最后一个 (处理浮点误差)
    return probs->size - 1;
}

// ============ 采样函数 ============

int sample_greedy(Tensor* logits) {
    return tensor_argmax(logits);
}

int sample_temperature(Tensor* logits, float temperature) {
    if (logits == NULL) return 0;

    int vocab_size = logits->size;
    int shape[] = {vocab_size};

    // 复制 logits
    Tensor* scaled_logits = tensor_zeros(1, shape);
    memcpy(scaled_logits->data, logits->data, vocab_size * sizeof(float));

    // 应用温度
    apply_temperature(scaled_logits, temperature);

    // Softmax
    Tensor* probs = tensor_zeros(1, shape);
    softmax(probs, scaled_logits);

    // 采样
    int token = sample_from_probs(probs);

    tensor_free(scaled_logits);
    tensor_free(probs);

    return token;
}

int sample_top_k(Tensor* logits, int k, float temperature) {
    if (logits == NULL) return 0;

    int vocab_size = logits->size;
    int shape[] = {vocab_size};

    // 复制 logits
    Tensor* filtered_logits = tensor_zeros(1, shape);
    memcpy(filtered_logits->data, logits->data, vocab_size * sizeof(float));

    // 应用 Top-K 过滤
    apply_top_k(filtered_logits, k);

    // 应用温度
    apply_temperature(filtered_logits, temperature);

    // Softmax
    Tensor* probs = tensor_zeros(1, shape);
    softmax(probs, filtered_logits);

    // 采样
    int token = sample_from_probs(probs);

    tensor_free(filtered_logits);
    tensor_free(probs);

    return token;
}

int sample_top_p(Tensor* logits, float p, float temperature) {
    if (logits == NULL) return 0;

    int vocab_size = logits->size;
    int shape[] = {vocab_size};

    // 复制 logits
    Tensor* filtered_logits = tensor_zeros(1, shape);
    memcpy(filtered_logits->data, logits->data, vocab_size * sizeof(float));

    // 应用 Top-P 过滤
    apply_top_p(filtered_logits, p);

    // 应用温度
    apply_temperature(filtered_logits, temperature);

    // Softmax
    Tensor* probs = tensor_zeros(1, shape);
    softmax(probs, filtered_logits);

    // 采样
    int token = sample_from_probs(probs);

    tensor_free(filtered_logits);
    tensor_free(probs);

    return token;
}

int sample_combined(Tensor* logits, GenerateConfig* config) {
    if (logits == NULL || config == NULL) return 0;

    // 贪婪模式
    if (!config->do_sample) {
        return sample_greedy(logits);
    }

    int vocab_size = logits->size;
    int shape[] = {vocab_size};

    // 复制 logits
    Tensor* filtered_logits = tensor_zeros(1, shape);
    memcpy(filtered_logits->data, logits->data, vocab_size * sizeof(float));

    // 应用 Top-K (如果启用)
    if (config->top_k > 0 && config->top_k < vocab_size) {
        apply_top_k(filtered_logits, config->top_k);
    }

    // 应用 Top-P (如果启用)
    if (config->top_p < 1.0f) {
        apply_top_p(filtered_logits, config->top_p);
    }

    // 应用温度
    if (config->temperature != 1.0f) {
        apply_temperature(filtered_logits, config->temperature);
    }

    // Softmax
    Tensor* probs = tensor_zeros(1, shape);
    softmax(probs, filtered_logits);

    // 采样
    int token = sample_from_probs(probs);

    tensor_free(filtered_logits);
    tensor_free(probs);

    return token;
}

// ============ 生成函数 ============

void generate_result_free(GenerateResult* result) {
    if (result == NULL) return;
    free(result->token_ids);
    free(result->text);
    free(result);
}

GenerateResult* generate_full(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
) {
    if (model == NULL || tokenizer == NULL || config == NULL) return NULL;

    // 设置随机种子
    set_random_seed(config->seed);

    // 编码 prompt
    int prompt_len;
    int* prompt_ids = tokenizer_encode(tokenizer, prompt, &prompt_len, 0, 0);
    if (prompt_ids == NULL) {
        prompt_len = 0;
        prompt_ids = (int*)malloc(sizeof(int));
        prompt_ids[0] = tokenizer->bos_id;
        prompt_len = 1;
    }

    // 分配 token 数组 (prompt + max_new_tokens)
    int max_len = prompt_len + config->max_new_tokens;
    if (max_len > model->config.max_seq_len) {
        max_len = model->config.max_seq_len;
    }

    int* tokens = (int*)malloc(max_len * sizeof(int));
    memcpy(tokens, prompt_ids, prompt_len * sizeof(int));
    free(prompt_ids);

    int current_len = prompt_len;
    int vocab_size = model->config.vocab_size;

    // 创建 logits 张量
    int logits_shape[] = {1, vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    int last_logits_shape[] = {vocab_size};
    Tensor* last_logits = tensor_zeros(1, last_logits_shape);

    // 自回归生成
    for (int i = 0; i < config->max_new_tokens; i++) {
        if (current_len >= max_len) break;

        // 前向传播 (使用所有已有的 tokens)
        // 为了效率，实际应用会使用 KV cache
        int seq_logits_shape[] = {current_len, vocab_size};
        Tensor* seq_logits = tensor_zeros(2, seq_logits_shape);

        model_forward(model, tokens, current_len, NULL, seq_logits);

        // 获取最后一个位置的 logits
        model_get_last_logits(seq_logits, current_len, last_logits);

        tensor_free(seq_logits);

        // 采样下一个 token
        int next_token = sample_combined(last_logits, config);

        // 检查是否是 EOS
        if (next_token == tokenizer->eos_id) {
            break;
        }

        // 添加到序列
        tokens[current_len] = next_token;
        current_len++;
    }

    tensor_free(logits);
    tensor_free(last_logits);

    // 创建结果
    GenerateResult* result = (GenerateResult*)malloc(sizeof(GenerateResult));
    result->token_ids = tokens;
    result->num_tokens = current_len;
    result->prompt_len = prompt_len;
    result->generated_len = current_len - prompt_len;

    // 解码文本
    result->text = tokenizer_decode(tokenizer, tokens, current_len);

    return result;
}

char* generate(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config
) {
    GenerateResult* result = generate_full(model, tokenizer, prompt, config);
    if (result == NULL) return NULL;

    // 复制文本
    char* text = result->text;
    result->text = NULL;  // 防止被 free

    // 释放结果 (但保留文本)
    free(result->token_ids);
    free(result);

    return text;
}

int generate_stream(
    GPTModel* model,
    Tokenizer* tokenizer,
    const char* prompt,
    GenerateConfig* config,
    GenerateCallback callback,
    void* user_data
) {
    if (model == NULL || tokenizer == NULL || config == NULL || callback == NULL) {
        return -1;
    }

    // 设置随机种子
    set_random_seed(config->seed);

    // 编码 prompt
    int prompt_len;
    int* prompt_ids = tokenizer_encode(tokenizer, prompt, &prompt_len, 0, 0);
    if (prompt_ids == NULL) {
        prompt_len = 0;
        prompt_ids = (int*)malloc(sizeof(int));
        prompt_ids[0] = tokenizer->bos_id;
        prompt_len = 1;
    }

    // 分配 token 数组
    int max_len = prompt_len + config->max_new_tokens;
    if (max_len > model->config.max_seq_len) {
        max_len = model->config.max_seq_len;
    }

    int* tokens = (int*)malloc(max_len * sizeof(int));
    memcpy(tokens, prompt_ids, prompt_len * sizeof(int));
    free(prompt_ids);

    int current_len = prompt_len;
    int vocab_size = model->config.vocab_size;
    int generated = 0;

    int last_logits_shape[] = {vocab_size};
    Tensor* last_logits = tensor_zeros(1, last_logits_shape);

    // 自回归生成
    for (int i = 0; i < config->max_new_tokens; i++) {
        if (current_len >= max_len) break;

        // 前向传播
        int seq_logits_shape[] = {current_len, vocab_size};
        Tensor* seq_logits = tensor_zeros(2, seq_logits_shape);

        model_forward(model, tokens, current_len, NULL, seq_logits);
        model_get_last_logits(seq_logits, current_len, last_logits);

        tensor_free(seq_logits);

        // 采样下一个 token
        int next_token = sample_combined(last_logits, config);

        // 检查是否是 EOS
        if (next_token == tokenizer->eos_id) {
            break;
        }

        // 添加到序列
        tokens[current_len] = next_token;
        current_len++;
        generated++;

        // 调用回调
        const char* token_str = tokenizer_decode_id(tokenizer, next_token);
        int should_stop = callback(next_token, token_str, user_data);
        if (should_stop) {
            break;
        }
    }

    tensor_free(last_logits);
    free(tokens);

    return generated;
}
