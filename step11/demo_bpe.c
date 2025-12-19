/**
 * BPE 模型测试 - 展示生成效果
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "src/bpe_tokenizer.h"
#include "src/model.h"
#include "src/tensor.h"

static void simple_softmax(float* data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

char* generate(GPTModel* model, BPETokenizer* tok, const char* prompt,
               int max_tokens, float temperature) {
    int prompt_len;
    int* input_ids = bpe_encode(tok, prompt, &prompt_len, 0, 0);
    if (!input_ids || prompt_len == 0) return strdup(prompt);

    int max_len = prompt_len + max_tokens;
    if (max_len > model->config.max_seq_len) max_len = model->config.max_seq_len;

    int* tokens = (int*)malloc(max_len * sizeof(int));
    memcpy(tokens, input_ids, prompt_len * sizeof(int));
    free(input_ids);

    int current_len = prompt_len;
    int vocab_size = model->config.vocab_size;

    GPTCache* cache = model_cache_create(model, max_len);
    Tensor* logits = tensor_create_2d(max_len, vocab_size);

    for (int i = 0; i < max_tokens && current_len < max_len; i++) {
        model_forward(model, tokens, current_len, cache, logits);
        float* last_logits = logits->data + (current_len - 1) * vocab_size;

        int next_token;
        if (temperature <= 0.01f) {
            next_token = 0;
            float max_val = last_logits[0];
            for (int j = 1; j < vocab_size; j++) {
                if (last_logits[j] > max_val) {
                    max_val = last_logits[j];
                    next_token = j;
                }
            }
        } else {
            float* probs = (float*)malloc(vocab_size * sizeof(float));
            for (int j = 0; j < vocab_size; j++) {
                probs[j] = last_logits[j] / temperature;
            }
            simple_softmax(probs, vocab_size);

            float r = (float)rand() / (float)RAND_MAX;
            float cumsum = 0.0f;
            next_token = vocab_size - 1;
            for (int j = 0; j < vocab_size; j++) {
                cumsum += probs[j];
                if (cumsum > r) { next_token = j; break; }
            }
            free(probs);
        }

        if (next_token == tok->eos_id) break;
        tokens[current_len++] = next_token;
    }

    model_cache_free(cache);
    tensor_free(logits);

    char* result = bpe_decode(tok, tokens, current_len);
    free(tokens);
    return result;
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     miniLLM BPE 模型生成效果展示                 ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // 加载模型和词汇表
    printf("加载模型和词汇表...\n");
    GPTModel* model = model_load("model_bpe.bin");
    if (!model) {
        printf("无法加载模型!\n");
        return 1;
    }

    BPETokenizer* tok = bpe_tokenizer_create();
    if (bpe_load_vocab(tok, "bpe_vocab.txt") != 0) {
        printf("无法加载词汇表!\n");
        return 1;
    }

    printf("模型参数: %d\n", model_num_params(model));
    printf("词汇表大小: %d\n\n", bpe_vocab_size(tok));

    srand(42);

    // 测试不同的 prompt
    printf("═══════════════════════════════════════════════════\n");
    printf("【贪婪解码 (temperature=0)】\n");
    printf("═══════════════════════════════════════════════════\n\n");

    const char* prompts[] = {
        "hello",
        "the cat",
        "good morning",
        "the sun",
        "how are"
    };

    for (int i = 0; i < 5; i++) {
        char* output = generate(model, tok, prompts[i], 50, 0.0f);
        printf("输入: \"%s\"\n", prompts[i]);
        printf("输出: \"%s\"\n\n", output);
        free(output);
    }

    printf("═══════════════════════════════════════════════════\n");
    printf("【温度采样 (temperature=0.8)】\n");
    printf("═══════════════════════════════════════════════════\n\n");

    for (int i = 0; i < 5; i++) {
        srand(time(NULL) + i);
        char* output = generate(model, tok, prompts[i], 50, 0.8f);
        printf("输入: \"%s\"\n", prompts[i]);
        printf("输出: \"%s\"\n\n", output);
        free(output);
    }

    // 清理
    model_free(model);
    bpe_tokenizer_free(tok);

    printf("═══════════════════════════════════════════════════\n");
    printf("训练数据: 1.3KB (约60行简单句子)\n");
    printf("模型大小: 234K 参数\n");
    printf("BPE压缩率: ~6.7x (1311字符 -> 195 tokens)\n");
    printf("═══════════════════════════════════════════════════\n");

    return 0;
}
