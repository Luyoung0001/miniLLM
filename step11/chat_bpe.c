/**
 * BPE 模型交互式对话
 * 用法: ./chat_bpe
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
               int max_tokens, float temperature, int top_k) {
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
            // 贪婪解码
            next_token = 0;
            float max_val = last_logits[0];
            for (int j = 1; j < vocab_size; j++) {
                if (last_logits[j] > max_val) {
                    max_val = last_logits[j];
                    next_token = j;
                }
            }
        } else {
            // 温度采样
            float* probs = (float*)malloc(vocab_size * sizeof(float));
            for (int j = 0; j < vocab_size; j++) {
                probs[j] = last_logits[j] / temperature;
            }
            simple_softmax(probs, vocab_size);

            // Top-K 过滤
            if (top_k > 0 && top_k < vocab_size) {
                float* sorted = (float*)malloc(vocab_size * sizeof(float));
                memcpy(sorted, probs, vocab_size * sizeof(float));
                for (int a = 0; a < top_k; a++) {
                    for (int b = a + 1; b < vocab_size; b++) {
                        if (sorted[b] > sorted[a]) {
                            float tmp = sorted[a];
                            sorted[a] = sorted[b];
                            sorted[b] = tmp;
                        }
                    }
                }
                float threshold = sorted[top_k - 1];
                free(sorted);

                float sum = 0.0f;
                for (int j = 0; j < vocab_size; j++) {
                    if (probs[j] < threshold) probs[j] = 0.0f;
                    else sum += probs[j];
                }
                if (sum > 0) {
                    for (int j = 0; j < vocab_size; j++) probs[j] /= sum;
                }
            }

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

void print_help() {
    printf("\n");
    printf("命令:\n");
    printf("  /help        - 显示帮助\n");
    printf("  /temp <值>   - 设置温度 (0-2, 默认0.8)\n");
    printf("  /len <值>    - 设置最大生成长度 (默认50)\n");
    printf("  /topk <值>   - 设置Top-K (默认20)\n");
    printf("  /quit        - 退出\n");
    printf("\n");
}

int main(int argc, char** argv) {
    const char* model_path = "model_bpe.bin";
    const char* vocab_path = "bpe_vocab.txt";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) vocab_path = argv[2];

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║        miniLLM BPE 交互式对话                    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    // 加载模型
    printf("加载模型: %s\n", model_path);
    GPTModel* model = model_load(model_path);
    if (!model) {
        printf("错误: 无法加载模型!\n");
        return 1;
    }

    // 加载词汇表
    printf("加载词汇表: %s\n", vocab_path);
    BPETokenizer* tok = bpe_tokenizer_create();
    if (bpe_load_vocab(tok, vocab_path) != 0) {
        printf("错误: 无法加载词汇表!\n");
        return 1;
    }

    printf("\n模型参数: %d\n", model_num_params(model));
    printf("词汇表大小: %d\n", bpe_vocab_size(tok));

    print_help();

    // 参数
    float temperature = 0.8f;
    int max_tokens = 50;
    int top_k = 20;

    srand(time(NULL));

    char input[1024];

    while (1) {
        printf("\n[你] > ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) break;

        // 移除换行
        int len = strlen(input);
        if (len > 0 && input[len-1] == '\n') input[len-1] = '\0';

        // 空输入
        if (strlen(input) == 0) continue;

        // 命令处理
        if (input[0] == '/') {
            if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
                printf("再见!\n");
                break;
            } else if (strcmp(input, "/help") == 0) {
                print_help();
            } else if (strncmp(input, "/temp ", 6) == 0) {
                temperature = atof(input + 6);
                printf("温度设置为: %.2f\n", temperature);
            } else if (strncmp(input, "/len ", 5) == 0) {
                max_tokens = atoi(input + 5);
                printf("最大长度设置为: %d\n", max_tokens);
            } else if (strncmp(input, "/topk ", 6) == 0) {
                top_k = atoi(input + 6);
                printf("Top-K设置为: %d\n", top_k);
            } else {
                printf("未知命令。输入 /help 查看帮助。\n");
            }
            continue;
        }

        // 生成回复
        char* response = generate(model, tok, input, max_tokens, temperature, top_k);

        printf("\n[模型] > %s\n", response);

        free(response);
    }

    // 清理
    model_free(model);
    bpe_tokenizer_free(tok);

    return 0;
}
