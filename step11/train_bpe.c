/**
 * BPE 训练脚本
 *
 * 使用 BPE tokenizer 训练语言模型
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "src/bpe_tokenizer.h"
#include "src/model.h"
#include "src/config.h"
#include "src/loss.h"
#include "src/backward.h"
#include "src/optimizer.h"
#include "src/math_ops.h"
#include "src/tensor.h"

// ============ 训练数据加载 ============

typedef struct {
    int* tokens;
    int num_tokens;
    int seq_len;
    int batch_size;
    int current_pos;
} BPEDataLoader;

BPEDataLoader* bpe_dataloader_create(BPETokenizer* tok, const char* filepath,
                                      int seq_len, int batch_size) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("无法打开文件: %s\n", filepath);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* text = (char*)malloc(size + 1);
    size_t read_size = fread(text, 1, size, f);
    text[read_size] = '\0';
    fclose(f);

    int num_tokens;
    int* tokens = bpe_encode(tok, text, &num_tokens, 0, 0);
    free(text);

    if (tokens == NULL || num_tokens < seq_len + 1) {
        printf("数据太少! 需要至少 %d tokens, 只有 %d\n", seq_len + 1, num_tokens);
        if (tokens) free(tokens);
        return NULL;
    }

    BPEDataLoader* dl = (BPEDataLoader*)malloc(sizeof(BPEDataLoader));
    dl->tokens = tokens;
    dl->num_tokens = num_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->current_pos = 0;

    printf("数据加载完成: %d tokens, seq_len=%d, batch_size=%d\n",
           num_tokens, seq_len, batch_size);

    return dl;
}

void bpe_dataloader_free(BPEDataLoader* dl) {
    if (dl) {
        if (dl->tokens) free(dl->tokens);
        free(dl);
    }
}

int bpe_dataloader_next_batch(BPEDataLoader* dl, int* input_ids, int* targets) {
    int samples_available = (dl->num_tokens - dl->current_pos - 1) / dl->seq_len;
    if (samples_available <= 0) return 0;

    int actual_batch = (samples_available < dl->batch_size) ? samples_available : dl->batch_size;

    for (int b = 0; b < actual_batch; b++) {
        int start = dl->current_pos + b * dl->seq_len;
        for (int i = 0; i < dl->seq_len; i++) {
            input_ids[b * dl->seq_len + i] = dl->tokens[start + i];
            targets[b * dl->seq_len + i] = dl->tokens[start + i + 1];
        }
    }

    dl->current_pos += actual_batch * dl->seq_len;
    return actual_batch;
}

void bpe_dataloader_reset(BPEDataLoader* dl) {
    dl->current_pos = 0;
}

// ============ 简单 softmax 实现 ============

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

// ============ BPE 文本生成 ============

char* bpe_generate(GPTModel* model, BPETokenizer* tok,
                   const char* prompt, int max_new_tokens,
                   float temperature, int top_k) {
    int prompt_len;
    int* input_ids = bpe_encode(tok, prompt, &prompt_len, 0, 0);
    if (!input_ids || prompt_len == 0) {
        return strdup(prompt);
    }

    int max_len = prompt_len + max_new_tokens;
    if (max_len > model->config.max_seq_len) {
        max_len = model->config.max_seq_len;
    }

    int* tokens = (int*)malloc(max_len * sizeof(int));
    memcpy(tokens, input_ids, prompt_len * sizeof(int));
    free(input_ids);

    int current_len = prompt_len;
    int vocab_size = model->config.vocab_size;
    srand(time(NULL));

    // 创建 cache 和 logits
    GPTCache* cache = model_cache_create(model, max_len);
    Tensor* logits = tensor_create_2d(max_len, vocab_size);

    for (int i = 0; i < max_new_tokens && current_len < max_len; i++) {
        // 前向传播
        model_forward(model, tokens, current_len, cache, logits);

        // 获取最后一个位置的 logits
        float* last_logits = logits->data + (current_len - 1) * vocab_size;

        // 采样
        int next_token;
        if (temperature <= 0.0f) {
            // 贪婪
            next_token = 0;
            float max_val = last_logits[0];
            for (int j = 1; j < vocab_size; j++) {
                if (last_logits[j] > max_val) {
                    max_val = last_logits[j];
                    next_token = j;
                }
            }
        } else {
            // 复制并应用温度
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
                    if (probs[j] < threshold) {
                        probs[j] = 0.0f;
                    } else {
                        sum += probs[j];
                    }
                }
                if (sum > 0) {
                    for (int j = 0; j < vocab_size; j++) {
                        probs[j] /= sum;
                    }
                }
            }

            // 从概率分布采样
            float r = (float)rand() / (float)RAND_MAX;
            float cumsum = 0.0f;
            next_token = vocab_size - 1;
            for (int j = 0; j < vocab_size; j++) {
                cumsum += probs[j];
                if (cumsum > r) {
                    next_token = j;
                    break;
                }
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

// ============ 主函数 ============

int main(int argc, char** argv) {
    printf("=== miniLLM BPE 训练 ===\n\n");

    const char* data_path = "train_data.txt";
    const char* vocab_path = "bpe_vocab.txt";
    const char* model_path = "model_bpe.bin";
    int num_merges = 500;
    int num_epochs = 10;
    int seq_len = 64;
    int batch_size = 4;
    float learning_rate = 0.0003f;  // 降低学习率，更稳定

    if (argc > 1) data_path = argv[1];
    if (argc > 2) num_epochs = atoi(argv[2]);
    if (argc > 3) learning_rate = atof(argv[3]);  // 可选: 自定义学习率

    // 1. 创建并训练 BPE tokenizer
    printf("1. 训练 BPE tokenizer...\n");
    BPETokenizer* tok = bpe_tokenizer_create();

    FILE* f = fopen(data_path, "r");
    if (!f) {
        printf("无法打开训练数据: %s\n", data_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = (char*)malloc(size + 1);
    size_t read_size = fread(text, 1, size, f);
    text[read_size] = '\0';
    fclose(f);

    printf("   训练数据大小: %.2f KB\n", size / 1024.0f);
    bpe_train(tok, text, num_merges);
    free(text);

    bpe_save_vocab(tok, vocab_path);
    printf("   词汇表已保存: %s\n", vocab_path);
    printf("   词汇表大小: %d\n\n", bpe_vocab_size(tok));

    // 2. 创建模型
    printf("2. 创建模型...\n");
    ModelConfig config = bpe_config(bpe_vocab_size(tok));  // 使用更大的模型
    config.max_seq_len = seq_len + 16;
    config_print(&config);

    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    int num_params = model_num_params(model);
    printf("   参数量: %d (%.2f K)\n\n", num_params, num_params / 1000.0f);

    // 3. 创建数据加载器
    printf("3. 创建数据加载器...\n");
    BPEDataLoader* dl = bpe_dataloader_create(tok, data_path, seq_len, batch_size);
    if (!dl) {
        printf("创建数据加载器失败!\n");
        return 1;
    }

    // 4. 创建优化器和梯度
    printf("4. 初始化训练...\n");
    AdamOptimizer* opt = adam_create(model, learning_rate);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));
    Tensor* logits = tensor_create_2d(seq_len, model->config.vocab_size);
    Tensor* grad_logits = tensor_create_2d(seq_len, model->config.vocab_size);

    // 5. 训练循环
    printf("\n5. 开始训练 (%d epochs)...\n\n", num_epochs);

    float best_loss = 1e9f;
    int total_steps = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        bpe_dataloader_reset(dl);
        float epoch_loss = 0.0f;
        int epoch_steps = 0;

        int actual_batch;
        while ((actual_batch = bpe_dataloader_next_batch(dl, input_ids, targets)) > 0) {
            // 对每个样本进行训练
            for (int b = 0; b < actual_batch; b++) {
                int* batch_input = input_ids + b * seq_len;
                int* batch_target = targets + b * seq_len;

                // 前向传播（带缓存）
                model_forward_with_cache(model, batch_input, seq_len, cache, logits);

                // 计算损失和梯度
                float loss = cross_entropy_loss_with_grad(
                    logits, batch_target, seq_len, model->config.vocab_size, grad_logits);

                // 累积梯度
                if (b == 0) gradients_zero(grads);
                model_backward(model, batch_input, seq_len, grad_logits, cache, grads);

                epoch_loss += loss;
            }

            // 更新权重
            adam_step(opt, model, grads);
            epoch_steps++;
            total_steps++;

            if (total_steps % 10 == 0) {
                printf("\r   Epoch %d, Step %d, Loss: %.4f",
                       epoch + 1, total_steps, epoch_loss / (epoch_steps * actual_batch));
                fflush(stdout);
            }
        }

        float avg_loss = epoch_loss / (epoch_steps * batch_size);
        printf("\n   Epoch %d 完成, 平均 Loss: %.4f, PPL: %.2f\n",
               epoch + 1, avg_loss, expf(avg_loss));

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            model_save(model, model_path);
            printf("   -> 保存最佳模型 (loss=%.4f)\n", best_loss);
        }

        // 生成示例
        if ((epoch + 1) % 2 == 0 || epoch == num_epochs - 1) {
            printf("\n   生成示例:\n");
            const char* prompts[] = {"hello", "the", "how"};
            for (int i = 0; i < 3; i++) {
                char* output = bpe_generate(model, tok, prompts[i], 30, 0.8f, 20);
                printf("     '%s' -> '%s'\n", prompts[i], output);
                free(output);
            }
            printf("\n");
        }
    }

    // 6. 最终测试
    printf("\n6. 最终生成测试:\n\n");
    const char* test_prompts[] = {"hello ", "the ", "how are ", "good ", "one "};

    for (int i = 0; i < 5; i++) {
        char* output = bpe_generate(model, tok, test_prompts[i], 40, 0.7f, 30);
        printf("   '%s' -> '%s'\n", test_prompts[i], output);
        free(output);
    }

    // 清理
    free(input_ids);
    free(targets);
    tensor_free(logits);
    tensor_free(grad_logits);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(opt);
    bpe_dataloader_free(dl);
    model_free(model);
    bpe_tokenizer_free(tok);

    printf("\n=== 训练完成! ===\n");
    printf("模型已保存: %s\n", model_path);
    printf("词汇表已保存: %s\n", vocab_path);

    return 0;
}
