/**
 * 继续训练 BPE 模型
 *
 * 用法: ./continue_train <训练数据文件> [epochs]
 *
 * 这个脚本会加载现有的模型和词汇表，然后继续训练
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
#include "src/tensor.h"

// 数据加载器
typedef struct {
    int* tokens;
    int num_tokens;
    int seq_len;
    int batch_size;
    int current_pos;
} BPEDataLoader;

BPEDataLoader* dataloader_create(BPETokenizer* tok, const char* filepath,
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

    return dl;
}

void dataloader_free(BPEDataLoader* dl) {
    if (dl) {
        if (dl->tokens) free(dl->tokens);
        free(dl);
    }
}

int dataloader_next_batch(BPEDataLoader* dl, int* input_ids, int* targets) {
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

void dataloader_reset(BPEDataLoader* dl) {
    dl->current_pos = 0;
}

// 简单生成函数
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

char* generate_sample(GPTModel* model, BPETokenizer* tok, const char* prompt, int max_tokens) {
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

        // 温度采样
        float* probs = (float*)malloc(vocab_size * sizeof(float));
        for (int j = 0; j < vocab_size; j++) {
            probs[j] = last_logits[j] / 0.8f;
        }
        simple_softmax(probs, vocab_size);

        float r = (float)rand() / (float)RAND_MAX;
        float cumsum = 0.0f;
        int next_token = vocab_size - 1;
        for (int j = 0; j < vocab_size; j++) {
            cumsum += probs[j];
            if (cumsum > r) { next_token = j; break; }
        }
        free(probs);

        if (next_token == tok->eos_id) break;
        tokens[current_len++] = next_token;
    }

    model_cache_free(cache);
    tensor_free(logits);

    char* result = bpe_decode(tok, tokens, current_len);
    free(tokens);
    return result;
}

int main(int argc, char** argv) {
    printf("\n=== 继续训练 BPE 模型 ===\n\n");

    if (argc < 2) {
        printf("用法: %s <训练数据文件> [epochs] [learning_rate]\n", argv[0]);
        printf("例如: %s train_data.txt 50 0.0005\n", argv[0]);
        return 1;
    }

    const char* data_path = argv[1];
    int num_epochs = (argc > 2) ? atoi(argv[2]) : 50;
    float learning_rate = (argc > 3) ? atof(argv[3]) : 0.0005f;

    const char* model_path = "model_bpe.bin";
    const char* vocab_path = "bpe_vocab.txt";

    // 1. 加载词汇表
    printf("1. 加载词汇表: %s\n", vocab_path);
    BPETokenizer* tok = bpe_tokenizer_create();
    if (bpe_load_vocab(tok, vocab_path) != 0) {
        printf("错误: 无法加载词汇表! 请先运行 train_bpe 训练。\n");
        return 1;
    }
    printf("   词汇表大小: %d\n", bpe_vocab_size(tok));

    // 2. 加载模型
    printf("\n2. 加载模型: %s\n", model_path);
    GPTModel* model = model_load(model_path);
    if (!model) {
        printf("错误: 无法加载模型! 请先运行 train_bpe 训练。\n");
        return 1;
    }
    printf("   模型参数: %d\n", model_num_params(model));

    int seq_len = model->config.max_seq_len - 16;
    int batch_size = 4;

    // 3. 创建数据加载器
    printf("\n3. 加载训练数据: %s\n", data_path);
    BPEDataLoader* dl = dataloader_create(tok, data_path, seq_len, batch_size);
    if (!dl) {
        printf("创建数据加载器失败!\n");
        return 1;
    }
    printf("   Tokens: %d, seq_len: %d, batch_size: %d\n",
           dl->num_tokens, seq_len, batch_size);

    // 4. 创建优化器
    printf("\n4. 初始化训练 (lr=%.6f, epochs=%d)\n", learning_rate, num_epochs);
    AdamOptimizer* opt = adam_create(model, learning_rate);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));
    Tensor* logits = tensor_create_2d(seq_len, model->config.vocab_size);
    Tensor* grad_logits = tensor_create_2d(seq_len, model->config.vocab_size);

    // 5. 训练
    printf("\n5. 开始继续训练...\n\n");

    srand(time(NULL));
    float best_loss = 1e9f;
    int total_steps = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        dataloader_reset(dl);
        float epoch_loss = 0.0f;
        int epoch_steps = 0;

        int actual_batch;
        while ((actual_batch = dataloader_next_batch(dl, input_ids, targets)) > 0) {
            for (int b = 0; b < actual_batch; b++) {
                int* batch_input = input_ids + b * seq_len;
                int* batch_target = targets + b * seq_len;

                model_forward_with_cache(model, batch_input, seq_len, cache, logits);

                float loss = cross_entropy_loss_with_grad(
                    logits, batch_target, seq_len, model->config.vocab_size, grad_logits);

                if (b == 0) gradients_zero(grads);
                model_backward(model, batch_input, seq_len, grad_logits, cache, grads);

                epoch_loss += loss;
            }

            adam_step(opt, model, grads);
            epoch_steps++;
            total_steps++;
        }

        float avg_loss = epoch_loss / (epoch_steps * batch_size);

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("Epoch %3d/%d | Loss: %.4f | PPL: %.2f\n",
                   epoch + 1, num_epochs, avg_loss, expf(avg_loss));
        }

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            model_save(model, model_path);
        }

        // 每20个epoch展示一次生成
        if ((epoch + 1) % 20 == 0) {
            printf("\n  生成示例:\n");
            const char* prompts[] = {"hello", "the", "good"};
            for (int i = 0; i < 3; i++) {
                char* output = generate_sample(model, tok, prompts[i], 30);
                printf("    '%s' -> '%.60s...'\n", prompts[i], output);
                free(output);
            }
            printf("\n");
        }
    }

    // 6. 保存
    printf("\n6. 保存模型...\n");
    model_save(model, model_path);
    printf("   模型已保存: %s\n", model_path);
    printf("   最佳 Loss: %.4f\n", best_loss);

    // 清理
    free(input_ids);
    free(targets);
    tensor_free(logits);
    tensor_free(grad_logits);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(opt);
    dataloader_free(dl);
    model_free(model);
    bpe_tokenizer_free(tok);

    printf("\n=== 训练完成! ===\n");
    return 0;
}
