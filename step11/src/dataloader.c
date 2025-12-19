#define _POSIX_C_SOURCE 200809L
#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 读取整个文件到字符串
static char* read_file(const char* filepath, int* out_len) {
    FILE* f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "Error: cannot open file '%s'\n", filepath);
        return NULL;
    }

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // 分配内存
    char* buffer = (char*)malloc(size + 1);
    if (buffer == NULL) {
        fclose(f);
        return NULL;
    }

    // 读取文件
    size_t read_size = fread(buffer, 1, size, f);
    buffer[read_size] = '\0';
    fclose(f);

    if (out_len) *out_len = (int)read_size;
    return buffer;
}

DataLoader* dataloader_create(
    const char* filepath,
    Tokenizer* tokenizer,
    int seq_len,
    int batch_size
) {
    int raw_len;
    char* raw_data = read_file(filepath, &raw_len);
    if (raw_data == NULL) return NULL;

    DataLoader* dl = dataloader_from_string(raw_data, tokenizer, seq_len, batch_size);
    if (dl != NULL) {
        dl->raw_data = raw_data;
        dl->raw_len = raw_len;
    } else {
        free(raw_data);
    }

    return dl;
}

DataLoader* dataloader_from_string(
    const char* text,
    Tokenizer* tokenizer,
    int seq_len,
    int batch_size
) {
    if (text == NULL || tokenizer == NULL) return NULL;

    DataLoader* dl = (DataLoader*)malloc(sizeof(DataLoader));
    if (dl == NULL) return NULL;

    // 复制原始文本
    int text_len = strlen(text);
    dl->raw_data = strdup(text);
    dl->raw_len = text_len;

    // 编码文本
    dl->tokens = tokenizer_encode(tokenizer, text, &dl->num_tokens, 0, 0);
    if (dl->tokens == NULL) {
        free(dl->raw_data);
        free(dl);
        return NULL;
    }

    dl->tokenizer = tokenizer;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;
    dl->current_pos = 0;

    // 计算样本数 (每个样本需要 seq_len + 1 个 token)
    dl->num_samples = (dl->num_tokens - 1) / seq_len;
    if (dl->num_samples < 1) {
        fprintf(stderr, "Warning: not enough tokens for even one sample\n");
        dl->num_samples = 1;
    }

    // 初始化打乱索引
    dl->shuffle_indices = (int*)malloc(dl->num_samples * sizeof(int));
    if (dl->shuffle_indices == NULL) {
        free(dl->tokens);
        free(dl->raw_data);
        free(dl);
        return NULL;
    }

    for (int i = 0; i < dl->num_samples; i++) {
        dl->shuffle_indices[i] = i * seq_len;
    }
    dl->current_sample = 0;

    return dl;
}

void dataloader_free(DataLoader* dl) {
    if (dl == NULL) return;
    free(dl->raw_data);
    free(dl->tokens);
    free(dl->shuffle_indices);
    free(dl);
}

int dataloader_next_batch(DataLoader* dl, int* input_ids, int* targets) {
    if (dl == NULL || input_ids == NULL || targets == NULL) return 0;

    int actual_batch_size = 0;

    for (int b = 0; b < dl->batch_size; b++) {
        if (dl->current_sample >= dl->num_samples) {
            break;
        }

        int start_pos = dl->shuffle_indices[dl->current_sample];

        // 确保不超出边界
        if (start_pos + dl->seq_len + 1 > dl->num_tokens) {
            start_pos = dl->num_tokens - dl->seq_len - 1;
            if (start_pos < 0) start_pos = 0;
        }

        // 复制 input_ids
        for (int s = 0; s < dl->seq_len; s++) {
            input_ids[b * dl->seq_len + s] = dl->tokens[start_pos + s];
        }

        // targets = input_ids 右移一位
        for (int s = 0; s < dl->seq_len; s++) {
            targets[b * dl->seq_len + s] = dl->tokens[start_pos + s + 1];
        }

        dl->current_sample++;
        actual_batch_size++;
    }

    return actual_batch_size;
}

void dataloader_reset(DataLoader* dl) {
    if (dl == NULL) return;
    dl->current_sample = 0;
    dl->current_pos = 0;
}

void dataloader_shuffle(DataLoader* dl) {
    if (dl == NULL) return;

    // Fisher-Yates 洗牌算法
    for (int i = dl->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = dl->shuffle_indices[i];
        dl->shuffle_indices[i] = dl->shuffle_indices[j];
        dl->shuffle_indices[j] = temp;
    }
}

int dataloader_num_batches(DataLoader* dl) {
    if (dl == NULL) return 0;
    return (dl->num_samples + dl->batch_size - 1) / dl->batch_size;
}

float dataloader_progress(DataLoader* dl) {
    if (dl == NULL || dl->num_samples == 0) return 0.0f;
    return (float)dl->current_sample / (float)dl->num_samples;
}

void dataloader_print_info(DataLoader* dl) {
    if (dl == NULL) {
        printf("DataLoader: NULL\n");
        return;
    }

    printf("DataLoader Info:\n");
    printf("  Raw text length: %d chars\n", dl->raw_len);
    printf("  Total tokens: %d\n", dl->num_tokens);
    printf("  Sequence length: %d\n", dl->seq_len);
    printf("  Batch size: %d\n", dl->batch_size);
    printf("  Total samples: %d\n", dl->num_samples);
    printf("  Total batches: %d\n", dataloader_num_batches(dl));
}
