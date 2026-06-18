#ifndef DATALOADER_H
#define DATALOADER_H

#include "tokenizer.h"

/**
 * 数据加载器
 *
 * 用于训练数据的加载和批处理
 * 支持:
 * - 从文本文件加载数据
 * - 自动分词
 * - 滑动窗口采样
 * - 随机打乱
 */

typedef struct {
    // 原始数据
    char* raw_data;         // 原始文本
    int raw_len;            // 原始文本长度

    // 编码后的数据
    int* tokens;            // token ID 数组
    int num_tokens;         // token 总数

    // Tokenizer
    Tokenizer* tokenizer;   // tokenizer (不拥有，不负责释放)

    // 批处理参数
    int seq_len;            // 每个样本的序列长度
    int batch_size;         // 批大小
    int current_pos;        // 当前读取位置

    // 随机打乱
    int* shuffle_indices;   // 打乱后的起始位置索引
    int num_samples;        // 总样本数
    int current_sample;     // 当前样本索引
} DataLoader;

/**
 * 创建数据加载器
 * @param filepath 数据文件路径
 * @param tokenizer tokenizer 实例
 * @param seq_len 序列长度
 * @param batch_size 批大小
 * @return 数据加载器实例
 */
DataLoader* dataloader_create(
    const char* filepath,
    Tokenizer* tokenizer,
    int seq_len,
    int batch_size
);

/**
 * 从字符串创建数据加载器 (用于测试)
 */
DataLoader* dataloader_from_string(
    const char* text,
    Tokenizer* tokenizer,
    int seq_len,
    int batch_size
);

/**
 * 释放数据加载器
 */
void dataloader_free(DataLoader* dl);

/**
 * 获取下一个批次
 * @param dl 数据加载器
 * @param input_ids 输入 token IDs [batch_size, seq_len]
 * @param targets 目标 token IDs [batch_size, seq_len] (input_ids 右移一位)
 * @return 实际批大小，0 表示数据用尽
 */
int dataloader_next_batch(DataLoader* dl, int* input_ids, int* targets);

/**
 * 重置数据加载器 (重新开始一个 epoch)
 */
void dataloader_reset(DataLoader* dl);

/**
 * 打乱数据顺序
 */
void dataloader_shuffle(DataLoader* dl);

/**
 * 获取总批次数
 */
int dataloader_num_batches(DataLoader* dl);

/**
 * 获取当前进度
 * @return 0.0 到 1.0 之间的值
 */
float dataloader_progress(DataLoader* dl);

/**
 * 打印数据加载器信息
 */
void dataloader_print_info(DataLoader* dl);

#endif // DATALOADER_H
