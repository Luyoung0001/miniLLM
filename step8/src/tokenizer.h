#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

/**
 * 特殊 token ID
 */
#define TOKEN_PAD 0     // <PAD> 填充
#define TOKEN_UNK 1     // <UNK> 未知
#define TOKEN_BOS 2     // <BOS> 序列开始
#define TOKEN_EOS 3     // <EOS> 序列结束

/**
 * 特殊 token 数量
 */
#define NUM_SPECIAL_TOKENS 4

/**
 * 默认词汇表大小 (特殊token + ASCII 0-255)
 */
#define DEFAULT_VOCAB_SIZE 260

/**
 * Tokenizer - 字符级分词器
 *
 * 使用简单的字符级编码:
 * - ID 0-3: 特殊 token (PAD, UNK, BOS, EOS)
 * - ID 4-259: ASCII 字符 0-255
 */
typedef struct {
    char** id_to_token;     // token ID -> 字符串表示
    int* char_to_id;        // ASCII 字符 -> token ID (大小为 256)
    int vocab_size;         // 词汇表大小
    int pad_id;
    int unk_id;
    int bos_id;
    int eos_id;
} Tokenizer;

// ============ 创建和销毁 ============

/**
 * 创建字符级 tokenizer
 * 默认词汇表: 4 个特殊 token + 256 个 ASCII 字符 = 260
 */
Tokenizer* tokenizer_create(void);

/**
 * 从词汇表文件加载 tokenizer
 * 文件格式: 每行一个 token
 */
Tokenizer* tokenizer_load(const char* vocab_path);

/**
 * 释放 tokenizer
 */
void tokenizer_free(Tokenizer* tok);

/**
 * 保存词汇表到文件
 */
int tokenizer_save(Tokenizer* tok, const char* path);

// ============ 编码和解码 ============

/**
 * 将文本编码为 token ID 序列
 * @param tok tokenizer
 * @param text 输入文本
 * @param out_len 输出: 编码后的长度
 * @param add_bos 是否添加 BOS token
 * @param add_eos 是否添加 EOS token
 * @return token ID 数组 (需要调用者 free)
 */
int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len,
                      int add_bos, int add_eos);

/**
 * 将 token ID 序列解码为文本
 * @param tok tokenizer
 * @param ids token ID 数组
 * @param len 数组长度
 * @return 解码后的字符串 (需要调用者 free)
 */
char* tokenizer_decode(Tokenizer* tok, int* ids, int len);

/**
 * 编码单个字符
 * @return token ID, 未知字符返回 UNK
 */
int tokenizer_encode_char(Tokenizer* tok, char c);

/**
 * 解码单个 token ID
 * @return 对应的字符串 (不需要 free, 指向内部存储)
 */
const char* tokenizer_decode_id(Tokenizer* tok, int id);

// ============ 工具函数 ============

/**
 * 获取词汇表大小
 */
int tokenizer_vocab_size(Tokenizer* tok);

/**
 * 检查是否为特殊 token
 */
int tokenizer_is_special(Tokenizer* tok, int id);

/**
 * 打印 tokenizer 信息
 */
void tokenizer_print_info(Tokenizer* tok);

/**
 * 打印编码结果 (用于调试)
 */
void tokenizer_print_tokens(Tokenizer* tok, int* ids, int len);

#endif // TOKENIZER_H
