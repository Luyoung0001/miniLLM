#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// 特殊 token 字符串
static const char* SPECIAL_TOKENS[] = {
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
};

// 创建字符的字符串表示
static char* char_to_string(unsigned char c) {
    char* str;
    if (c >= 32 && c < 127 && c != '\\') {
        // 可打印字符
        str = (char*)malloc(2);
        str[0] = c;
        str[1] = '\0';
    } else {
        // 不可打印字符，使用转义序列
        str = (char*)malloc(8);
        switch (c) {
            case '\n': strcpy(str, "\\n"); break;
            case '\r': strcpy(str, "\\r"); break;
            case '\t': strcpy(str, "\\t"); break;
            case '\\': strcpy(str, "\\\\"); break;
            case '\0': strcpy(str, "\\0"); break;
            default:
                sprintf(str, "\\x%02x", c);
        }
    }
    return str;
}

// 从字符串表示解析字符
static int string_to_char(const char* str, unsigned char* out) {
    if (str[0] != '\\') {
        *out = (unsigned char)str[0];
        return 1;
    }

    // 转义序列
    if (str[1] == 'n') { *out = '\n'; return 1; }
    if (str[1] == 'r') { *out = '\r'; return 1; }
    if (str[1] == 't') { *out = '\t'; return 1; }
    if (str[1] == '\\') { *out = '\\'; return 1; }
    if (str[1] == '0') { *out = '\0'; return 1; }
    if (str[1] == 'x' && strlen(str) >= 4) {
        int val;
        if (sscanf(str + 2, "%2x", &val) == 1) {
            *out = (unsigned char)val;
            return 1;
        }
    }

    return 0;  // 解析失败
}

Tokenizer* tokenizer_create(void) {
    Tokenizer* tok = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (tok == NULL) return NULL;

    tok->vocab_size = DEFAULT_VOCAB_SIZE;
    tok->pad_id = TOKEN_PAD;
    tok->unk_id = TOKEN_UNK;
    tok->bos_id = TOKEN_BOS;
    tok->eos_id = TOKEN_EOS;

    // 分配 id_to_token
    tok->id_to_token = (char**)malloc(tok->vocab_size * sizeof(char*));
    if (tok->id_to_token == NULL) {
        free(tok);
        return NULL;
    }

    // 分配 char_to_id
    tok->char_to_id = (int*)malloc(256 * sizeof(int));
    if (tok->char_to_id == NULL) {
        free(tok->id_to_token);
        free(tok);
        return NULL;
    }

    // 初始化所有字符映射为 UNK
    for (int i = 0; i < 256; i++) {
        tok->char_to_id[i] = TOKEN_UNK;
    }

    // 设置特殊 token
    for (int i = 0; i < NUM_SPECIAL_TOKENS; i++) {
        tok->id_to_token[i] = strdup(SPECIAL_TOKENS[i]);
    }

    // 设置 ASCII 字符
    for (int i = 0; i < 256; i++) {
        int id = NUM_SPECIAL_TOKENS + i;  // ID 4-259
        tok->id_to_token[id] = char_to_string((unsigned char)i);
        tok->char_to_id[i] = id;
    }

    return tok;
}

Tokenizer* tokenizer_load(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (f == NULL) {
        fprintf(stderr, "Error: cannot open vocab file '%s'\n", vocab_path);
        return NULL;
    }

    // 先计算词汇表大小
    int vocab_size = 0;
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL) {
        vocab_size++;
    }
    rewind(f);

    if (vocab_size < NUM_SPECIAL_TOKENS) {
        fprintf(stderr, "Error: vocab file too small\n");
        fclose(f);
        return NULL;
    }

    // 创建 tokenizer
    Tokenizer* tok = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (tok == NULL) {
        fclose(f);
        return NULL;
    }

    tok->vocab_size = vocab_size;
    tok->pad_id = TOKEN_PAD;
    tok->unk_id = TOKEN_UNK;
    tok->bos_id = TOKEN_BOS;
    tok->eos_id = TOKEN_EOS;

    tok->id_to_token = (char**)malloc(vocab_size * sizeof(char*));
    tok->char_to_id = (int*)malloc(256 * sizeof(int));

    if (tok->id_to_token == NULL || tok->char_to_id == NULL) {
        if (tok->id_to_token) free(tok->id_to_token);
        if (tok->char_to_id) free(tok->char_to_id);
        free(tok);
        fclose(f);
        return NULL;
    }

    // 初始化字符映射为 UNK
    for (int i = 0; i < 256; i++) {
        tok->char_to_id[i] = TOKEN_UNK;
    }

    // 读取词汇表
    int id = 0;
    while (fgets(line, sizeof(line), f) != NULL && id < vocab_size) {
        // 移除换行符
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }

        tok->id_to_token[id] = strdup(line);

        // 如果是单字符 token，建立反向映射
        if (id >= NUM_SPECIAL_TOKENS) {
            unsigned char c;
            if (string_to_char(line, &c)) {
                tok->char_to_id[c] = id;
            }
        }

        id++;
    }

    fclose(f);
    return tok;
}

void tokenizer_free(Tokenizer* tok) {
    if (tok == NULL) return;

    if (tok->id_to_token != NULL) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->id_to_token[i] != NULL) {
                free(tok->id_to_token[i]);
            }
        }
        free(tok->id_to_token);
    }

    if (tok->char_to_id != NULL) {
        free(tok->char_to_id);
    }

    free(tok);
}

int tokenizer_save(Tokenizer* tok, const char* path) {
    if (tok == NULL || path == NULL) return -1;

    FILE* f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "Error: cannot create file '%s'\n", path);
        return -1;
    }

    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(f, "%s\n", tok->id_to_token[i]);
    }

    fclose(f);
    return 0;
}

int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len,
                      int add_bos, int add_eos) {
    if (tok == NULL || text == NULL || out_len == NULL) return NULL;

    int text_len = strlen(text);
    int max_len = text_len + (add_bos ? 1 : 0) + (add_eos ? 1 : 0);

    int* ids = (int*)malloc(max_len * sizeof(int));
    if (ids == NULL) return NULL;

    int pos = 0;

    // 添加 BOS
    if (add_bos) {
        ids[pos++] = tok->bos_id;
    }

    // 编码每个字符
    for (int i = 0; i < text_len; i++) {
        unsigned char c = (unsigned char)text[i];
        ids[pos++] = tok->char_to_id[c];
    }

    // 添加 EOS
    if (add_eos) {
        ids[pos++] = tok->eos_id;
    }

    *out_len = pos;
    return ids;
}

char* tokenizer_decode(Tokenizer* tok, int* ids, int len) {
    if (tok == NULL) return NULL;

    // 空序列返回空字符串
    if (ids == NULL || len <= 0) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    // 估计输出长度 (每个 token 最多 8 字节)
    int max_out_len = len * 8 + 1;
    char* output = (char*)malloc(max_out_len);
    if (output == NULL) return NULL;

    int pos = 0;
    for (int i = 0; i < len; i++) {
        int id = ids[i];

        // 跳过特殊 token (除了可能想保留的)
        if (id == tok->pad_id || id == tok->bos_id || id == tok->eos_id) {
            continue;
        }

        if (id == tok->unk_id) {
            // UNK token 用特殊字符表示
            output[pos++] = '\xEF';  // UTF-8 replacement char
            output[pos++] = '\xBF';
            output[pos++] = '\xBD';
            continue;
        }

        if (id < 0 || id >= tok->vocab_size) {
            continue;  // 无效 ID
        }

        // 解码 token
        const char* token_str = tok->id_to_token[id];
        if (token_str == NULL) continue;

        // 如果是字符 token (ID >= 4)
        if (id >= NUM_SPECIAL_TOKENS) {
            unsigned char c;
            if (string_to_char(token_str, &c)) {
                output[pos++] = c;
            }
        }
    }

    output[pos] = '\0';

    // 缩小到实际大小
    char* result = (char*)realloc(output, pos + 1);
    return result ? result : output;
}

int tokenizer_encode_char(Tokenizer* tok, char c) {
    if (tok == NULL) return TOKEN_UNK;
    return tok->char_to_id[(unsigned char)c];
}

const char* tokenizer_decode_id(Tokenizer* tok, int id) {
    if (tok == NULL || id < 0 || id >= tok->vocab_size) {
        return "<INVALID>";
    }
    return tok->id_to_token[id];
}

int tokenizer_vocab_size(Tokenizer* tok) {
    if (tok == NULL) return 0;
    return tok->vocab_size;
}

int tokenizer_is_special(Tokenizer* tok, int id) {
    if (tok == NULL) return 0;
    return id == tok->pad_id || id == tok->unk_id ||
           id == tok->bos_id || id == tok->eos_id;
}

void tokenizer_print_info(Tokenizer* tok) {
    if (tok == NULL) {
        printf("Tokenizer: NULL\n");
        return;
    }

    printf("Tokenizer Info:\n");
    printf("  vocab_size: %d\n", tok->vocab_size);
    printf("  pad_id: %d (%s)\n", tok->pad_id, tok->id_to_token[tok->pad_id]);
    printf("  unk_id: %d (%s)\n", tok->unk_id, tok->id_to_token[tok->unk_id]);
    printf("  bos_id: %d (%s)\n", tok->bos_id, tok->id_to_token[tok->bos_id]);
    printf("  eos_id: %d (%s)\n", tok->eos_id, tok->id_to_token[tok->eos_id]);
}

void tokenizer_print_tokens(Tokenizer* tok, int* ids, int len) {
    if (tok == NULL || ids == NULL) return;

    printf("Tokens [%d]: ", len);
    for (int i = 0; i < len; i++) {
        int id = ids[i];
        const char* str = tokenizer_decode_id(tok, id);
        printf("%d('%s')", id, str);
        if (i < len - 1) printf(", ");
    }
    printf("\n");
}
