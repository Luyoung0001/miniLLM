#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/bpe_tokenizer.h"

int main() {
    printf("=== BPE Tokenizer 测试 ===\n\n");

    // 1. 创建 tokenizer
    printf("1. 创建 BPE tokenizer...\n");
    BPETokenizer* tok = bpe_tokenizer_create();
    if (tok == NULL) {
        printf("创建失败!\n");
        return 1;
    }
    printf("   初始词汇表大小: %d\n\n", bpe_vocab_size(tok));

    // 2. 测试训练数据
    const char* train_text =
        "hello world hello world hello world "
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox jumps over the lazy dog "
        "how are you how are you how are you "
        "good morning good morning good morning "
        "hello hello hello world world world "
        "this is a test this is a test this is a test ";

    // 3. 训练 BPE
    printf("2. 训练 BPE (500 次合并)...\n");
    int result = bpe_train(tok, train_text, 500);
    if (result != 0) {
        printf("训练失败!\n");
        bpe_tokenizer_free(tok);
        return 1;
    }
    printf("\n");

    // 4. 打印统计
    printf("3. Tokenizer 统计:\n");
    bpe_print_stats(tok);
    printf("\n");

    // 5. 测试编码
    printf("4. 测试编码:\n");
    const char* test_texts[] = {
        "hello world",
        "the quick brown fox",
        "how are you",
        "good morning",
        "this is a test"
    };

    for (int i = 0; i < 5; i++) {
        int len;
        int* tokens = bpe_encode(tok, test_texts[i], &len, 0, 0);
        printf("   '%s' -> %d tokens: ", test_texts[i], len);
        bpe_print_tokens(tok, tokens, len);

        // 测试解码
        char* decoded = bpe_decode(tok, tokens, len);
        printf("   解码: '%s'\n", decoded);

        // 验证
        if (strcmp(decoded, test_texts[i]) == 0) {
            printf("   ✓ 编码/解码正确!\n\n");
        } else {
            printf("   ✗ 编码/解码不匹配!\n\n");
        }

        free(tokens);
        free(decoded);
    }

    // 6. 测试保存/加载
    printf("5. 测试保存词汇表...\n");
    result = bpe_save_vocab(tok, "bpe_vocab.txt");
    if (result == 0) {
        printf("   保存成功: bpe_vocab.txt\n");
    } else {
        printf("   保存失败!\n");
    }

    // 创建新 tokenizer 并加载
    printf("6. 测试加载词汇表...\n");
    BPETokenizer* tok2 = bpe_tokenizer_create();
    result = bpe_load_vocab(tok2, "bpe_vocab.txt");
    if (result == 0) {
        printf("   加载成功!\n");
        printf("   词汇表大小: %d\n", bpe_vocab_size(tok2));

        // 验证加载后的编码
        int len1, len2;
        int* tokens1 = bpe_encode(tok, "hello world", &len1, 0, 0);
        int* tokens2 = bpe_encode(tok2, "hello world", &len2, 0, 0);

        if (len1 == len2) {
            int match = 1;
            for (int i = 0; i < len1; i++) {
                if (tokens1[i] != tokens2[i]) match = 0;
            }
            if (match) {
                printf("   ✓ 加载后编码结果一致!\n");
            } else {
                printf("   ✗ 加载后编码结果不一致!\n");
            }
        } else {
            printf("   ✗ 加载后 token 数量不一致!\n");
        }

        free(tokens1);
        free(tokens2);
    } else {
        printf("   加载失败!\n");
    }

    // 7. 对比字节级分词
    printf("\n7. 效率对比 (BPE vs 字节级):\n");
    const char* sample = "hello world how are you";
    int bpe_len;
    int* bpe_tokens = bpe_encode(tok, sample, &bpe_len, 0, 0);
    int byte_len = strlen(sample);

    printf("   文本: '%s'\n", sample);
    printf("   字节级: %d tokens (每个字符一个)\n", byte_len);
    printf("   BPE:    %d tokens\n", bpe_len);
    printf("   压缩比: %.1fx\n", (float)byte_len / bpe_len);

    free(bpe_tokens);

    // 清理
    bpe_tokenizer_free(tok);
    bpe_tokenizer_free(tok2);

    printf("\n=== BPE Tokenizer 测试完成 ===\n");
    return 0;
}
