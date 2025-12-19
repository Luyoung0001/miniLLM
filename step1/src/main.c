#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

void assert_true(int condition, const char* test_name) {
    if (condition) {
        printf("[%s] %s\n", TEST_PASS, test_name);
        tests_passed++;
    } else {
        printf("[%s] %s\n", TEST_FAIL, test_name);
        tests_failed++;
    }
}

void assert_int_eq(int a, int b, const char* test_name) {
    if (a == b) {
        printf("[%s] %s (%d == %d)\n", TEST_PASS, test_name, a, b);
        tests_passed++;
    } else {
        printf("[%s] %s (%d != %d)\n", TEST_FAIL, test_name, a, b);
        tests_failed++;
    }
}

void assert_str_eq(const char* a, const char* b, const char* test_name) {
    if (a != NULL && b != NULL && strcmp(a, b) == 0) {
        printf("[%s] %s (\"%s\" == \"%s\")\n", TEST_PASS, test_name, a, b);
        tests_passed++;
    } else {
        printf("[%s] %s (\"%s\" != \"%s\")\n", TEST_FAIL, test_name,
               a ? a : "NULL", b ? b : "NULL");
        tests_failed++;
    }
}

// ============ Tokenizer 测试 ============

void test_tokenizer_create() {
    printf("\n=== 测试: Tokenizer 创建 ===\n");

    Tokenizer* tok = tokenizer_create();
    assert_true(tok != NULL, "创建 tokenizer 成功");
    assert_int_eq(tok->vocab_size, 260, "词汇表大小为 260");
    assert_int_eq(tok->pad_id, 0, "PAD ID = 0");
    assert_int_eq(tok->unk_id, 1, "UNK ID = 1");
    assert_int_eq(tok->bos_id, 2, "BOS ID = 2");
    assert_int_eq(tok->eos_id, 3, "EOS ID = 3");

    tokenizer_free(tok);
}

void test_tokenizer_special_tokens() {
    printf("\n=== 测试: 特殊 Token ===\n");

    Tokenizer* tok = tokenizer_create();

    assert_str_eq(tokenizer_decode_id(tok, 0), "<PAD>", "ID 0 = <PAD>");
    assert_str_eq(tokenizer_decode_id(tok, 1), "<UNK>", "ID 1 = <UNK>");
    assert_str_eq(tokenizer_decode_id(tok, 2), "<BOS>", "ID 2 = <BOS>");
    assert_str_eq(tokenizer_decode_id(tok, 3), "<EOS>", "ID 3 = <EOS>");

    assert_true(tokenizer_is_special(tok, 0), "PAD 是特殊 token");
    assert_true(tokenizer_is_special(tok, 1), "UNK 是特殊 token");
    assert_true(tokenizer_is_special(tok, 2), "BOS 是特殊 token");
    assert_true(tokenizer_is_special(tok, 3), "EOS 是特殊 token");
    assert_true(!tokenizer_is_special(tok, 4), "ID 4 不是特殊 token");

    tokenizer_free(tok);
}

void test_tokenizer_encode_char() {
    printf("\n=== 测试: 字符编码 ===\n");

    Tokenizer* tok = tokenizer_create();

    // 'a' = ASCII 97, ID = 4 + 97 = 101
    assert_int_eq(tokenizer_encode_char(tok, 'a'), 101, "'a' -> ID 101");
    // 'A' = ASCII 65, ID = 4 + 65 = 69
    assert_int_eq(tokenizer_encode_char(tok, 'A'), 69, "'A' -> ID 69");
    // '0' = ASCII 48, ID = 4 + 48 = 52
    assert_int_eq(tokenizer_encode_char(tok, '0'), 52, "'0' -> ID 52");
    // ' ' = ASCII 32, ID = 4 + 32 = 36
    assert_int_eq(tokenizer_encode_char(tok, ' '), 36, "' ' -> ID 36");
    // '\n' = ASCII 10, ID = 4 + 10 = 14
    assert_int_eq(tokenizer_encode_char(tok, '\n'), 14, "'\\n' -> ID 14");

    tokenizer_free(tok);
}

void test_tokenizer_encode_text() {
    printf("\n=== 测试: 文本编码 ===\n");

    Tokenizer* tok = tokenizer_create();

    // 编码 "Hi"
    int len;
    int* ids = tokenizer_encode(tok, "Hi", &len, 0, 0);
    assert_true(ids != NULL, "编码 'Hi' 成功");
    assert_int_eq(len, 2, "'Hi' 长度为 2");
    // 'H' = 72, ID = 76; 'i' = 105, ID = 109
    assert_int_eq(ids[0], 76, "'H' -> ID 76");
    assert_int_eq(ids[1], 109, "'i' -> ID 109");
    free(ids);

    // 带 BOS 和 EOS
    ids = tokenizer_encode(tok, "Hi", &len, 1, 1);
    assert_int_eq(len, 4, "'Hi' + BOS + EOS 长度为 4");
    assert_int_eq(ids[0], TOKEN_BOS, "第一个是 BOS");
    assert_int_eq(ids[len-1], TOKEN_EOS, "最后是 EOS");
    free(ids);

    // 空字符串
    ids = tokenizer_encode(tok, "", &len, 0, 0);
    assert_int_eq(len, 0, "空字符串长度为 0");
    free(ids);

    tokenizer_free(tok);
}

void test_tokenizer_decode() {
    printf("\n=== 测试: 文本解码 ===\n");

    Tokenizer* tok = tokenizer_create();

    // 解码 "Hi"
    int ids1[] = {76, 109};  // 'H', 'i'
    char* text1 = tokenizer_decode(tok, ids1, 2);
    assert_str_eq(text1, "Hi", "解码 [76, 109] = 'Hi'");
    free(text1);

    // 带特殊 token (应该被跳过)
    int ids2[] = {TOKEN_BOS, 76, 109, TOKEN_EOS};
    char* text2 = tokenizer_decode(tok, ids2, 4);
    assert_str_eq(text2, "Hi", "解码跳过特殊 token");
    free(text2);

    // 解码包含换行符
    int ids3[] = {76, 109, 14};  // 'H', 'i', '\n'
    char* text3 = tokenizer_decode(tok, ids3, 3);
    assert_str_eq(text3, "Hi\n", "解码包含换行符");
    free(text3);

    tokenizer_free(tok);
}

void test_tokenizer_roundtrip() {
    printf("\n=== 测试: 编解码往返 ===\n");

    Tokenizer* tok = tokenizer_create();

    // 测试各种字符串
    const char* test_strings[] = {
        "Hello, World!",
        "你好",  // UTF-8 中文 (会被编码为多个字节)
        "123 + 456 = 579",
        "Line1\nLine2\tTab",
        "Special: @#$%^&*()",
        ""
    };

    int num_tests = sizeof(test_strings) / sizeof(test_strings[0]);
    for (int i = 0; i < num_tests; i++) {
        const char* original = test_strings[i];
        int len;
        int* ids = tokenizer_encode(tok, original, &len, 0, 0);
        char* decoded = tokenizer_decode(tok, ids, len);

        char test_name[128];
        snprintf(test_name, sizeof(test_name), "往返测试 #%d", i + 1);

        if (decoded != NULL && strcmp(original, decoded) == 0) {
            printf("[%s] %s: \"%s\"\n", TEST_PASS, test_name, original);
            tests_passed++;
        } else {
            printf("[%s] %s: \"%s\" != \"%s\"\n", TEST_FAIL, test_name,
                   original, decoded ? decoded : "NULL");
            tests_failed++;
        }

        free(ids);
        free(decoded);
    }

    tokenizer_free(tok);
}

void test_tokenizer_save_load() {
    printf("\n=== 测试: 保存和加载 ===\n");

    Tokenizer* tok1 = tokenizer_create();

    // 保存
    const char* vocab_path = "/tmp/test_vocab.txt";
    int save_result = tokenizer_save(tok1, vocab_path);
    assert_true(save_result == 0, "保存词汇表成功");

    // 加载
    Tokenizer* tok2 = tokenizer_load(vocab_path);
    assert_true(tok2 != NULL, "加载词汇表成功");
    assert_int_eq(tok2->vocab_size, tok1->vocab_size, "加载后词汇表大小相同");

    // 验证编解码一致
    const char* test_text = "Test123";
    int len1, len2;
    int* ids1 = tokenizer_encode(tok1, test_text, &len1, 0, 0);
    int* ids2 = tokenizer_encode(tok2, test_text, &len2, 0, 0);

    assert_int_eq(len1, len2, "编码长度一致");

    int ids_match = 1;
    for (int i = 0; i < len1; i++) {
        if (ids1[i] != ids2[i]) {
            ids_match = 0;
            break;
        }
    }
    assert_true(ids_match, "编码结果一致");

    free(ids1);
    free(ids2);
    tokenizer_free(tok1);
    tokenizer_free(tok2);

    // 清理临时文件
    remove(vocab_path);
}

void test_tokenizer_info() {
    printf("\n=== 测试: Tokenizer 信息 ===\n");

    Tokenizer* tok = tokenizer_create();
    printf("Tokenizer 信息输出:\n");
    tokenizer_print_info(tok);

    // 测试 token 打印
    int len;
    int* ids = tokenizer_encode(tok, "Hi!", &len, 1, 1);
    printf("\n编码 'Hi!' (带 BOS/EOS):\n");
    tokenizer_print_tokens(tok, ids, len);

    free(ids);
    tokenizer_free(tok);

    tests_passed++;  // 只要不崩溃就算通过
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step1 测试套件\n");
    printf("         Tokenizer 测试\n");
    printf("========================================\n");

    test_tokenizer_create();
    test_tokenizer_special_tokens();
    test_tokenizer_encode_char();
    test_tokenizer_encode_text();
    test_tokenizer_decode();
    test_tokenizer_roundtrip();
    test_tokenizer_save_load();
    test_tokenizer_info();

    // 打印总结
    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
