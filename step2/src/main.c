#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "embedding.h"

#define EPSILON 1e-5
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

void assert_float_near(float a, float b, float eps, const char* test_name) {
    if (fabsf(a - b) < eps) {
        printf("[%s] %s (%.6f ≈ %.6f)\n", TEST_PASS, test_name, a, b);
        tests_passed++;
    } else {
        printf("[%s] %s (%.6f != %.6f)\n", TEST_FAIL, test_name, a, b);
        tests_failed++;
    }
}

// ============ Embedding 测试 ============

void test_embedding_create() {
    printf("\n=== 测试: Embedding 创建 ===\n");

    int vocab_size = 260;
    int hidden_dim = 64;
    int max_seq_len = 128;

    Embedding* emb = embedding_create(vocab_size, hidden_dim, max_seq_len);
    assert_true(emb != NULL, "创建 Embedding 成功");
    assert_int_eq(emb->vocab_size, vocab_size, "vocab_size 正确");
    assert_int_eq(emb->hidden_dim, hidden_dim, "hidden_dim 正确");
    assert_int_eq(emb->max_seq_len, max_seq_len, "max_seq_len 正确");

    assert_true(emb->token_embedding != NULL, "token_embedding 已创建");
    assert_true(emb->position_embedding != NULL, "position_embedding 已创建");

    assert_int_eq(emb->token_embedding->shape[0], vocab_size, "token_embedding 形状[0] 正确");
    assert_int_eq(emb->token_embedding->shape[1], hidden_dim, "token_embedding 形状[1] 正确");
    assert_int_eq(emb->position_embedding->shape[0], max_seq_len, "position_embedding 形状[0] 正确");
    assert_int_eq(emb->position_embedding->shape[1], hidden_dim, "position_embedding 形状[1] 正确");

    int expected_params = vocab_size * hidden_dim + max_seq_len * hidden_dim;
    assert_int_eq(embedding_num_params(emb), expected_params, "参数数量正确");

    embedding_free(emb);
}

void test_embedding_init_random() {
    printf("\n=== 测试: 随机初始化 ===\n");

    Embedding* emb = embedding_create(100, 32, 64);
    embedding_init_random(emb, 0.02f);

    // 检查不全为零
    float sum = 0.0f;
    for (int i = 0; i < emb->token_embedding->size; i++) {
        sum += fabsf(emb->token_embedding->data[i]);
    }
    assert_true(sum > 0.0f, "token_embedding 不全为零");

    // 检查均值接近 0
    float mean = tensor_mean(emb->token_embedding);
    assert_true(fabsf(mean) < 0.1f, "token_embedding 均值接近 0");

    // 检查标准差接近 0.02
    float std = tensor_std(emb->token_embedding);
    assert_true(fabsf(std - 0.02f) < 0.01f, "token_embedding 标准差接近 0.02");

    embedding_free(emb);
}

void test_embedding_sinusoidal_position() {
    printf("\n=== 测试: 正弦位置编码 ===\n");

    Embedding* emb = embedding_create(100, 64, 128);
    embedding_init_sinusoidal_position(emb);

    // 位置 0 的偶数位应该是 sin(0) = 0
    float pos0_dim0 = emb->position_embedding->data[0];
    assert_float_near(pos0_dim0, 0.0f, 0.001f, "pos=0, dim=0: sin(0) = 0");

    // 位置 0 的奇数位应该是 cos(0) = 1
    float pos0_dim1 = emb->position_embedding->data[1];
    assert_float_near(pos0_dim1, 1.0f, 0.001f, "pos=0, dim=1: cos(0) = 1");

    // 不同位置应该有不同的编码
    float pos0_sum = 0.0f, pos1_sum = 0.0f;
    for (int d = 0; d < emb->hidden_dim; d++) {
        pos0_sum += emb->position_embedding->data[0 * emb->hidden_dim + d];
        pos1_sum += emb->position_embedding->data[1 * emb->hidden_dim + d];
    }
    assert_true(fabsf(pos0_sum - pos1_sum) > 0.1f, "不同位置编码不同");

    // 值应该在 [-1, 1] 范围内
    float max_val = tensor_max(emb->position_embedding);
    float min_val = tensor_min(emb->position_embedding);
    assert_true(max_val <= 1.0f + EPSILON, "位置编码最大值 <= 1");
    assert_true(min_val >= -1.0f - EPSILON, "位置编码最小值 >= -1");

    embedding_free(emb);
}

void test_embedding_forward() {
    printf("\n=== 测试: Embedding 前向传播 ===\n");

    int vocab_size = 260;
    int hidden_dim = 64;
    int max_seq_len = 128;

    Embedding* emb = embedding_create(vocab_size, hidden_dim, max_seq_len);
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    // 测试序列
    int token_ids[] = {76, 109, 112, 112, 115};  // "Hello" 的 token
    int seq_len = 5;

    // 创建输出张量
    int out_shape[] = {seq_len, hidden_dim};
    Tensor* output = tensor_zeros(2, out_shape);

    // 前向传播
    embedding_forward(emb, token_ids, seq_len, output);

    assert_true(output->shape[0] == seq_len, "输出 shape[0] 正确");
    assert_true(output->shape[1] == hidden_dim, "输出 shape[1] 正确");

    // 输出不应全为零
    float sum = 0.0f;
    for (int i = 0; i < output->size; i++) {
        sum += fabsf(output->data[i]);
    }
    assert_true(sum > 0.0f, "输出不全为零");

    // 验证: output = token_emb + pos_emb
    Tensor* token_out = tensor_zeros(2, out_shape);
    Tensor* pos_out = tensor_zeros(2, out_shape);
    embedding_get_token(emb, token_ids, seq_len, token_out);
    embedding_get_position(emb, seq_len, pos_out);

    // 检查 output[0,0] = token_out[0,0] + pos_out[0,0]
    float expected = token_out->data[0] + pos_out->data[0];
    assert_float_near(output->data[0], expected, EPSILON, "output = token + position");

    tensor_free(output);
    tensor_free(token_out);
    tensor_free(pos_out);
    embedding_free(emb);
}

void test_embedding_same_token() {
    printf("\n=== 测试: 相同 Token 嵌入 ===\n");

    Embedding* emb = embedding_create(260, 32, 64);
    embedding_init_random(emb, 0.02f);

    // 相同的 token 在不同位置
    int token_ids1[] = {100, 100};  // 两个相同的 token

    int out_shape[] = {2, 32};
    Tensor* token_out = tensor_zeros(2, out_shape);
    embedding_get_token(emb, token_ids1, 2, token_out);

    // 相同 token 的嵌入应该相同
    int same = 1;
    for (int d = 0; d < 32; d++) {
        if (fabsf(token_out->data[d] - token_out->data[32 + d]) > EPSILON) {
            same = 0;
            break;
        }
    }
    assert_true(same, "相同 token 的嵌入向量相同");

    // 但加上位置编码后应该不同
    embedding_init_sinusoidal_position(emb);
    Tensor* output = tensor_zeros(2, out_shape);
    embedding_forward(emb, token_ids1, 2, output);

    int different = 0;
    for (int d = 0; d < 32; d++) {
        if (fabsf(output->data[d] - output->data[32 + d]) > EPSILON) {
            different = 1;
            break;
        }
    }
    assert_true(different, "加位置编码后，相同 token 在不同位置嵌入不同");

    tensor_free(token_out);
    tensor_free(output);
    embedding_free(emb);
}

void test_embedding_get_vector() {
    printf("\n=== 测试: 获取单个嵌入向量 ===\n");

    Embedding* emb = embedding_create(260, 32, 64);
    embedding_init_random(emb, 0.02f);

    // 获取 token 100 的嵌入向量
    float* vec = embedding_get_vector(emb, 100);
    assert_true(vec != NULL, "获取嵌入向量成功");

    // 验证指向正确位置
    float* expected_ptr = &emb->token_embedding->data[100 * 32];
    assert_true(vec == expected_ptr, "嵌入向量指针正确");

    // 测试边界情况
    float* invalid_vec = embedding_get_vector(emb, -1);
    assert_true(invalid_vec == NULL, "无效 ID 返回 NULL");

    invalid_vec = embedding_get_vector(emb, 300);
    assert_true(invalid_vec == NULL, "超出范围 ID 返回 NULL");

    embedding_free(emb);
}

void test_embedding_with_tokenizer() {
    printf("\n=== 测试: 与 Tokenizer 集成 ===\n");

    // 创建 tokenizer 和 embedding
    Tokenizer* tok = tokenizer_create();
    Embedding* emb = embedding_create(tok->vocab_size, 64, 128);
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    // 编码文本
    const char* text = "Hello";
    int len;
    int* ids = tokenizer_encode(tok, text, &len, 0, 0);

    assert_true(ids != NULL, "文本编码成功");
    assert_int_eq(len, 5, "'Hello' 编码长度为 5");

    // Embedding 前向传播
    int out_shape[] = {len, 64};
    Tensor* output = tensor_zeros(2, out_shape);
    embedding_forward(emb, ids, len, output);

    assert_true(output->shape[0] == len, "输出序列长度正确");
    assert_true(output->shape[1] == 64, "输出隐藏维度正确");

    // 打印一些信息
    printf("  文本: \"%s\"\n", text);
    printf("  Token IDs: ");
    for (int i = 0; i < len; i++) {
        printf("%d ", ids[i]);
    }
    printf("\n");
    printf("  输出形状: [%d, %d]\n", output->shape[0], output->shape[1]);

    free(ids);
    tensor_free(output);
    embedding_free(emb);
    tokenizer_free(tok);

    tests_passed++;  // 集成测试通过
}

void test_embedding_info() {
    printf("\n=== 测试: Embedding 信息 ===\n");

    Embedding* emb = embedding_create(260, 64, 128);
    embedding_init_random(emb, 0.02f);
    embedding_init_sinusoidal_position(emb);

    printf("Embedding 信息输出:\n");
    embedding_print_info(emb);

    embedding_free(emb);
    tests_passed++;
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step2 测试套件\n");
    printf("         Embedding 测试\n");
    printf("========================================\n");

    // 设置随机种子以获得可重复结果
    srand(42);

    test_embedding_create();
    test_embedding_init_random();
    test_embedding_sinusoidal_position();
    test_embedding_forward();
    test_embedding_same_token();
    test_embedding_get_vector();
    test_embedding_with_tokenizer();
    test_embedding_info();

    // 打印总结
    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
