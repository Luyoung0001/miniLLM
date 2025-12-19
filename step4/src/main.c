#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "layernorm.h"
#include "ffn.h"
#include "attention.h"
#include "transformer.h"

#define EPSILON 1e-4
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

// ============ LayerNorm 测试 ============

void test_layernorm_create() {
    printf("\n=== 测试: LayerNorm 创建 ===\n");

    int hidden_dim = 64;
    LayerNorm* ln = layernorm_create(hidden_dim, 1e-5f);

    assert_true(ln != NULL, "创建 LayerNorm 成功");
    assert_int_eq(ln->hidden_dim, hidden_dim, "hidden_dim 正确");
    assert_true(ln->gamma != NULL, "gamma 已创建");
    assert_true(ln->beta != NULL, "beta 已创建");

    // gamma 应该是 1
    assert_float_near(ln->gamma->data[0], 1.0f, EPSILON, "gamma 初始化为 1");
    // beta 应该是 0
    assert_float_near(ln->beta->data[0], 0.0f, EPSILON, "beta 初始化为 0");

    assert_int_eq(layernorm_num_params(ln), 2 * hidden_dim, "参数数量正确");

    layernorm_free(ln);
}

void test_layernorm_forward() {
    printf("\n=== 测试: LayerNorm 前向传播 ===\n");

    int hidden_dim = 4;
    LayerNorm* ln = layernorm_create(hidden_dim, 1e-5f);

    // 1D 输入测试
    int shape1d[] = {hidden_dim};
    Tensor* input1d = tensor_create(1, shape1d);
    float data1d[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_fill_data(input1d, data1d);

    Tensor* output1d = tensor_zeros(1, shape1d);
    layernorm_forward(ln, input1d, output1d);

    // 归一化后均值应接近 0
    float mean = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        mean += output1d->data[i];
    }
    mean /= hidden_dim;
    assert_float_near(mean, 0.0f, 0.01f, "1D LayerNorm 输出均值接近 0");

    // 归一化后方差应接近 1
    float var = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        float diff = output1d->data[i] - mean;
        var += diff * diff;
    }
    var /= hidden_dim;
    assert_float_near(var, 1.0f, 0.01f, "1D LayerNorm 输出方差接近 1");

    tensor_free(input1d);
    tensor_free(output1d);

    // 2D 输入测试
    int seq_len = 3;
    int shape2d[] = {seq_len, hidden_dim};
    Tensor* input2d = tensor_randn(2, shape2d, 5.0f, 2.0f);  // 非零均值
    Tensor* output2d = tensor_zeros(2, shape2d);

    layernorm_forward(ln, input2d, output2d);

    // 每行均值应接近 0
    int mean_ok = 1;
    for (int s = 0; s < seq_len; s++) {
        float row_mean = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            row_mean += output2d->data[s * hidden_dim + d];
        }
        row_mean /= hidden_dim;
        if (fabsf(row_mean) > 0.01f) {
            mean_ok = 0;
            break;
        }
    }
    assert_true(mean_ok, "2D LayerNorm 每行均值接近 0");

    tensor_free(input2d);
    tensor_free(output2d);
    layernorm_free(ln);
}

// ============ FFN 测试 ============

void test_ffn_create() {
    printf("\n=== 测试: FFN 创建 ===\n");

    int hidden_dim = 64;
    int ffn_dim = 256;

    FFN* ffn = ffn_create(hidden_dim, ffn_dim);
    assert_true(ffn != NULL, "创建 FFN 成功");
    assert_int_eq(ffn->hidden_dim, hidden_dim, "hidden_dim 正确");
    assert_int_eq(ffn->ffn_dim, ffn_dim, "ffn_dim 正确");

    assert_true(ffn->W1 != NULL, "W1 已创建");
    assert_true(ffn->b1 != NULL, "b1 已创建");
    assert_true(ffn->W2 != NULL, "W2 已创建");
    assert_true(ffn->b2 != NULL, "b2 已创建");

    int expected_params = hidden_dim * ffn_dim + ffn_dim + ffn_dim * hidden_dim + hidden_dim;
    assert_int_eq(ffn_num_params(ffn), expected_params, "参数数量正确");

    ffn_free(ffn);
}

void test_ffn_forward() {
    printf("\n=== 测试: FFN 前向传播 ===\n");

    int hidden_dim = 32;
    int ffn_dim = 128;
    int seq_len = 5;

    FFN* ffn = ffn_create(hidden_dim, ffn_dim);
    ffn_init_random(ffn, 0.02f);

    FFNCache* cache = ffn_cache_create(seq_len, ffn_dim);

    int shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);

    ffn_forward(ffn, input, cache, output);

    assert_int_eq(output->shape[0], seq_len, "输出 shape[0] 正确");
    assert_int_eq(output->shape[1], hidden_dim, "输出 shape[1] 正确");

    // 输出不应全为零
    float sum = 0.0f;
    for (int i = 0; i < output->size; i++) {
        sum += fabsf(output->data[i]);
    }
    assert_true(sum > 0.0f, "FFN 输出不全为零");

    tensor_free(input);
    tensor_free(output);
    ffn_cache_free(cache);
    ffn_free(ffn);
}

// ============ TransformerBlock 测试 ============

void test_transformer_block_create() {
    printf("\n=== 测试: TransformerBlock 创建 ===\n");

    int hidden_dim = 64;
    int num_heads = 4;
    int ffn_dim = 256;

    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    assert_true(block != NULL, "创建 TransformerBlock 成功");
    assert_int_eq(block->hidden_dim, hidden_dim, "hidden_dim 正确");
    assert_int_eq(block->num_heads, num_heads, "num_heads 正确");
    assert_int_eq(block->ffn_dim, ffn_dim, "ffn_dim 正确");

    assert_true(block->ln1 != NULL, "ln1 已创建");
    assert_true(block->attn != NULL, "attn 已创建");
    assert_true(block->ln2 != NULL, "ln2 已创建");
    assert_true(block->ffn != NULL, "ffn 已创建");

    // 计算预期参数数量
    int ln_params = 2 * hidden_dim;  // gamma + beta
    int attn_params = 4 * hidden_dim * hidden_dim;  // W_q, W_k, W_v, W_o
    int ffn_params = hidden_dim * ffn_dim + ffn_dim + ffn_dim * hidden_dim + hidden_dim;
    int expected = 2 * ln_params + attn_params + ffn_params;

    assert_int_eq(transformer_block_num_params(block), expected, "参数数量正确");

    transformer_block_free(block);
}

void test_transformer_block_forward() {
    printf("\n=== 测试: TransformerBlock 前向传播 ===\n");

    int hidden_dim = 32;
    int num_heads = 4;
    int ffn_dim = 128;
    int seq_len = 5;

    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    transformer_block_init(block, 0.02f);

    TransformerCache* cache = transformer_cache_create(seq_len, hidden_dim, num_heads, ffn_dim);

    int shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);
    Tensor* mask = create_causal_mask(seq_len);

    transformer_block_forward(block, input, mask, cache, output);

    assert_int_eq(output->shape[0], seq_len, "输出 shape[0] 正确");
    assert_int_eq(output->shape[1], hidden_dim, "输出 shape[1] 正确");

    // 输出不应全为零
    float sum = 0.0f;
    for (int i = 0; i < output->size; i++) {
        sum += fabsf(output->data[i]);
    }
    assert_true(sum > 0.0f, "TransformerBlock 输出不全为零");

    // 输出应该与输入不同 (因为有残差连接，不会完全相同)
    int different = 0;
    for (int i = 0; i < input->size; i++) {
        if (fabsf(input->data[i] - output->data[i]) > EPSILON) {
            different = 1;
            break;
        }
    }
    assert_true(different, "输出与输入不同");

    tensor_free(input);
    tensor_free(output);
    tensor_free(mask);
    transformer_cache_free(cache);
    transformer_block_free(block);
}

void test_transformer_residual() {
    printf("\n=== 测试: 残差连接 ===\n");

    int hidden_dim = 16;
    int num_heads = 2;
    int ffn_dim = 64;
    int seq_len = 3;

    // 创建一个初始化权重很小的 block
    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    transformer_block_init(block, 0.001f);  // 很小的权重

    TransformerCache* cache = transformer_cache_create(seq_len, hidden_dim, num_heads, ffn_dim);

    int shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, shape);

    transformer_block_forward(block, input, NULL, cache, output);

    // 由于权重很小，输出应该接近输入 (残差连接的作用)
    float diff_sum = 0.0f;
    for (int i = 0; i < input->size; i++) {
        diff_sum += fabsf(input->data[i] - output->data[i]);
    }
    float avg_diff = diff_sum / input->size;
    assert_true(avg_diff < 1.0f, "小权重时输出接近输入 (残差连接有效)");

    tensor_free(input);
    tensor_free(output);
    transformer_cache_free(cache);
    transformer_block_free(block);
}

void test_transformer_different_seq_len() {
    printf("\n=== 测试: 不同序列长度 ===\n");

    int hidden_dim = 32;
    int num_heads = 4;
    int ffn_dim = 128;

    TransformerBlock* block = transformer_block_create(hidden_dim, num_heads, ffn_dim);
    transformer_block_init(block, 0.02f);

    TransformerCache* cache = transformer_cache_create(1, hidden_dim, num_heads, ffn_dim);

    int seq_lens[] = {1, 3, 5, 10};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lens[t];

        int shape[] = {seq_len, hidden_dim};
        Tensor* input = tensor_randn(2, shape, 0.0f, 1.0f);
        Tensor* output = tensor_zeros(2, shape);
        Tensor* mask = create_causal_mask(seq_len);

        transformer_block_forward(block, input, mask, cache, output);

        char test_name[64];
        snprintf(test_name, sizeof(test_name), "seq_len=%d 前向传播成功", seq_len);
        assert_true(output->shape[0] == seq_len, test_name);

        tensor_free(input);
        tensor_free(output);
        tensor_free(mask);
    }

    transformer_cache_free(cache);
    transformer_block_free(block);
}

void test_transformer_info() {
    printf("\n=== 测试: TransformerBlock 信息 ===\n");

    TransformerBlock* block = transformer_block_create(64, 4, 256);
    transformer_block_init(block, 0.02f);

    printf("TransformerBlock 信息输出:\n");
    transformer_block_print_info(block);

    transformer_block_free(block);
    tests_passed++;
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step4 测试套件\n");
    printf("      Transformer Block 测试\n");
    printf("========================================\n");

    srand(42);

    // LayerNorm 测试
    test_layernorm_create();
    test_layernorm_forward();

    // FFN 测试
    test_ffn_create();
    test_ffn_forward();

    // TransformerBlock 测试
    test_transformer_block_create();
    test_transformer_block_forward();
    test_transformer_residual();
    test_transformer_different_seq_len();
    test_transformer_info();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
