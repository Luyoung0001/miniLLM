#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"

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

// ============ Attention 测试 ============

void test_attention_create() {
    printf("\n=== 测试: Attention 创建 ===\n");

    int hidden_dim = 64;
    int num_heads = 4;

    MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
    assert_true(attn != NULL, "创建 Attention 成功");
    assert_int_eq(attn->hidden_dim, hidden_dim, "hidden_dim 正确");
    assert_int_eq(attn->num_heads, num_heads, "num_heads 正确");
    assert_int_eq(attn->head_dim, 16, "head_dim = 64/4 = 16");

    float expected_scale = 1.0f / sqrtf(16.0f);
    assert_float_near(attn->scale, expected_scale, 0.001f, "scale = 1/sqrt(16)");

    assert_true(attn->W_q != NULL, "W_q 已创建");
    assert_true(attn->W_k != NULL, "W_k 已创建");
    assert_true(attn->W_v != NULL, "W_v 已创建");
    assert_true(attn->W_o != NULL, "W_o 已创建");

    int expected_params = 4 * hidden_dim * hidden_dim;
    assert_int_eq(attention_num_params(attn), expected_params, "参数数量正确");

    attention_free(attn);

    // 测试无效配置
    MultiHeadAttention* invalid = attention_create(64, 5);  // 64 不能被 5 整除
    assert_true(invalid == NULL, "无效配置返回 NULL");
}

void test_attention_cache() {
    printf("\n=== 测试: Attention Cache ===\n");

    int seq_len = 10;
    int hidden_dim = 64;
    int num_heads = 4;

    AttentionCache* cache = attention_cache_create(seq_len, hidden_dim, num_heads);
    assert_true(cache != NULL, "创建 Cache 成功");
    assert_int_eq(cache->seq_len, seq_len, "seq_len 正确");

    assert_true(cache->Q != NULL, "Q 已创建");
    assert_int_eq(cache->Q->shape[0], seq_len, "Q shape[0] 正确");
    assert_int_eq(cache->Q->shape[1], hidden_dim, "Q shape[1] 正确");

    assert_true(cache->scores != NULL, "scores 已创建");
    assert_int_eq(cache->scores->shape[0], num_heads, "scores shape[0] = num_heads");
    assert_int_eq(cache->scores->shape[1], seq_len, "scores shape[1] = seq_len");
    assert_int_eq(cache->scores->shape[2], seq_len, "scores shape[2] = seq_len");

    // 测试 resize
    int result = attention_cache_resize(cache, 20);
    assert_true(result == 0, "Cache resize 成功");
    assert_int_eq(cache->seq_len, 20, "resize 后 seq_len = 20");
    assert_int_eq(cache->Q->shape[0], 20, "resize 后 Q shape[0] = 20");

    attention_cache_free(cache);
}

void test_causal_mask() {
    printf("\n=== 测试: 因果掩码 ===\n");

    int seq_len = 4;
    Tensor* mask = create_causal_mask(seq_len);

    assert_true(mask != NULL, "创建掩码成功");
    assert_int_eq(mask->shape[0], seq_len, "掩码 shape[0] 正确");
    assert_int_eq(mask->shape[1], seq_len, "掩码 shape[1] 正确");

    // 验证掩码值
    // 期望:
    // [0,    -inf, -inf, -inf]
    // [0,    0,    -inf, -inf]
    // [0,    0,    0,    -inf]
    // [0,    0,    0,    0   ]

    assert_float_near(mask->data[0], 0.0f, EPSILON, "mask[0,0] = 0");
    assert_true(mask->data[1] < -1e8f, "mask[0,1] = -inf");
    assert_true(mask->data[2] < -1e8f, "mask[0,2] = -inf");
    assert_true(mask->data[3] < -1e8f, "mask[0,3] = -inf");

    assert_float_near(mask->data[4], 0.0f, EPSILON, "mask[1,0] = 0");
    assert_float_near(mask->data[5], 0.0f, EPSILON, "mask[1,1] = 0");
    assert_true(mask->data[6] < -1e8f, "mask[1,2] = -inf");

    assert_float_near(mask->data[15], 0.0f, EPSILON, "mask[3,3] = 0");

    tensor_free(mask);
}

void test_single_head_attention() {
    printf("\n=== 测试: 单头注意力 ===\n");

    int seq_len = 3;
    int head_dim = 4;

    // 创建简单的 Q, K, V
    int shape[] = {seq_len, head_dim};
    Tensor* Q = tensor_zeros(2, shape);
    Tensor* K = tensor_zeros(2, shape);
    Tensor* V = tensor_zeros(2, shape);
    Tensor* output = tensor_zeros(2, shape);

    // 设置简单值: 单位矩阵风格
    // Q = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    // K = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    // V = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    Q->data[0] = 1.0f; Q->data[5] = 1.0f; Q->data[10] = 1.0f;
    K->data[0] = 1.0f; K->data[5] = 1.0f; K->data[10] = 1.0f;

    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            V->data[i * head_dim + d] = (float)(i * head_dim + d + 1);
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    single_head_attention(Q, K, V, NULL, scale, output);

    // 输出不应全为零
    float sum = 0.0f;
    for (int i = 0; i < output->size; i++) {
        sum += fabsf(output->data[i]);
    }
    assert_true(sum > 0.0f, "单头注意力输出不为零");

    // 带因果掩码测试
    Tensor* mask = create_causal_mask(seq_len);
    Tensor* output_masked = tensor_zeros(2, shape);
    single_head_attention(Q, K, V, mask, scale, output_masked);

    // 第一个位置只能看自己，应该更接近 V[0]
    assert_true(fabsf(output_masked->data[0] - V->data[0]) <
                fabsf(output->data[0] - V->data[0]) + 0.1f,
                "因果掩码使第一位置更关注自己");

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(output);
    tensor_free(output_masked);
    tensor_free(mask);
}

void test_attention_forward() {
    printf("\n=== 测试: 多头注意力前向传播 ===\n");

    int hidden_dim = 32;
    int num_heads = 4;
    int seq_len = 5;

    // 创建注意力层
    MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
    attention_init_random(attn, 0.02f);

    // 创建缓存
    AttentionCache* cache = attention_cache_create(seq_len, hidden_dim, num_heads);

    // 创建输入
    int in_shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, in_shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, in_shape);

    // 无掩码前向传播
    attention_forward(attn, input, NULL, cache, output);

    assert_int_eq(output->shape[0], seq_len, "输出 shape[0] 正确");
    assert_int_eq(output->shape[1], hidden_dim, "输出 shape[1] 正确");

    // 输出不应全为零
    float sum = 0.0f;
    for (int i = 0; i < output->size; i++) {
        sum += fabsf(output->data[i]);
    }
    assert_true(sum > 0.0f, "输出不全为零");

    // 带因果掩码
    Tensor* mask = create_causal_mask(seq_len);
    Tensor* output_masked = tensor_zeros(2, in_shape);
    attention_forward(attn, input, mask, cache, output_masked);

    // 掩码后输出应该不同
    int different = 0;
    for (int i = 0; i < output->size; i++) {
        if (fabsf(output->data[i] - output_masked->data[i]) > EPSILON) {
            different = 1;
            break;
        }
    }
    assert_true(different, "有无掩码输出不同");

    tensor_free(input);
    tensor_free(output);
    tensor_free(output_masked);
    tensor_free(mask);
    attention_cache_free(cache);
    attention_free(attn);
}

void test_attention_weights() {
    printf("\n=== 测试: 注意力权重 ===\n");

    int hidden_dim = 16;
    int num_heads = 2;
    int seq_len = 4;

    MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
    attention_init_random(attn, 0.02f);

    AttentionCache* cache = attention_cache_create(seq_len, hidden_dim, num_heads);

    int in_shape[] = {seq_len, hidden_dim};
    Tensor* input = tensor_randn(2, in_shape, 0.0f, 1.0f);
    Tensor* output = tensor_zeros(2, in_shape);

    Tensor* mask = create_causal_mask(seq_len);
    attention_forward(attn, input, mask, cache, output);

    // 获取注意力权重
    Tensor* weights = attention_get_weights(cache);
    assert_true(weights != NULL, "获取注意力权重成功");
    assert_int_eq(weights->shape[0], num_heads, "权重 shape[0] = num_heads");
    assert_int_eq(weights->shape[1], seq_len, "权重 shape[1] = seq_len");
    assert_int_eq(weights->shape[2], seq_len, "权重 shape[2] = seq_len");

    // 验证每行和为 1
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                row_sum += weights->data[h * seq_len * seq_len + i * seq_len + j];
            }
            if (fabsf(row_sum - 1.0f) > EPSILON) {
                printf("[%s] head=%d, row=%d: sum=%.6f != 1.0\n",
                       TEST_FAIL, h, i, row_sum);
                tests_failed++;
                goto end_weight_test;
            }
        }
    }
    printf("[%s] 所有注意力权重行和为 1\n", TEST_PASS);
    tests_passed++;

    // 验证因果掩码效果: 上三角应该为 0
    int causal_ok = 1;
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) {
                float val = weights->data[h * seq_len * seq_len + i * seq_len + j];
                if (val > EPSILON) {
                    causal_ok = 0;
                    break;
                }
            }
        }
    }
    assert_true(causal_ok, "因果掩码生效 (上三角为 0)");

end_weight_test:
    tensor_free(input);
    tensor_free(output);
    tensor_free(mask);
    attention_cache_free(cache);
    attention_free(attn);
}

void test_attention_different_seq_len() {
    printf("\n=== 测试: 不同序列长度 ===\n");

    int hidden_dim = 32;
    int num_heads = 4;

    MultiHeadAttention* attn = attention_create(hidden_dim, num_heads);
    attention_init_random(attn, 0.02f);

    AttentionCache* cache = attention_cache_create(5, hidden_dim, num_heads);

    // 测试多个序列长度
    int seq_lens[] = {1, 3, 5, 10};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lens[t];

        int in_shape[] = {seq_len, hidden_dim};
        Tensor* input = tensor_randn(2, in_shape, 0.0f, 1.0f);
        Tensor* output = tensor_zeros(2, in_shape);
        Tensor* mask = create_causal_mask(seq_len);

        attention_forward(attn, input, mask, cache, output);

        char test_name[64];
        snprintf(test_name, sizeof(test_name), "seq_len=%d 前向传播成功", seq_len);
        assert_true(output->shape[0] == seq_len, test_name);

        tensor_free(input);
        tensor_free(output);
        tensor_free(mask);
    }

    attention_cache_free(cache);
    attention_free(attn);
}

void test_attention_info() {
    printf("\n=== 测试: Attention 信息 ===\n");

    MultiHeadAttention* attn = attention_create(64, 4);
    attention_init_random(attn, 0.02f);

    printf("Attention 信息输出:\n");
    attention_print_info(attn);

    attention_free(attn);
    tests_passed++;
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step3 测试套件\n");
    printf("       Attention 注意力测试\n");
    printf("========================================\n");

    srand(42);

    test_attention_create();
    test_attention_cache();
    test_causal_mask();
    test_single_head_attention();
    test_attention_forward();
    test_attention_weights();
    test_attention_different_seq_len();
    test_attention_info();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
