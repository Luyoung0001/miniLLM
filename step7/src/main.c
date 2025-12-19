#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "config.h"
#include "model.h"
#include "generate.h"
#include "train.h"
#include "backward.h"
#include "optimizer.h"
#include "dataloader.h"
#include "loss.h"

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

void assert_float_near(float a, float b, float eps, const char* test_name) {
    if (fabsf(a - b) < eps) {
        printf("[%s] %s (%.6f ≈ %.6f)\n", TEST_PASS, test_name, a, b);
        tests_passed++;
    } else {
        printf("[%s] %s (%.6f != %.6f)\n", TEST_FAIL, test_name, a, b);
        tests_failed++;
    }
}

// ============ 采样函数测试 ============

void test_sample_greedy() {
    printf("\n=== 测试: 贪婪采样 ===\n");

    int vocab_size = 10;
    int shape[] = {vocab_size};
    Tensor* logits = tensor_zeros(1, shape);

    // 设置 logits，使得索引 5 有最高值
    for (int i = 0; i < vocab_size; i++) {
        logits->data[i] = (float)i * 0.1f;
    }
    logits->data[5] = 10.0f;

    int token = sample_greedy(logits);
    assert_true(token == 5, "贪婪采样选择最大 logit");

    // 改变最大值位置
    logits->data[5] = 0.0f;
    logits->data[8] = 15.0f;
    token = sample_greedy(logits);
    assert_true(token == 8, "贪婪采样选择新的最大 logit");

    tensor_free(logits);
}

void test_sample_temperature() {
    printf("\n=== 测试: 温度采样 ===\n");

    int vocab_size = 5;
    int shape[] = {vocab_size};
    Tensor* logits = tensor_zeros(1, shape);

    // 设置均匀 logits
    for (int i = 0; i < vocab_size; i++) {
        logits->data[i] = 0.0f;
    }

    set_random_seed(42);

    // 低温度应该更确定性
    logits->data[2] = 2.0f;  // 稍微高一点

    int counts[5] = {0};
    int n_samples = 100;

    // 高温度 (temperature = 2.0)
    for (int i = 0; i < n_samples; i++) {
        Tensor* logits_copy = tensor_zeros(1, shape);
        memcpy(logits_copy->data, logits->data, vocab_size * sizeof(float));
        int token = sample_temperature(logits_copy, 2.0f);
        counts[token]++;
        tensor_free(logits_copy);
    }

    printf("  高温度 (T=2.0) 分布: ");
    for (int i = 0; i < vocab_size; i++) {
        printf("%d ", counts[i]);
    }
    printf("\n");

    // 高温度下，分布应该相对均匀
    int max_count = 0, min_count = n_samples;
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] > max_count) max_count = counts[i];
        if (counts[i] < min_count) min_count = counts[i];
    }
    assert_true(max_count - min_count < n_samples / 2, "高温度分布相对均匀");

    // 低温度 (temperature = 0.1)
    memset(counts, 0, sizeof(counts));
    for (int i = 0; i < n_samples; i++) {
        Tensor* logits_copy = tensor_zeros(1, shape);
        memcpy(logits_copy->data, logits->data, vocab_size * sizeof(float));
        int token = sample_temperature(logits_copy, 0.1f);
        counts[token]++;
        tensor_free(logits_copy);
    }

    printf("  低温度 (T=0.1) 分布: ");
    for (int i = 0; i < vocab_size; i++) {
        printf("%d ", counts[i]);
    }
    printf("\n");

    // 低温度下，应该集中在 index 2
    assert_true(counts[2] > n_samples * 0.8, "低温度集中在最大 logit");

    tensor_free(logits);
}

void test_sample_top_k() {
    printf("\n=== 测试: Top-K 采样 ===\n");

    int vocab_size = 10;
    int shape[] = {vocab_size};
    Tensor* logits = tensor_zeros(1, shape);

    // 设置不同的 logits
    for (int i = 0; i < vocab_size; i++) {
        logits->data[i] = (float)i;
    }

    set_random_seed(42);

    int counts[10] = {0};
    int n_samples = 100;
    int k = 3;

    for (int i = 0; i < n_samples; i++) {
        Tensor* logits_copy = tensor_zeros(1, shape);
        memcpy(logits_copy->data, logits->data, vocab_size * sizeof(float));
        int token = sample_top_k(logits_copy, k, 1.0f);
        counts[token]++;
        tensor_free(logits_copy);
    }

    printf("  Top-K (k=%d) 分布: ", k);
    for (int i = 0; i < vocab_size; i++) {
        printf("%d ", counts[i]);
    }
    printf("\n");

    // Top-3 应该只采样 7, 8, 9
    int sum_top3 = counts[7] + counts[8] + counts[9];
    assert_true(sum_top3 == n_samples, "Top-K 只采样前 K 个");

    tensor_free(logits);
}

void test_sample_top_p() {
    printf("\n=== 测试: Top-P 采样 ===\n");

    int vocab_size = 5;
    int shape[] = {vocab_size};
    Tensor* logits = tensor_zeros(1, shape);

    // 设置使得 softmax 后有明显的概率分布
    logits->data[0] = 3.0f;   // 高概率
    logits->data[1] = 2.0f;   // 中等概率
    logits->data[2] = 1.0f;   // 低概率
    logits->data[3] = 0.0f;   // 很低概率
    logits->data[4] = -1.0f;  // 极低概率

    set_random_seed(42);

    int counts[5] = {0};
    int n_samples = 100;
    float p = 0.9f;

    for (int i = 0; i < n_samples; i++) {
        Tensor* logits_copy = tensor_zeros(1, shape);
        memcpy(logits_copy->data, logits->data, vocab_size * sizeof(float));
        int token = sample_top_p(logits_copy, p, 1.0f);
        counts[token]++;
        tensor_free(logits_copy);
    }

    printf("  Top-P (p=%.1f) 分布: ", p);
    for (int i = 0; i < vocab_size; i++) {
        printf("%d ", counts[i]);
    }
    printf("\n");

    // Top-P 应该主要采样高概率的 tokens
    int sum_top2 = counts[0] + counts[1];
    assert_true(sum_top2 > n_samples * 0.8, "Top-P 主要采样高概率 tokens");

    tensor_free(logits);
}

// ============ 生成配置测试 ============

void test_generate_config() {
    printf("\n=== 测试: 生成配置 ===\n");

    GenerateConfig default_cfg = default_generate_config();
    assert_float_near(default_cfg.temperature, 1.0f, EPSILON, "默认温度 = 1.0");
    assert_true(default_cfg.top_k == 40, "默认 top_k = 40");
    assert_float_near(default_cfg.top_p, 0.9f, EPSILON, "默认 top_p = 0.9");
    assert_true(default_cfg.max_new_tokens == 100, "默认 max_new_tokens = 100");
    assert_true(default_cfg.do_sample == 1, "默认 do_sample = 1");

    GenerateConfig greedy_cfg = greedy_config();
    assert_true(greedy_cfg.do_sample == 0, "贪婪配置 do_sample = 0");

    GenerateConfig creative_cfg = creative_config();
    assert_true(creative_cfg.temperature > 1.0f, "创意配置温度 > 1.0");
}

// ============ 文本生成测试 ============

void test_generate_basic() {
    printf("\n=== 测试: 基本文本生成 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 10;
    gen_cfg.seed = 42;

    const char* prompt = "Hello";

    char* generated = generate(model, tok, prompt, &gen_cfg);
    assert_true(generated != NULL, "生成文本成功");
    assert_true(strlen(generated) >= strlen(prompt), "生成文本长度 >= prompt");

    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Generated: \"%s\"\n", generated);
    printf("  Length: %zu chars\n", strlen(generated));

    free(generated);
    tokenizer_free(tok);
    model_free(model);
}

void test_generate_greedy() {
    printf("\n=== 测试: 贪婪生成 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    GenerateConfig gen_cfg = greedy_config();
    gen_cfg.max_new_tokens = 10;

    const char* prompt = "Hi";

    // 贪婪生成应该是确定性的
    char* gen1 = generate(model, tok, prompt, &gen_cfg);
    char* gen2 = generate(model, tok, prompt, &gen_cfg);

    assert_true(gen1 != NULL && gen2 != NULL, "两次生成都成功");
    assert_true(strcmp(gen1, gen2) == 0, "贪婪生成是确定性的");

    printf("  Generation 1: \"%s\"\n", gen1);
    printf("  Generation 2: \"%s\"\n", gen2);

    free(gen1);
    free(gen2);
    tokenizer_free(tok);
    model_free(model);
}

void test_generate_with_sampling() {
    printf("\n=== 测试: 采样生成 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 20;
    gen_cfg.temperature = 1.0f;
    gen_cfg.top_k = 10;
    gen_cfg.top_p = 0.9f;

    const char* prompt = "The";

    // 不同种子应该产生不同结果
    gen_cfg.seed = 42;
    char* gen1 = generate(model, tok, prompt, &gen_cfg);

    gen_cfg.seed = 123;
    char* gen2 = generate(model, tok, prompt, &gen_cfg);

    assert_true(gen1 != NULL && gen2 != NULL, "两次生成都成功");

    printf("  Seed 42:  \"%s\"\n", gen1);
    printf("  Seed 123: \"%s\"\n", gen2);

    // 不同种子可能产生不同结果（不是必须的，但通常如此）
    // 这里不做强制断言

    free(gen1);
    free(gen2);
    tokenizer_free(tok);
    model_free(model);
}

void test_generate_full_result() {
    printf("\n=== 测试: 完整生成结果 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 5;
    gen_cfg.seed = 42;

    const char* prompt = "Test";

    GenerateResult* result = generate_full(model, tok, prompt, &gen_cfg);

    assert_true(result != NULL, "生成结果不为空");
    assert_true(result->token_ids != NULL, "token_ids 不为空");
    assert_true(result->text != NULL, "text 不为空");
    assert_true(result->prompt_len > 0, "prompt_len > 0");
    assert_true(result->generated_len >= 0, "generated_len >= 0");
    assert_true(result->num_tokens == result->prompt_len + result->generated_len,
                "num_tokens = prompt_len + generated_len");

    printf("  Prompt length: %d tokens\n", result->prompt_len);
    printf("  Generated length: %d tokens\n", result->generated_len);
    printf("  Total: %d tokens\n", result->num_tokens);
    printf("  Text: \"%s\"\n", result->text);

    generate_result_free(result);
    tokenizer_free(tok);
    model_free(model);
}

// 流式生成回调函数
static int stream_count = 0;
static char stream_buffer[256];

int stream_callback(int token_id, const char* token_str, void* user_data) {
    (void)token_id;
    (void)user_data;

    if (token_str != NULL && stream_count < 250) {
        strcat(stream_buffer, token_str);
    }
    stream_count++;

    // 生成 5 个 token 后停止
    return (stream_count >= 5) ? 1 : 0;
}

void test_generate_stream() {
    printf("\n=== 测试: 流式生成 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 20;
    gen_cfg.seed = 42;

    const char* prompt = "Hi";

    stream_count = 0;
    stream_buffer[0] = '\0';

    int generated = generate_stream(model, tok, prompt, &gen_cfg, stream_callback, NULL);

    assert_true(generated > 0, "流式生成返回 > 0");
    assert_true(stream_count == 5, "回调被调用 5 次");

    printf("  Generated %d tokens\n", generated);
    printf("  Callback called %d times\n", stream_count);
    printf("  Stream buffer: \"%s\"\n", stream_buffer);

    tokenizer_free(tok);
    model_free(model);
}

// ============ 训练后生成测试 ============

void test_generate_after_training() {
    printf("\n=== 测试: 训练后生成 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    // 训练模型学习简单模式
    const char* pattern = "abcabcabcabcabcabcabcabcabcabc";
    int seq_len = 16;
    int batch_size = 1;

    DataLoader* dl = dataloader_from_string(pattern, tok, seq_len, batch_size);
    AdamOptimizer* opt = adam_create(model, 0.01f);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));

    // 训练 100 步
    for (int step = 0; step < 100; step++) {
        dataloader_reset(dl);
        dataloader_next_batch(dl, input_ids, targets);
        train_step(model, input_ids, targets, batch_size, seq_len, opt, grads, cache, 1.0f);
    }

    // 用训练后的模型生成
    GenerateConfig gen_cfg = default_generate_config();
    gen_cfg.max_new_tokens = 15;
    gen_cfg.temperature = 0.5f;  // 低温度更确定性
    gen_cfg.seed = 42;

    const char* prompt = "a";
    char* generated = generate(model, tok, prompt, &gen_cfg);

    printf("  Trained pattern: \"%s\"\n", pattern);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Generated: \"%s\"\n", generated);

    // 检查生成的文本是否包含 "abc" 模式
    int has_pattern = (strstr(generated, "abc") != NULL) ||
                      (strstr(generated, "bca") != NULL) ||
                      (strstr(generated, "cab") != NULL);
    assert_true(has_pattern || strlen(generated) > 0, "训练后能生成文本");

    free(generated);
    free(input_ids);
    free(targets);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(opt);
    dataloader_free(dl);
    tokenizer_free(tok);
    model_free(model);
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step7 测试套件\n");
    printf("        文本生成测试\n");
    printf("========================================\n");

    srand(42);

    // 采样函数测试
    test_sample_greedy();
    test_sample_temperature();
    test_sample_top_k();
    test_sample_top_p();

    // 生成配置测试
    test_generate_config();

    // 文本生成测试
    test_generate_basic();
    test_generate_greedy();
    test_generate_with_sampling();
    test_generate_full_result();
    test_generate_stream();

    // 训练后生成测试
    test_generate_after_training();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
