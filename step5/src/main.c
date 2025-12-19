#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "config.h"
#include "model.h"

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

// ============ Config 测试 ============

void test_config() {
    printf("\n=== 测试: Config 配置 ===\n");

    ModelConfig config = default_config();
    assert_int_eq(config.vocab_size, 260, "默认 vocab_size = 260");
    assert_int_eq(config.hidden_dim, 64, "默认 hidden_dim = 64");
    assert_int_eq(config.num_heads, 4, "默认 num_heads = 4");
    assert_int_eq(config.num_layers, 4, "默认 num_layers = 4");
    assert_int_eq(config.ffn_dim, 256, "默认 ffn_dim = 256");
    assert_int_eq(config.max_seq_len, 128, "默认 max_seq_len = 128");

    assert_true(config_validate(&config), "默认配置有效");

    // 测试无效配置
    ModelConfig invalid = config;
    invalid.hidden_dim = 65;  // 不能被 num_heads 整除
    assert_true(!config_validate(&invalid), "无效配置被检测");

    // 测试保存和加载
    const char* config_path = "/tmp/test_config.bin";
    assert_true(config_save(&config, config_path) == 0, "保存配置成功");

    ModelConfig loaded;
    assert_true(config_load(&loaded, config_path) == 0, "加载配置成功");
    assert_int_eq(loaded.vocab_size, config.vocab_size, "加载后 vocab_size 相同");
    assert_int_eq(loaded.hidden_dim, config.hidden_dim, "加载后 hidden_dim 相同");

    remove(config_path);
}

// ============ Model 测试 ============

void test_model_create() {
    printf("\n=== 测试: Model 创建 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);

    assert_true(model != NULL, "创建模型成功");
    assert_true(model->embedding != NULL, "embedding 已创建");
    assert_true(model->layers != NULL, "layers 已创建");
    assert_true(model->final_ln != NULL, "final_ln 已创建");
    assert_true(model->lm_head != NULL, "lm_head 已创建");

    assert_int_eq(model->lm_head->shape[0], config.hidden_dim, "lm_head shape[0] 正确");
    assert_int_eq(model->lm_head->shape[1], config.vocab_size, "lm_head shape[1] 正确");

    int num_params = model_num_params(model);
    assert_true(num_params > 0, "参数数量 > 0");
    printf("  Tiny model params: %d\n", num_params);

    model_free(model);
}

void test_model_forward() {
    printf("\n=== 测试: Model 前向传播 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    int seq_len = 5;
    GPTCache* cache = model_cache_create(model, seq_len);

    // 创建输入
    int input_ids[] = {76, 105, 112, 112, 115};  // "Hello"

    // 创建输出 logits
    int logits_shape[] = {seq_len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);

    // 前向传播
    model_forward(model, input_ids, seq_len, cache, logits);

    assert_int_eq(logits->shape[0], seq_len, "logits shape[0] 正确");
    assert_int_eq(logits->shape[1], config.vocab_size, "logits shape[1] 正确");

    // logits 不应全为零
    float sum = 0.0f;
    for (int i = 0; i < logits->size; i++) {
        sum += fabsf(logits->data[i]);
    }
    assert_true(sum > 0.0f, "logits 不全为零");

    // 测试 softmax 后的概率分布
    int last_logits_shape[] = {config.vocab_size};
    Tensor* last_logits = tensor_zeros(1, last_logits_shape);
    model_get_last_logits(logits, seq_len, last_logits);

    Tensor* probs = tensor_zeros(1, last_logits_shape);
    softmax(probs, last_logits);

    float prob_sum = tensor_sum(probs);
    assert_float_near(prob_sum, 1.0f, 0.001f, "softmax 后概率和为 1");

    tensor_free(logits);
    tensor_free(last_logits);
    tensor_free(probs);
    model_cache_free(cache);
    model_free(model);
}

void test_model_different_seq_len() {
    printf("\n=== 测试: 不同序列长度 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    GPTCache* cache = model_cache_create(model, 1);

    int seq_lens[] = {1, 3, 5, 10};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lens[t];

        int* input_ids = (int*)malloc(seq_len * sizeof(int));
        for (int i = 0; i < seq_len; i++) {
            input_ids[i] = 76 + i;  // 随意的 token IDs
        }

        int logits_shape[] = {seq_len, config.vocab_size};
        Tensor* logits = tensor_zeros(2, logits_shape);

        model_forward(model, input_ids, seq_len, cache, logits);

        char test_name[64];
        snprintf(test_name, sizeof(test_name), "seq_len=%d 前向传播成功", seq_len);
        assert_true(logits->shape[0] == seq_len, test_name);

        free(input_ids);
        tensor_free(logits);
    }

    model_cache_free(cache);
    model_free(model);
}

void test_model_save_load() {
    printf("\n=== 测试: Model 保存和加载 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model1 = model_create(config);
    model_init_random(model1, 0.02f);

    // 保存模型
    const char* model_path = "/tmp/test_model.bin";
    int save_result = model_save(model1, model_path);
    assert_true(save_result == 0, "保存模型成功");

    // 加载模型
    GPTModel* model2 = model_load(model_path);
    assert_true(model2 != NULL, "加载模型成功");

    // 验证配置相同
    assert_int_eq(model2->config.vocab_size, model1->config.vocab_size, "vocab_size 相同");
    assert_int_eq(model2->config.hidden_dim, model1->config.hidden_dim, "hidden_dim 相同");
    assert_int_eq(model2->config.num_layers, model1->config.num_layers, "num_layers 相同");

    // 验证参数数量相同
    assert_int_eq(model_num_params(model2), model_num_params(model1), "参数数量相同");

    // 验证前向传播结果相同
    int seq_len = 3;
    int input_ids[] = {76, 105, 112};

    int logits_shape[] = {seq_len, config.vocab_size};
    Tensor* logits1 = tensor_zeros(2, logits_shape);
    Tensor* logits2 = tensor_zeros(2, logits_shape);

    model_forward(model1, input_ids, seq_len, NULL, logits1);
    model_forward(model2, input_ids, seq_len, NULL, logits2);

    // 比较输出
    int outputs_match = 1;
    for (int i = 0; i < logits1->size; i++) {
        if (fabsf(logits1->data[i] - logits2->data[i]) > EPSILON) {
            outputs_match = 0;
            break;
        }
    }
    assert_true(outputs_match, "保存/加载后输出相同");

    tensor_free(logits1);
    tensor_free(logits2);
    model_free(model1);
    model_free(model2);
    remove(model_path);
}

void test_model_with_tokenizer() {
    printf("\n=== 测试: 与 Tokenizer 集成 ===\n");

    // 创建 tokenizer 和 model
    Tokenizer* tok = tokenizer_create();
    ModelConfig config = tiny_config();
    config.vocab_size = tok->vocab_size;

    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    // 编码文本
    const char* text = "Hello";
    int len;
    int* ids = tokenizer_encode(tok, text, &len, 0, 0);

    assert_true(ids != NULL, "文本编码成功");
    printf("  输入文本: \"%s\"\n", text);
    printf("  Token IDs: ");
    for (int i = 0; i < len; i++) {
        printf("%d ", ids[i]);
    }
    printf("\n");

    // 前向传播
    int logits_shape[] = {len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    model_forward(model, ids, len, NULL, logits);

    // 获取下一个 token 预测
    int last_shape[] = {config.vocab_size};
    Tensor* last_logits = tensor_zeros(1, last_shape);
    model_get_last_logits(logits, len, last_logits);

    int predicted_id = tensor_argmax(last_logits);
    const char* predicted_token = tokenizer_decode_id(tok, predicted_id);
    printf("  预测下一个 token: %d ('%s')\n", predicted_id, predicted_token);

    assert_true(predicted_id >= 0 && predicted_id < config.vocab_size, "预测 ID 在有效范围内");

    free(ids);
    tensor_free(logits);
    tensor_free(last_logits);
    model_free(model);
    tokenizer_free(tok);

    tests_passed++;  // 集成测试通过
}

void test_model_info() {
    printf("\n=== 测试: Model 信息 ===\n");

    ModelConfig config = default_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    model_print_info(model);

    model_free(model);
    tests_passed++;
}

void test_model_params_breakdown() {
    printf("\n=== 测试: 参数数量细分 ===\n");

    ModelConfig config = default_config();
    GPTModel* model = model_create(config);

    int embedding_params = embedding_num_params(model->embedding);
    int layer_params = 0;
    for (int i = 0; i < config.num_layers; i++) {
        layer_params += transformer_block_num_params(model->layers[i]);
    }
    int final_ln_params = layernorm_num_params(model->final_ln);
    int lm_head_params = model->lm_head->size;
    int total = embedding_params + layer_params + final_ln_params + lm_head_params;

    printf("  Embedding: %d\n", embedding_params);
    printf("  Transformer Layers: %d\n", layer_params);
    printf("  Final LayerNorm: %d\n", final_ln_params);
    printf("  LM Head: %d\n", lm_head_params);
    printf("  计算总和: %d\n", total);
    printf("  model_num_params: %d\n", model_num_params(model));

    assert_int_eq(model_num_params(model), total, "参数数量计算一致");

    model_free(model);
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step5 测试套件\n");
    printf("       完整 GPT 模型测试\n");
    printf("========================================\n");

    srand(42);

    test_config();
    test_model_create();
    test_model_forward();
    test_model_different_seq_len();
    test_model_save_load();
    test_model_with_tokenizer();
    test_model_params_breakdown();
    test_model_info();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
