#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "config.h"
#include "model.h"
#include "loss.h"
#include "backward.h"
#include "optimizer.h"
#include "dataloader.h"
#include "train.h"

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

// ============ Loss 测试 ============

void test_cross_entropy_loss() {
    printf("\n=== 测试: Cross Entropy Loss ===\n");

    int vocab_size = 10;
    int seq_len = 3;

    int logits_shape[] = {seq_len, vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);

    // 设置 logits (模拟模型输出)
    for (int s = 0; s < seq_len; s++) {
        for (int v = 0; v < vocab_size; v++) {
            logits->data[s * vocab_size + v] = (float)(v - vocab_size / 2) * 0.5f;
        }
    }

    int targets[] = {5, 7, 3};  // 目标 token

    // 计算损失
    float loss = cross_entropy_loss(logits, targets, seq_len, vocab_size);
    assert_true(loss > 0.0f, "损失值 > 0");
    printf("  Loss: %.4f\n", loss);

    // 计算带梯度的损失
    Tensor* grad_logits = tensor_zeros(2, logits_shape);
    float loss_with_grad = cross_entropy_loss_with_grad(
        logits, targets, seq_len, vocab_size, grad_logits
    );

    assert_float_near(loss, loss_with_grad, EPSILON, "两种方式损失相同");

    // 检查梯度形状
    assert_true(grad_logits->shape[0] == seq_len, "梯度 shape[0] 正确");
    assert_true(grad_logits->shape[1] == vocab_size, "梯度 shape[1] 正确");

    // 梯度应该不全为零
    float grad_sum = 0.0f;
    for (int i = 0; i < grad_logits->size; i++) {
        grad_sum += fabsf(grad_logits->data[i]);
    }
    assert_true(grad_sum > 0.0f, "梯度不全为零");

    // 目标位置的梯度应该是负的 (softmax - 1)
    for (int s = 0; s < seq_len; s++) {
        float grad_at_target = grad_logits->data[s * vocab_size + targets[s]];
        assert_true(grad_at_target < 0.0f, "目标位置梯度 < 0");
    }

    tensor_free(logits);
    tensor_free(grad_logits);
}

void test_loss_convergence() {
    printf("\n=== 测试: Loss 与概率关系 ===\n");

    int vocab_size = 5;
    int seq_len = 1;

    int logits_shape[] = {seq_len, vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);

    int targets[] = {2};  // 目标 token

    // 当模型非常确定正确答案时，loss 应该接近 0
    logits->data[2] = 10.0f;  // 正确答案有很高的 logit
    float loss_confident = cross_entropy_loss(logits, targets, seq_len, vocab_size);

    // 当模型不确定时，loss 应该更高
    for (int i = 0; i < vocab_size; i++) {
        logits->data[i] = 0.0f;  // 均匀分布
    }
    float loss_uncertain = cross_entropy_loss(logits, targets, seq_len, vocab_size);

    printf("  Confident loss: %.4f\n", loss_confident);
    printf("  Uncertain loss: %.4f\n", loss_uncertain);

    assert_true(loss_confident < loss_uncertain, "确定时 loss 更低");
    assert_true(loss_confident < 0.1f, "高置信度时 loss 接近 0");

    tensor_free(logits);
}

// ============ Gradients 测试 ============

void test_gradients() {
    printf("\n=== 测试: Gradients 结构 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);

    Gradients* grads = gradients_create(model);
    assert_true(grads != NULL, "创建梯度成功");
    assert_true(grads->d_token_embedding != NULL, "token_embedding 梯度已创建");
    assert_true(grads->d_lm_head != NULL, "lm_head 梯度已创建");

    // 测试梯度清零
    grads->d_token_embedding->data[0] = 1.0f;
    grads->d_lm_head->data[0] = 2.0f;
    gradients_zero(grads);

    assert_float_near(grads->d_token_embedding->data[0], 0.0f, EPSILON, "清零后 token_embedding 梯度为 0");
    assert_float_near(grads->d_lm_head->data[0], 0.0f, EPSILON, "清零后 lm_head 梯度为 0");

    // 测试梯度范数
    grads->d_lm_head->data[0] = 3.0f;
    grads->d_lm_head->data[1] = 4.0f;
    float norm = gradients_norm(grads);
    assert_true(norm >= 5.0f - EPSILON, "梯度范数计算正确 (至少 5)");

    // 测试梯度裁剪
    float clipped_norm = gradients_clip(grads, 1.0f);
    printf("  Original norm: %.4f, Clipped norm: %.4f\n", norm, clipped_norm);
    float new_norm = gradients_norm(grads);
    assert_true(new_norm <= 1.0f + EPSILON, "裁剪后范数 <= 1");

    gradients_free(grads);
    model_free(model);
}

// ============ Optimizer 测试 ============

void test_adam_optimizer() {
    printf("\n=== 测试: Adam Optimizer ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    AdamOptimizer* opt = adam_create(model, 0.001f);
    assert_true(opt != NULL, "创建 Adam 优化器成功");
    assert_float_near(opt->lr, 0.001f, EPSILON, "学习率正确");
    assert_float_near(opt->beta1, 0.9f, EPSILON, "beta1 正确");
    assert_float_near(opt->beta2, 0.999f, EPSILON, "beta2 正确");

    Gradients* grads = gradients_create(model);

    // 设置一些梯度
    for (int i = 0; i < grads->d_lm_head->size; i++) {
        grads->d_lm_head->data[i] = 0.1f;
    }

    // 保存原始参数
    float original_param = model->lm_head->data[0];

    // 执行一步优化
    adam_step(opt, model, grads);

    // 参数应该改变
    assert_true(fabsf(model->lm_head->data[0] - original_param) > EPSILON,
                "参数更新后改变");
    printf("  Original: %.6f, Updated: %.6f\n",
           original_param, model->lm_head->data[0]);

    // 时间步应该增加
    assert_true(opt->t == 1, "时间步为 1");

    gradients_free(grads);
    adam_free(opt);
    model_free(model);
}

void test_learning_rate_schedule() {
    printf("\n=== 测试: 学习率调度 ===\n");

    float lr_max = 0.001f;
    float lr_min = 0.0001f;
    int warmup_steps = 100;
    int total_steps = 1000;

    // Warmup 阶段
    float lr_0 = warmup_lr(lr_max, 0, warmup_steps);
    float lr_50 = warmup_lr(lr_max, 50, warmup_steps);
    float lr_100 = warmup_lr(lr_max, 100, warmup_steps);

    assert_float_near(lr_0, 0.0f, EPSILON, "step 0 时 lr = 0");
    assert_true(lr_50 > lr_0, "step 50 时 lr > step 0");
    assert_float_near(lr_100, lr_max, EPSILON, "step 100 时 lr = lr_max");

    // 余弦退火
    float lr_start = cosine_lr(lr_max, lr_min, 0, total_steps);
    float lr_mid = cosine_lr(lr_max, lr_min, total_steps / 2, total_steps);
    float lr_end = cosine_lr(lr_max, lr_min, total_steps, total_steps);

    assert_float_near(lr_start, lr_max, EPSILON, "开始时 lr = lr_max");
    assert_true(lr_mid < lr_max && lr_mid > lr_min, "中间时 lr_min < lr < lr_max");
    assert_float_near(lr_end, lr_min, EPSILON, "结束时 lr = lr_min");

    printf("  Warmup: 0->%.6f, 50->%.6f, 100->%.6f\n", lr_0, lr_50, lr_100);
    printf("  Cosine: start=%.6f, mid=%.6f, end=%.6f\n", lr_start, lr_mid, lr_end);
}

// ============ DataLoader 测试 ============

void test_dataloader() {
    printf("\n=== 测试: DataLoader ===\n");

    Tokenizer* tok = tokenizer_create();
    const char* text = "Hello world! This is a test. We are testing the dataloader.";

    DataLoader* dl = dataloader_from_string(text, tok, 8, 2);
    assert_true(dl != NULL, "创建 DataLoader 成功");

    dataloader_print_info(dl);

    // 测试获取批次
    int* input_ids = (int*)malloc(2 * 8 * sizeof(int));
    int* targets = (int*)malloc(2 * 8 * sizeof(int));

    int batch_size = dataloader_next_batch(dl, input_ids, targets);
    assert_true(batch_size > 0, "获取批次成功");

    // targets 应该是 input_ids 右移一位
    printf("  Sample input:  ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", input_ids[i]);
    }
    printf("\n");
    printf("  Sample target: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", targets[i]);
    }
    printf("\n");

    // 测试打乱
    dataloader_reset(dl);
    dataloader_shuffle(dl);
    int batch_size2 = dataloader_next_batch(dl, input_ids, targets);
    assert_true(batch_size2 > 0, "打乱后获取批次成功");

    free(input_ids);
    free(targets);
    dataloader_free(dl);
    tokenizer_free(tok);
}

// ============ Backward 测试 ============

void test_backward_cache() {
    printf("\n=== 测试: Backward Cache ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    int seq_len = 5;
    BackwardCache* cache = backward_cache_create(model, seq_len);
    assert_true(cache != NULL, "创建 BackwardCache 成功");
    assert_true(cache->embed_output != NULL, "embed_output 已创建");
    assert_true(cache->layer_inputs != NULL, "layer_inputs 已创建");
    assert_true(cache->final_ln_output != NULL, "final_ln_output 已创建");

    backward_cache_free(cache);
    model_free(model);
}

void test_forward_with_cache() {
    printf("\n=== 测试: Forward with Cache ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    int seq_len = 5;
    int input_ids[] = {76, 105, 112, 112, 115};

    BackwardCache* cache = backward_cache_create(model, seq_len);

    int logits_shape[] = {seq_len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);

    model_forward_with_cache(model, input_ids, seq_len, cache, logits);

    // 检查 logits 不全为零
    float sum = 0.0f;
    for (int i = 0; i < logits->size; i++) {
        sum += fabsf(logits->data[i]);
    }
    assert_true(sum > 0.0f, "带缓存的前向传播输出不全为零");

    // 检查缓存被填充
    float embed_sum = 0.0f;
    for (int i = 0; i < cache->embed_output->size; i++) {
        embed_sum += fabsf(cache->embed_output->data[i]);
    }
    assert_true(embed_sum > 0.0f, "embed_output 缓存已填充");

    tensor_free(logits);
    backward_cache_free(cache);
    model_free(model);
}

// ============ Training Step 测试 ============

void test_train_step() {
    printf("\n=== 测试: 单步训练 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    int seq_len = 8;
    int batch_size = 2;

    AdamOptimizer* opt = adam_create(model, 0.001f);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    // 创建简单的训练数据
    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            input_ids[b * seq_len + s] = 76 + s;  // "Hello..."
            targets[b * seq_len + s] = 77 + s;    // 下一个字符
        }
    }

    // 计算初始损失
    int logits_shape[] = {seq_len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    model_forward(model, input_ids, seq_len, NULL, logits);
    float initial_loss = cross_entropy_loss(logits, targets, seq_len, config.vocab_size);
    tensor_free(logits);

    printf("  Initial loss: %.4f\n", initial_loss);

    // 训练几步
    float loss = initial_loss;
    for (int step = 0; step < 10; step++) {
        loss = train_step(model, input_ids, targets, batch_size, seq_len,
                         opt, grads, cache, 1.0f);
    }

    printf("  After 10 steps loss: %.4f\n", loss);

    // 损失应该下降 (至少不应该爆炸)
    assert_true(loss < initial_loss * 2, "训练后损失没有爆炸");
    assert_true(!isnan(loss) && !isinf(loss), "损失值有效 (非 NaN/Inf)");

    free(input_ids);
    free(targets);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(opt);
    model_free(model);
}

void test_training_loss_decreases() {
    printf("\n=== 测试: 训练损失下降 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    config.vocab_size = tok->vocab_size;

    int seq_len = 16;
    int batch_size = 1;

    // 创建一个简单的重复模式让模型学习
    const char* pattern = "abcabcabcabcabcabcabcabcabcabc";
    DataLoader* dl = dataloader_from_string(pattern, tok, seq_len, batch_size);

    AdamOptimizer* opt = adam_create(model, 0.01f);  // 较大学习率加速测试
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));

    // 记录初始损失
    dataloader_next_batch(dl, input_ids, targets);
    int logits_shape[] = {seq_len, config.vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    model_forward(model, input_ids, seq_len, NULL, logits);
    float initial_loss = cross_entropy_loss(logits, targets, seq_len, config.vocab_size);
    tensor_free(logits);

    // 训练多步
    float final_loss = initial_loss;
    int num_steps = 50;
    for (int step = 0; step < num_steps; step++) {
        dataloader_reset(dl);
        dataloader_next_batch(dl, input_ids, targets);
        final_loss = train_step(model, input_ids, targets, batch_size, seq_len,
                               opt, grads, cache, 1.0f);
    }

    printf("  Initial loss: %.4f\n", initial_loss);
    printf("  Final loss: %.4f\n", final_loss);
    printf("  Reduction: %.2f%%\n", (1.0f - final_loss / initial_loss) * 100);

    assert_true(final_loss < initial_loss, "训练后损失下降");

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
    printf("      miniLLM Step6 测试套件\n");
    printf("        训练功能测试\n");
    printf("========================================\n");

    srand(42);

    // Loss 测试
    test_cross_entropy_loss();
    test_loss_convergence();

    // Gradients 测试
    test_gradients();

    // Optimizer 测试
    test_adam_optimizer();
    test_learning_rate_schedule();

    // DataLoader 测试
    test_dataloader();

    // Backward 测试
    test_backward_cache();
    test_forward_with_cache();

    // Training 测试
    test_train_step();
    test_training_loss_decreases();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
