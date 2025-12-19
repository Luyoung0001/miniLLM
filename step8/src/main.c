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
#include "chat.h"

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

void assert_str_equal(const char* a, const char* b, const char* test_name) {
    if (a == NULL && b == NULL) {
        printf("[%s] %s (both NULL)\n", TEST_PASS, test_name);
        tests_passed++;
    } else if (a != NULL && b != NULL && strcmp(a, b) == 0) {
        printf("[%s] %s\n", TEST_PASS, test_name);
        tests_passed++;
    } else {
        printf("[%s] %s (\"%s\" != \"%s\")\n", TEST_FAIL, test_name,
               a ? a : "NULL", b ? b : "NULL");
        tests_failed++;
    }
}

// ============ 对话管理测试 ============

void test_conversation_create() {
    printf("\n=== 测试: 对话创建 ===\n");

    Conversation* conv = conversation_create();
    assert_true(conv != NULL, "创建对话成功");
    assert_true(conversation_length(conv) == 0, "新对话消息数为 0");

    conversation_free(conv);
}

void test_conversation_add() {
    printf("\n=== 测试: 添加消息 ===\n");

    Conversation* conv = conversation_create();

    conversation_add(conv, "user", "你好");
    assert_true(conversation_length(conv) == 1, "添加后消息数为 1");

    Message* msg = conversation_get(conv, 0);
    assert_true(msg != NULL, "获取消息成功");
    assert_str_equal(msg->role, "user", "消息角色正确");
    assert_str_equal(msg->content, "你好", "消息内容正确");

    conversation_add(conv, "assistant", "你好！有什么可以帮助你的？");
    assert_true(conversation_length(conv) == 2, "添加后消息数为 2");

    Message* last = conversation_last(conv);
    assert_str_equal(last->role, "assistant", "最后消息角色正确");

    conversation_free(conv);
}

void test_conversation_clear() {
    printf("\n=== 测试: 清空对话 ===\n");

    Conversation* conv = conversation_create();
    conversation_set_system(conv, "你是一个有帮助的助手。");

    conversation_add(conv, "user", "你好");
    conversation_add(conv, "assistant", "你好！");

    assert_true(conversation_length(conv) == 2, "清空前消息数为 2");

    conversation_clear(conv);
    assert_true(conversation_length(conv) == 0, "清空后消息数为 0");
    assert_true(conv->system_prompt != NULL, "系统提示保留");

    conversation_free(conv);
}

void test_conversation_system_prompt() {
    printf("\n=== 测试: 系统提示 ===\n");

    Conversation* conv = conversation_create();
    assert_true(conv->system_prompt == NULL, "初始系统提示为 NULL");

    conversation_set_system(conv, "你是一个有帮助的助手。");
    assert_true(conv->system_prompt != NULL, "设置系统提示成功");
    assert_str_equal(conv->system_prompt, "你是一个有帮助的助手。", "系统提示内容正确");

    // 更新系统提示
    conversation_set_system(conv, "新的系统提示");
    assert_str_equal(conv->system_prompt, "新的系统提示", "更新系统提示成功");

    conversation_free(conv);
}

// ============ Prompt 格式化测试 ============

void test_format_prompt() {
    printf("\n=== 测试: Prompt 格式化 ===\n");

    Conversation* conv = conversation_create();
    conversation_add(conv, "user", "你好");
    conversation_add(conv, "assistant", "你好！");

    char* prompt = format_prompt(conv);
    assert_true(prompt != NULL, "格式化 prompt 成功");
    assert_true(strstr(prompt, "<|user|>") != NULL, "包含 user 标签");
    assert_true(strstr(prompt, "<|assistant|>") != NULL, "包含 assistant 标签");
    assert_true(strstr(prompt, "你好") != NULL, "包含用户消息");

    printf("  Formatted prompt:\n%s\n", prompt);

    free(prompt);
    conversation_free(conv);
}

void test_format_prompt_with_system() {
    printf("\n=== 测试: 带系统提示的格式化 ===\n");

    Conversation* conv = conversation_create();
    conversation_set_system(conv, "你是一个有帮助的助手。");
    conversation_add(conv, "user", "你好");

    char* prompt = format_prompt(conv);
    assert_true(strstr(prompt, "<|system|>") != NULL, "包含 system 标签");
    assert_true(strstr(prompt, "有帮助的助手") != NULL, "包含系统提示");

    printf("  Formatted prompt with system:\n%s\n", prompt);

    free(prompt);
    conversation_free(conv);
}

void test_format_prompt_with_input() {
    printf("\n=== 测试: 带用户输入的格式化 ===\n");

    Conversation* conv = conversation_create();
    conversation_add(conv, "user", "你好");
    conversation_add(conv, "assistant", "你好！有什么可以帮助你的？");

    char* prompt = format_prompt_with_input(conv, "今天天气怎么样？");
    assert_true(prompt != NULL, "格式化成功");

    // 应该以 <|assistant|>\n 结尾
    assert_true(strstr(prompt, "今天天气怎么样") != NULL, "包含新用户输入");

    // 统计 assistant 标签数量
    int count = 0;
    const char* ptr = prompt;
    while ((ptr = strstr(ptr, "<|assistant|>")) != NULL) {
        count++;
        ptr += strlen("<|assistant|>");
    }
    assert_true(count == 2, "包含 2 个 assistant 标签 (历史 + 待生成)");

    printf("  Formatted prompt with input:\n%s\n", prompt);

    free(prompt);
    conversation_free(conv);
}

// ============ 辅助函数测试 ============

void test_trim_string() {
    printf("\n=== 测试: 字符串裁剪 ===\n");

    char* result;

    result = trim_string("  hello  ");
    assert_str_equal(result, "hello", "裁剪两端空白");
    free(result);

    result = trim_string("hello");
    assert_str_equal(result, "hello", "无空白不变");
    free(result);

    result = trim_string("   ");
    assert_str_equal(result, "", "全空白返回空字符串");
    free(result);

    result = trim_string("\n\thello\n\t");
    assert_str_equal(result, "hello", "裁剪换行和制表符");
    free(result);
}

void test_starts_ends_with() {
    printf("\n=== 测试: 字符串前后缀检查 ===\n");

    assert_true(starts_with("hello world", "hello"), "starts_with 匹配");
    assert_true(!starts_with("hello world", "world"), "starts_with 不匹配");
    assert_true(ends_with("hello world", "world"), "ends_with 匹配");
    assert_true(!ends_with("hello world", "hello"), "ends_with 不匹配");
}

void test_extract_response() {
    printf("\n=== 测试: 响应提取 ===\n");

    // 简单情况
    const char* output1 = "<|assistant|>\n这是回复\n";
    char* response1 = extract_response(output1);
    assert_str_equal(response1, "这是回复", "提取简单回复");
    free(response1);

    // 带多轮对话
    const char* output2 = "<|user|>\n你好\n<|assistant|>\n你好！\n<|user|>\n谢谢\n<|assistant|>\n不客气！\n";
    char* response2 = extract_response(output2);
    assert_str_equal(response2, "不客气！", "提取最后一个回复");
    free(response2);

    // 带结束标签
    const char* output3 = "<|assistant|>\n回复内容<|end|>";
    char* response3 = extract_response(output3);
    assert_str_equal(response3, "回复内容", "在结束标签前截止");
    free(response3);
}

// ============ 对话配置测试 ============

void test_chat_config() {
    printf("\n=== 测试: 对话配置 ===\n");

    ChatConfig config = default_chat_config();
    assert_true(config.gen_config.temperature > 0, "温度 > 0");
    assert_true(config.gen_config.max_new_tokens > 0, "max_new_tokens > 0");
    assert_true(config.max_history > 0, "max_history > 0");
    assert_true(config.trim_history == 1, "trim_history 默认启用");

    printf("  默认配置:\n");
    printf("    temperature: %.2f\n", config.gen_config.temperature);
    printf("    max_new_tokens: %d\n", config.gen_config.max_new_tokens);
    printf("    max_history: %d\n", config.max_history);
}

// ============ 对话功能测试 ============

void test_chat_basic() {
    printf("\n=== 测试: 基本对话 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 10;
    chat_cfg.gen_config.seed = 42;

    char* response = chat(model, tok, conv, "你好", &chat_cfg);

    assert_true(response != NULL, "对话生成成功");
    assert_true(strlen(response) >= 0, "响应不为 NULL");
    assert_true(conversation_length(conv) == 2, "对话历史有 2 条消息");

    printf("  用户: 你好\n");
    printf("  助手: %s\n", response);

    // 检查对话历史
    Message* user_msg = conversation_get(conv, 0);
    Message* asst_msg = conversation_get(conv, 1);
    assert_str_equal(user_msg->role, "user", "第一条是用户消息");
    assert_str_equal(asst_msg->role, "assistant", "第二条是助手消息");

    free(response);
    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

void test_chat_multi_turn() {
    printf("\n=== 测试: 多轮对话 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 5;
    chat_cfg.gen_config.seed = 42;

    // 第一轮
    char* response1 = chat(model, tok, conv, "你好", &chat_cfg);
    assert_true(response1 != NULL, "第一轮对话成功");
    assert_true(conversation_length(conv) == 2, "第一轮后有 2 条消息");
    printf("  轮次 1 - 用户: 你好\n");
    printf("  轮次 1 - 助手: %s\n", response1);
    free(response1);

    // 第二轮
    char* response2 = chat(model, tok, conv, "谢谢", &chat_cfg);
    assert_true(response2 != NULL, "第二轮对话成功");
    assert_true(conversation_length(conv) == 4, "第二轮后有 4 条消息");
    printf("  轮次 2 - 用户: 谢谢\n");
    printf("  轮次 2 - 助手: %s\n", response2);
    free(response2);

    // 第三轮
    char* response3 = chat(model, tok, conv, "再见", &chat_cfg);
    assert_true(response3 != NULL, "第三轮对话成功");
    assert_true(conversation_length(conv) == 6, "第三轮后有 6 条消息");
    printf("  轮次 3 - 用户: 再见\n");
    printf("  轮次 3 - 助手: %s\n", response3);
    free(response3);

    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

void test_chat_with_system_prompt() {
    printf("\n=== 测试: 带系统提示的对话 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();
    conversation_set_system(conv, "你是一个有帮助的助手。");

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 10;
    chat_cfg.gen_config.seed = 42;

    char* response = chat(model, tok, conv, "你好", &chat_cfg);

    assert_true(response != NULL, "带系统提示的对话成功");
    printf("  系统: 你是一个有帮助的助手。\n");
    printf("  用户: 你好\n");
    printf("  助手: %s\n", response);

    free(response);
    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

void test_chat_history_trimming() {
    printf("\n=== 测试: 历史消息裁剪 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 3;
    chat_cfg.gen_config.seed = 42;
    chat_cfg.max_history = 4;  // 最多保留 4 条消息
    chat_cfg.trim_history = 1;

    // 进行多轮对话
    for (int i = 0; i < 5; i++) {
        char input[32];
        sprintf(input, "消息 %d", i + 1);
        char* response = chat(model, tok, conv, input, &chat_cfg);
        free(response);
    }

    // 检查历史是否被裁剪
    assert_true(conversation_length(conv) <= chat_cfg.max_history + 2,
                "历史消息被正确裁剪");

    printf("  进行了 5 轮对话\n");
    printf("  max_history = %d\n", chat_cfg.max_history);
    printf("  当前消息数: %d\n", conversation_length(conv));

    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

// ============ 流式对话测试 ============

static int stream_token_count = 0;
static char stream_output[4096];

int test_stream_callback(const char* token, void* user_data) {
    (void)user_data;
    size_t current_len = strlen(stream_output);
    if (token != NULL && current_len + strlen(token) < sizeof(stream_output) - 1) {
        strcat(stream_output, token);
    }
    stream_token_count++;
    return (stream_token_count >= 5) ? 1 : 0;  // 生成 5 个 token 后停止
}

void test_chat_stream() {
    printf("\n=== 测试: 流式对话 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 20;
    chat_cfg.gen_config.seed = 42;

    stream_token_count = 0;
    stream_output[0] = '\0';

    int generated = chat_stream(model, tok, conv, "你好", &chat_cfg,
                                test_stream_callback, NULL);

    assert_true(generated > 0, "流式对话生成 > 0 tokens");
    assert_true(stream_token_count == 5, "回调被调用 5 次");
    assert_true(conversation_length(conv) == 2, "对话历史已更新");

    printf("  生成 tokens: %d\n", generated);
    printf("  回调次数: %d\n", stream_token_count);
    printf("  流式输出: %s\n", stream_output);

    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

// ============ 训练后对话测试 ============

void test_chat_after_training() {
    printf("\n=== 测试: 训练后对话 ===\n");

    ModelConfig config = tiny_config();
    GPTModel* model = model_create(config);
    model_init_random(model, 0.02f);

    Tokenizer* tok = tokenizer_create();

    // 训练数据：简单的对话模式
    const char* train_data = "<|user|>\nhi\n<|assistant|>\nhello\n<|user|>\nbye\n<|assistant|>\ngoodbye\n";

    int seq_len = 32;
    int batch_size = 1;

    DataLoader* dl = dataloader_from_string(train_data, tok, seq_len, batch_size);
    AdamOptimizer* opt = adam_create(model, 0.01f);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, seq_len);

    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));

    // 训练 50 步
    printf("  训练中...\n");
    for (int step = 0; step < 50; step++) {
        dataloader_reset(dl);
        if (dataloader_next_batch(dl, input_ids, targets)) {
            train_step(model, input_ids, targets, batch_size, seq_len, opt, grads, cache, 1.0f);
        }
    }
    printf("  训练完成\n");

    // 用训练后的模型对话
    Conversation* conv = conversation_create();

    ChatConfig chat_cfg = default_chat_config();
    chat_cfg.gen_config.max_new_tokens = 10;
    chat_cfg.gen_config.temperature = 0.5f;
    chat_cfg.gen_config.seed = 42;

    char* response = chat(model, tok, conv, "hi", &chat_cfg);

    assert_true(response != NULL, "训练后对话成功");
    printf("  用户: hi\n");
    printf("  助手: %s\n", response);

    free(response);
    free(input_ids);
    free(targets);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(opt);
    dataloader_free(dl);
    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step8 测试套件\n");
    printf("        对话系统测试\n");
    printf("========================================\n");

    srand(42);

    // 对话管理测试
    test_conversation_create();
    test_conversation_add();
    test_conversation_clear();
    test_conversation_system_prompt();

    // Prompt 格式化测试
    test_format_prompt();
    test_format_prompt_with_system();
    test_format_prompt_with_input();

    // 辅助函数测试
    test_trim_string();
    test_starts_ends_with();
    test_extract_response();

    // 配置测试
    test_chat_config();

    // 对话功能测试
    test_chat_basic();
    test_chat_multi_turn();
    test_chat_with_system_prompt();
    test_chat_history_trimming();

    // 流式对话测试
    test_chat_stream();

    // 训练后对话测试
    test_chat_after_training();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
