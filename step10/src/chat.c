#include "chat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#define INITIAL_CAPACITY 16
#define BUFFER_SIZE 4096

// ============ 辅助函数 ============

char* trim_string(const char* str) {
    if (str == NULL) return NULL;

    // 跳过前导空白
    while (*str && isspace((unsigned char)*str)) {
        str++;
    }

    if (*str == '\0') {
        char* empty = (char*)malloc(1);
        empty[0] = '\0';
        return empty;
    }

    // 找到末尾
    const char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }

    // 复制结果
    size_t len = (size_t)(end - str + 1);
    char* result = (char*)malloc(len + 1);
    memcpy(result, str, len);
    result[len] = '\0';

    return result;
}

int starts_with(const char* str, const char* prefix) {
    if (str == NULL || prefix == NULL) return 0;
    size_t prefix_len = strlen(prefix);
    return strncmp(str, prefix, prefix_len) == 0;
}

int ends_with(const char* str, const char* suffix) {
    if (str == NULL || suffix == NULL) return 0;
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

// 复制字符串
static char* str_dup(const char* str) {
    if (str == NULL) return NULL;
    size_t len = strlen(str);
    char* result = (char*)malloc(len + 1);
    memcpy(result, str, len + 1);
    return result;
}

// ============ 对话管理 ============

Conversation* conversation_create(void) {
    Conversation* conv = (Conversation*)malloc(sizeof(Conversation));
    conv->capacity = INITIAL_CAPACITY;
    conv->messages = (Message*)malloc(conv->capacity * sizeof(Message));
    conv->num_messages = 0;
    conv->system_prompt = NULL;
    return conv;
}

void conversation_free(Conversation* conv) {
    if (conv == NULL) return;

    // 释放每条消息
    for (int i = 0; i < conv->num_messages; i++) {
        free(conv->messages[i].role);
        free(conv->messages[i].content);
    }
    free(conv->messages);
    free(conv->system_prompt);
    free(conv);
}

void conversation_add(Conversation* conv, const char* role, const char* content) {
    if (conv == NULL || role == NULL || content == NULL) return;

    // 扩容
    if (conv->num_messages >= conv->capacity) {
        conv->capacity *= 2;
        conv->messages = (Message*)realloc(conv->messages, conv->capacity * sizeof(Message));
    }

    // 添加消息
    conv->messages[conv->num_messages].role = str_dup(role);
    conv->messages[conv->num_messages].content = str_dup(content);
    conv->num_messages++;
}

void conversation_clear(Conversation* conv) {
    if (conv == NULL) return;

    // 释放消息内容
    for (int i = 0; i < conv->num_messages; i++) {
        free(conv->messages[i].role);
        free(conv->messages[i].content);
    }
    conv->num_messages = 0;
    // system_prompt 保留
}

void conversation_set_system(Conversation* conv, const char* system_prompt) {
    if (conv == NULL) return;
    free(conv->system_prompt);
    conv->system_prompt = str_dup(system_prompt);
}

int conversation_length(Conversation* conv) {
    if (conv == NULL) return 0;
    return conv->num_messages;
}

Message* conversation_get(Conversation* conv, int index) {
    if (conv == NULL || index < 0 || index >= conv->num_messages) return NULL;
    return &conv->messages[index];
}

Message* conversation_last(Conversation* conv) {
    if (conv == NULL || conv->num_messages == 0) return NULL;
    return &conv->messages[conv->num_messages - 1];
}

// ============ Prompt 格式化 ============

char* format_prompt(Conversation* conv) {
    if (conv == NULL) return NULL;

    // 估算缓冲区大小
    size_t total_len = 0;
    if (conv->system_prompt != NULL) {
        total_len += strlen(CHAT_SYSTEM_TAG) + strlen(conv->system_prompt) + 3;
    }
    for (int i = 0; i < conv->num_messages; i++) {
        total_len += strlen(conv->messages[i].role) + strlen(conv->messages[i].content) + 20;
    }
    total_len += 100;  // 额外空间

    char* prompt = (char*)malloc(total_len);
    prompt[0] = '\0';

    // 添加系统提示
    if (conv->system_prompt != NULL && strlen(conv->system_prompt) > 0) {
        strcat(prompt, CHAT_SYSTEM_TAG);
        strcat(prompt, "\n");
        strcat(prompt, conv->system_prompt);
        strcat(prompt, "\n");
    }

    // 添加对话历史
    for (int i = 0; i < conv->num_messages; i++) {
        Message* msg = &conv->messages[i];

        if (strcmp(msg->role, "user") == 0) {
            strcat(prompt, CHAT_USER_TAG);
        } else if (strcmp(msg->role, "assistant") == 0) {
            strcat(prompt, CHAT_ASSISTANT_TAG);
        } else if (strcmp(msg->role, "system") == 0) {
            strcat(prompt, CHAT_SYSTEM_TAG);
        }

        strcat(prompt, "\n");
        strcat(prompt, msg->content);
        strcat(prompt, "\n");
    }

    return prompt;
}

char* format_prompt_with_input(Conversation* conv, const char* user_input) {
    if (conv == NULL || user_input == NULL) return NULL;

    // 估算缓冲区大小
    size_t total_len = 0;
    if (conv->system_prompt != NULL) {
        total_len += strlen(CHAT_SYSTEM_TAG) + strlen(conv->system_prompt) + 3;
    }
    for (int i = 0; i < conv->num_messages; i++) {
        total_len += strlen(conv->messages[i].role) + strlen(conv->messages[i].content) + 20;
    }
    total_len += strlen(user_input) + strlen(CHAT_USER_TAG) + strlen(CHAT_ASSISTANT_TAG) + 100;

    char* prompt = (char*)malloc(total_len);
    prompt[0] = '\0';

    // 添加系统提示
    if (conv->system_prompt != NULL && strlen(conv->system_prompt) > 0) {
        strcat(prompt, CHAT_SYSTEM_TAG);
        strcat(prompt, "\n");
        strcat(prompt, conv->system_prompt);
        strcat(prompt, "\n");
    }

    // 添加对话历史
    for (int i = 0; i < conv->num_messages; i++) {
        Message* msg = &conv->messages[i];

        if (strcmp(msg->role, "user") == 0) {
            strcat(prompt, CHAT_USER_TAG);
        } else if (strcmp(msg->role, "assistant") == 0) {
            strcat(prompt, CHAT_ASSISTANT_TAG);
        } else if (strcmp(msg->role, "system") == 0) {
            strcat(prompt, CHAT_SYSTEM_TAG);
        }

        strcat(prompt, "\n");
        strcat(prompt, msg->content);
        strcat(prompt, "\n");
    }

    // 添加当前用户输入
    strcat(prompt, CHAT_USER_TAG);
    strcat(prompt, "\n");
    strcat(prompt, user_input);
    strcat(prompt, "\n");

    // 添加助手标签，准备生成
    strcat(prompt, CHAT_ASSISTANT_TAG);
    strcat(prompt, "\n");

    return prompt;
}

// ============ 配置 ============

ChatConfig default_chat_config(void) {
    ChatConfig config;
    config.gen_config = default_generate_config();
    config.gen_config.max_new_tokens = 200;
    config.gen_config.temperature = 0.7f;
    config.max_history = 20;  // 最多保留 20 条消息
    config.trim_history = 1;   // 自动裁剪
    return config;
}

// ============ 响应提取 ============

char* extract_response(const char* output) {
    if (output == NULL) return NULL;

    // 查找最后一个 <|assistant|> 后的内容
    const char* last_assistant = NULL;
    const char* ptr = output;

    while ((ptr = strstr(ptr, CHAT_ASSISTANT_TAG)) != NULL) {
        last_assistant = ptr;
        ptr += strlen(CHAT_ASSISTANT_TAG);
    }

    const char* response_start;
    if (last_assistant != NULL) {
        response_start = last_assistant + strlen(CHAT_ASSISTANT_TAG);
        // 跳过换行
        while (*response_start == '\n' || *response_start == '\r') {
            response_start++;
        }
    } else {
        response_start = output;
    }

    // 查找结束标记
    const char* response_end = response_start + strlen(response_start);

    // 查找下一个标签或结束
    const char* tags[] = {CHAT_USER_TAG, CHAT_ASSISTANT_TAG, CHAT_SYSTEM_TAG, CHAT_END_TAG};
    for (int i = 0; i < 4; i++) {
        const char* tag_pos = strstr(response_start, tags[i]);
        if (tag_pos != NULL && tag_pos < response_end) {
            response_end = tag_pos;
        }
    }

    // 复制响应
    size_t len = (size_t)(response_end - response_start);
    char* response = (char*)malloc(len + 1);
    memcpy(response, response_start, len);
    response[len] = '\0';

    // 裁剪空白
    char* trimmed = trim_string(response);
    free(response);

    return trimmed;
}

// ============ 对话函数 ============

// 裁剪历史消息
static void trim_conversation_history(Conversation* conv, int max_history) {
    if (conv == NULL || max_history <= 0) return;
    if (conv->num_messages <= max_history) return;

    int remove_count = conv->num_messages - max_history;

    // 释放要删除的消息
    for (int i = 0; i < remove_count; i++) {
        free(conv->messages[i].role);
        free(conv->messages[i].content);
    }

    // 移动剩余消息
    memmove(conv->messages, conv->messages + remove_count,
            (size_t)(conv->num_messages - remove_count) * sizeof(Message));

    conv->num_messages -= remove_count;
}

char* chat(
    GPTModel* model,
    Tokenizer* tokenizer,
    Conversation* conv,
    const char* user_input,
    ChatConfig* config
) {
    if (model == NULL || tokenizer == NULL || conv == NULL ||
        user_input == NULL || config == NULL) {
        return NULL;
    }

    // 裁剪历史
    if (config->trim_history && config->max_history > 0) {
        trim_conversation_history(conv, config->max_history - 1);  // 留一个位置给新消息
    }

    // 格式化 prompt
    char* prompt = format_prompt_with_input(conv, user_input);
    if (prompt == NULL) return NULL;

    // 生成回复
    char* full_output = generate(model, tokenizer, prompt, &config->gen_config);
    free(prompt);

    if (full_output == NULL) return NULL;

    // 提取助手回复
    char* response = extract_response(full_output);
    free(full_output);

    if (response == NULL) return NULL;

    // 更新对话历史
    conversation_add(conv, "user", user_input);
    conversation_add(conv, "assistant", response);

    // 返回复制的响应 (因为 response 已存储在 conv 中)
    char* result = str_dup(response);
    free(response);

    return result;
}

// 流式生成内部回调包装
typedef struct {
    ChatCallback user_callback;
    void* user_data;
    char* accumulated;
    size_t accum_len;
    size_t accum_cap;
} StreamChatContext;

static int stream_chat_callback(int token_id, const char* token_str, void* user_data) {
    (void)token_id;
    StreamChatContext* ctx = (StreamChatContext*)user_data;

    if (token_str == NULL) return 0;

    // 检查是否遇到结束标签
    // 累积 token
    size_t token_len = strlen(token_str);
    if (ctx->accum_len + token_len >= ctx->accum_cap) {
        ctx->accum_cap = (ctx->accum_len + token_len + 1) * 2;
        ctx->accumulated = (char*)realloc(ctx->accumulated, ctx->accum_cap);
    }
    memcpy(ctx->accumulated + ctx->accum_len, token_str, token_len);
    ctx->accum_len += token_len;
    ctx->accumulated[ctx->accum_len] = '\0';

    // 检查是否包含结束标签
    if (strstr(ctx->accumulated, CHAT_USER_TAG) != NULL ||
        strstr(ctx->accumulated, CHAT_END_TAG) != NULL) {
        return 1;  // 停止生成
    }

    // 调用用户回调
    if (ctx->user_callback != NULL) {
        return ctx->user_callback(token_str, ctx->user_data);
    }

    return 0;
}

int chat_stream(
    GPTModel* model,
    Tokenizer* tokenizer,
    Conversation* conv,
    const char* user_input,
    ChatConfig* config,
    ChatCallback callback,
    void* user_data
) {
    if (model == NULL || tokenizer == NULL || conv == NULL ||
        user_input == NULL || config == NULL) {
        return -1;
    }

    // 裁剪历史
    if (config->trim_history && config->max_history > 0) {
        trim_conversation_history(conv, config->max_history - 1);
    }

    // 格式化 prompt
    char* prompt = format_prompt_with_input(conv, user_input);
    if (prompt == NULL) return -1;

    // 设置流式上下文
    StreamChatContext ctx;
    ctx.user_callback = callback;
    ctx.user_data = user_data;
    ctx.accum_cap = BUFFER_SIZE;
    ctx.accumulated = (char*)malloc(ctx.accum_cap);
    ctx.accumulated[0] = '\0';
    ctx.accum_len = 0;

    // 流式生成
    int generated = generate_stream(model, tokenizer, prompt, &config->gen_config,
                                   stream_chat_callback, &ctx);

    free(prompt);

    // 提取并保存回复
    if (ctx.accum_len > 0) {
        char* response = trim_string(ctx.accumulated);

        // 移除可能的结束标签
        char* tag_pos = strstr(response, CHAT_USER_TAG);
        if (tag_pos != NULL) *tag_pos = '\0';
        tag_pos = strstr(response, CHAT_END_TAG);
        if (tag_pos != NULL) *tag_pos = '\0';

        char* clean_response = trim_string(response);
        free(response);

        // 更新对话历史
        conversation_add(conv, "user", user_input);
        conversation_add(conv, "assistant", clean_response);

        free(clean_response);
    }

    free(ctx.accumulated);

    return generated;
}
