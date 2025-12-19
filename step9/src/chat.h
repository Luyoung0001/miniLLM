#ifndef CHAT_H
#define CHAT_H

#include "model.h"
#include "tokenizer.h"
#include "generate.h"

/**
 * 对话系统模块
 *
 * 支持多轮对话，管理对话历史，格式化 prompt
 *
 * 对话格式:
 * <|user|>
 * 你好
 * <|assistant|>
 * 你好！有什么可以帮助你的？
 */

// ============ 消息结构 ============

/**
 * 单条消息
 */
typedef struct {
    char* role;     // "user" 或 "assistant" 或 "system"
    char* content;  // 消息内容
} Message;

/**
 * 对话历史
 */
typedef struct {
    Message* messages;      // 消息数组
    int num_messages;       // 当前消息数
    int capacity;           // 数组容量
    char* system_prompt;    // 系统提示 (可选)
} Conversation;

// ============ 对话管理 ============

/**
 * 创建对话
 */
Conversation* conversation_create(void);

/**
 * 释放对话
 */
void conversation_free(Conversation* conv);

/**
 * 添加消息
 * @param conv 对话
 * @param role 角色 ("user", "assistant", "system")
 * @param content 内容
 */
void conversation_add(Conversation* conv, const char* role, const char* content);

/**
 * 清空对话历史 (保留系统提示)
 */
void conversation_clear(Conversation* conv);

/**
 * 设置系统提示
 */
void conversation_set_system(Conversation* conv, const char* system_prompt);

/**
 * 获取消息数量
 */
int conversation_length(Conversation* conv);

/**
 * 获取指定位置的消息
 */
Message* conversation_get(Conversation* conv, int index);

/**
 * 获取最后一条消息
 */
Message* conversation_last(Conversation* conv);

// ============ Prompt 格式化 ============

/**
 * 格式化对话历史为模型输入
 * @param conv 对话
 * @return 格式化的 prompt 字符串 (需要 free)
 *
 * 格式:
 * <|system|>
 * {system_prompt}
 * <|user|>
 * {user_message}
 * <|assistant|>
 * {assistant_message}
 * ...
 * <|assistant|>
 */
char* format_prompt(Conversation* conv);

/**
 * 格式化对话历史并添加用户输入
 * @param conv 对话
 * @param user_input 用户输入
 * @return 格式化的 prompt 字符串 (需要 free)
 */
char* format_prompt_with_input(Conversation* conv, const char* user_input);

// ============ 对话函数 ============

/**
 * 对话配置
 */
typedef struct {
    GenerateConfig gen_config;  // 生成配置
    int max_history;            // 最大历史消息数 (0 = 不限制)
    int trim_history;           // 超过限制时是否自动裁剪
} ChatConfig;

/**
 * 创建默认对话配置
 */
ChatConfig default_chat_config(void);

/**
 * 进行一轮对话
 * @param model 模型
 * @param tokenizer tokenizer
 * @param conv 对话 (会被修改，添加用户输入和助手回复)
 * @param user_input 用户输入
 * @param config 对话配置
 * @return 助手回复 (需要 free)
 */
char* chat(
    GPTModel* model,
    Tokenizer* tokenizer,
    Conversation* conv,
    const char* user_input,
    ChatConfig* config
);

/**
 * 流式对话
 * @param callback 回调函数，返回 0 继续，非 0 停止
 * @return 生成的 token 数
 */
typedef int (*ChatCallback)(const char* token, void* user_data);

int chat_stream(
    GPTModel* model,
    Tokenizer* tokenizer,
    Conversation* conv,
    const char* user_input,
    ChatConfig* config,
    ChatCallback callback,
    void* user_data
);

// ============ 辅助函数 ============

/**
 * 解析模型输出，提取助手回复
 * @param output 模型输出
 * @return 提取的回复 (需要 free)
 */
char* extract_response(const char* output);

/**
 * 裁剪字符串两端空白
 * @param str 字符串
 * @return 裁剪后的字符串 (需要 free)
 */
char* trim_string(const char* str);

/**
 * 检查字符串是否以指定前缀开头
 */
int starts_with(const char* str, const char* prefix);

/**
 * 检查字符串是否以指定后缀结尾
 */
int ends_with(const char* str, const char* suffix);

// ============ 特殊 Token ============

#define CHAT_USER_TAG       "<|user|>"
#define CHAT_ASSISTANT_TAG  "<|assistant|>"
#define CHAT_SYSTEM_TAG     "<|system|>"
#define CHAT_END_TAG        "<|end|>"

#endif // CHAT_H
