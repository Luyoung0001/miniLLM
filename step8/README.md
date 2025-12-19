# Step 8: Chat - 对话系统实现

本步骤实现完整的对话系统，支持多轮对话和上下文管理。

## 目标

1. 实现对话消息管理
2. 实现 Prompt 格式化
3. 实现多轮对话
4. 支持系统提示
5. 实现流式对话
6. 实现历史消息裁剪

## 知识铺垫

### 对话格式

语言模型对话需要特定的格式来区分用户和助手的消息：

```
<|system|>
你是一个有帮助的助手。
<|user|>
你好
<|assistant|>
你好！有什么可以帮助你的？
<|user|>
今天天气怎么样？
<|assistant|>
{模型在这里生成回复}
```

### 多轮对话

多轮对话需要：
1. 维护对话历史
2. 将历史格式化为模型输入
3. 从输出中提取助手回复
4. 更新对话历史

### 上下文管理

由于模型有最大序列长度限制，需要：
1. 限制历史消息数量
2. 自动裁剪旧消息
3. 保留系统提示

## 文件结构

```
step8/
├── Makefile
├── README.md
├── src/
│   ├── ... (继承 step7)
│   ├── chat.h / chat.c   # 对话系统
│   └── main.c
└── models/
    └── minillm.bin       # 预训练模型 (可选)
```

## 核心 API

### Message 和 Conversation

```c
// 单条消息
typedef struct {
    char* role;     // "user", "assistant", "system"
    char* content;  // 消息内容
} Message;

// 对话历史
typedef struct {
    Message* messages;      // 消息数组
    int num_messages;       // 当前消息数
    int capacity;           // 数组容量
    char* system_prompt;    // 系统提示 (可选)
} Conversation;

// 对话管理
Conversation* conversation_create(void);
void conversation_free(Conversation* conv);
void conversation_add(Conversation* conv, const char* role, const char* content);
void conversation_clear(Conversation* conv);
void conversation_set_system(Conversation* conv, const char* system_prompt);
```

### Prompt 格式化

```c
// 格式化对话历史为模型输入
char* format_prompt(Conversation* conv);

// 格式化并添加用户输入，准备生成
char* format_prompt_with_input(Conversation* conv, const char* user_input);

// 从模型输出中提取助手回复
char* extract_response(const char* output);
```

### Chat 函数

```c
// 对话配置
typedef struct {
    GenerateConfig gen_config;  // 生成配置
    int max_history;            // 最大历史消息数
    int trim_history;           // 是否自动裁剪
} ChatConfig;

ChatConfig default_chat_config(void);

// 进行一轮对话
char* chat(
    GPTModel* model,
    Tokenizer* tokenizer,
    Conversation* conv,
    const char* user_input,
    ChatConfig* config
);

// 流式对话
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
```

## 使用示例

### 基本对话

```c
#include "chat.h"

int main() {
    // 加载模型
    GPTModel* model = model_load("models/minillm.bin");
    Tokenizer* tok = tokenizer_create();

    // 创建对话
    Conversation* conv = conversation_create();
    conversation_set_system(conv, "你是一个有帮助的助手。");

    // 对话配置
    ChatConfig config = default_chat_config();
    config.gen_config.max_new_tokens = 100;

    // 第一轮对话
    char* response1 = chat(model, tok, conv, "你好", &config);
    printf("助手: %s\n", response1);
    free(response1);

    // 第二轮对话 (自动保留上下文)
    char* response2 = chat(model, tok, conv, "今天天气怎么样？", &config);
    printf("助手: %s\n", response2);
    free(response2);

    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
    return 0;
}
```

### 交互式对话

```c
int main() {
    GPTModel* model = model_load("models/minillm.bin");
    Tokenizer* tok = tokenizer_create();
    Conversation* conv = conversation_create();
    ChatConfig config = default_chat_config();

    char input[1024];
    printf("miniLLM Chat (输入 'quit' 退出)\n\n");

    while (1) {
        printf("You: ");
        if (fgets(input, sizeof(input), stdin) == NULL) break;

        // 移除换行符
        input[strcspn(input, "\n")] = 0;

        if (strcmp(input, "quit") == 0) break;
        if (strcmp(input, "clear") == 0) {
            conversation_clear(conv);
            printf("对话已清空\n\n");
            continue;
        }

        char* response = chat(model, tok, conv, input, &config);
        printf("Assistant: %s\n\n", response);
        free(response);
    }

    conversation_free(conv);
    tokenizer_free(tok);
    model_free(model);
    return 0;
}
```

### 流式对话

```c
int print_token(const char* token, void* user_data) {
    (void)user_data;
    printf("%s", token);
    fflush(stdout);
    return 0;  // 继续生成
}

int main() {
    // ... 初始化模型和对话 ...

    printf("Assistant: ");
    chat_stream(model, tok, conv, "给我讲个故事", &config, print_token, NULL);
    printf("\n");

    // ... 清理 ...
}
```

## 对话格式详解

### 标签定义

```c
#define CHAT_USER_TAG       "<|user|>"
#define CHAT_ASSISTANT_TAG  "<|assistant|>"
#define CHAT_SYSTEM_TAG     "<|system|>"
#define CHAT_END_TAG        "<|end|>"
```

### 格式化示例

输入对话历史：
```
系统: 你是一个有帮助的助手。
用户: 你好
助手: 你好！
用户: 谢谢
```

格式化输出：
```
<|system|>
你是一个有帮助的助手。
<|user|>
你好
<|assistant|>
你好！
<|user|>
谢谢
<|assistant|>
```

模型在最后的 `<|assistant|>` 后生成回复。

## 历史裁剪

当对话过长时，需要裁剪历史以适应模型的最大序列长度：

```c
ChatConfig config = default_chat_config();
config.max_history = 10;    // 最多保留 10 条消息
config.trim_history = 1;    // 启用自动裁剪

// 进行多轮对话，旧消息会被自动删除
```

裁剪策略：
- 保留最近的 N 条消息
- 系统提示不会被裁剪
- 总是保留完整的消息对 (用户 + 助手)

## 编译和运行

```bash
make        # 编译
make test   # 运行测试
make clean  # 清理
```

## 预期输出

```
========================================
      miniLLM Step8 测试套件
        对话系统测试
========================================

=== 测试: 对话创建 ===
[PASS] 创建对话成功
[PASS] 新对话消息数为 0

=== 测试: 添加消息 ===
[PASS] 添加后消息数为 1
[PASS] 获取消息成功
[PASS] 消息角色正确
[PASS] 消息内容正确
...

=== 测试: Prompt 格式化 ===
[PASS] 格式化 prompt 成功
[PASS] 包含 user 标签
[PASS] 包含 assistant 标签
...

=== 测试: 基本对话 ===
[PASS] 对话生成成功
[PASS] 对话历史有 2 条消息
  用户: 你好
  助手: ...

=== 测试: 多轮对话 ===
[PASS] 第一轮对话成功
[PASS] 第二轮对话成功
[PASS] 第三轮对话成功
...

=== 测试: 流式对话 ===
[PASS] 流式对话生成 > 0 tokens
[PASS] 回调被调用 5 次
[PASS] 对话历史已更新

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 对话创建和管理正确
- [ ] 消息添加和清空正确
- [ ] 系统提示正确设置和保留
- [ ] Prompt 格式化正确
- [ ] 响应提取正确
- [ ] 多轮对话历史保持正确
- [ ] 历史裁剪工作正常
- [ ] 流式对话正确调用回调
- [ ] 无内存泄漏

## 下一步

完成 Step 8 后，可以继续 Step 9 实现 HTTP 服务（REST API 接口）。
