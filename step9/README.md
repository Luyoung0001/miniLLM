# miniLLM Step9: HTTP 服务

本步骤实现了 HTTP 服务器，提供 OpenAI 兼容的 REST API 接口。

## 新增功能

### HTTP 服务器 (`http_server.h/c`)

- **POSIX Socket 服务器**: 基于 TCP 的简单 HTTP 服务器
- **OpenAI 兼容 API**: `/v1/chat/completions` 端点
- **JSON 解析**: 轻量级 JSON 解析器 (无外部依赖)
- **CORS 支持**: 支持跨域请求
- **信号处理**: 支持 Ctrl+C 优雅关闭

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/v1/chat/completions` | 对话补全 (OpenAI 兼容) |

## 编译

```bash
make
```

## 运行

### 运行测试
```bash
make test
# 或
./step9 --test
```

### 启动服务器
```bash
make serve
# 或
./step9 --serve

# 指定端口
./step9 --serve --port 3000
```

### 帮助
```bash
./step9 --help
```

## API 使用

### 健康检查

```bash
curl http://localhost:8080/health
```

响应:
```json
{"status": "ok", "model": "miniLLM"}
```

### 对话补全

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "Hello"}
       ],
       "temperature": 0.7,
       "max_tokens": 50
     }'
```

响应:
```json
{
  "id": "chatcmpl-miniLLM",
  "object": "chat.completion",
  "model": "miniLLM",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | array | 必填 | 消息数组 |
| temperature | float | 1.0 | 采样温度 |
| max_tokens | int | 100 | 最大生成 token 数 |
| top_k | int | 40 | Top-K 采样 |
| top_p | float | 0.9 | Top-P 采样 |

## 架构

```
┌─────────────────────────────────────────────────┐
│                   客户端                         │
│              (curl / 浏览器)                     │
└─────────────────┬───────────────────────────────┘
                  │ HTTP Request
                  ▼
┌─────────────────────────────────────────────────┐
│              HTTP Server                         │
│  ┌─────────────────────────────────────────┐    │
│  │          parse_http_request()           │    │
│  └─────────────────┬───────────────────────┘    │
│                    ▼                             │
│  ┌─────────────────────────────────────────┐    │
│  │              路由分发                    │    │
│  │  /health → handle_health_check()        │    │
│  │  /v1/chat/completions                   │    │
│  │         → handle_chat_completion()      │    │
│  └─────────────────┬───────────────────────┘    │
└────────────────────┼────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│              Chat System (Step8)                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐    │
│  │Conversation│ │ Tokenizer │ │   Model    │    │
│  └──────────┘  └──────────┘  └────────────┘    │
└─────────────────────────────────────────────────┘
```

## 文件结构

```
step9/
├── Makefile
├── README.md
└── src/
    ├── http_server.h    # HTTP 服务器头文件
    ├── http_server.c    # HTTP 服务器实现
    ├── chat.h/c         # 对话系统 (Step8)
    ├── generate.h/c     # 文本生成 (Step7)
    ├── train.h/c        # 训练系统 (Step6)
    ├── model.h/c        # 模型定义 (Step5)
    ├── transformer.h/c  # Transformer 层 (Step4)
    ├── attention.h/c    # 注意力机制 (Step3)
    ├── embedding.h/c    # 嵌入层 (Step2)
    ├── tokenizer.h/c    # 分词器 (Step1)
    ├── tensor.h/c       # 张量库 (Step0)
    └── main.c           # 主程序入口
```

## 测试结果

运行 `make test` 将执行以下测试:

1. **对话管理测试** - 创建/添加/清空对话
2. **Prompt 格式化测试** - 消息格式化
3. **辅助函数测试** - 字符串处理
4. **对话功能测试** - 基本对话/多轮对话
5. **流式对话测试** - 流式生成
6. **HTTP 服务器测试** - 配置/JSON解析/请求处理

## 注意事项

- 服务器为单线程实现，适合开发和测试
- 生产环境建议使用多线程或异步 I/O
- 当前使用随机初始化的模型，实际应用需加载训练好的权重
