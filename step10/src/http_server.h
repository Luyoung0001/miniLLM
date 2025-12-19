#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include "model.h"
#include "tokenizer.h"
#include "chat.h"

/**
 * 简单 HTTP 服务器模块
 *
 * 提供 REST API 接口，兼容 OpenAI API 格式
 *
 * 支持的端点:
 * - POST /v1/chat/completions - 对话补全
 * - GET /health - 健康检查
 */

// ============ 服务器配置 ============

typedef struct {
    int port;               // 监听端口 (默认 8080)
    int max_connections;    // 最大并发连接数 (默认 10)
    int request_timeout;    // 请求超时 (秒, 默认 30)
    int buffer_size;        // 缓冲区大小 (默认 65536)
} ServerConfig;

/**
 * 创建默认服务器配置
 */
ServerConfig default_server_config(void);

// ============ HTTP 服务器 ============

typedef struct HttpServer HttpServer;

/**
 * 创建 HTTP 服务器
 * @param model 模型
 * @param tokenizer tokenizer
 * @param config 服务器配置
 * @return 服务器实例
 */
HttpServer* http_server_create(
    GPTModel* model,
    Tokenizer* tokenizer,
    ServerConfig* config
);

/**
 * 释放 HTTP 服务器
 */
void http_server_free(HttpServer* server);

/**
 * 启动服务器 (阻塞)
 * @return 0 成功, -1 失败
 */
int http_server_start(HttpServer* server);

/**
 * 停止服务器
 */
void http_server_stop(HttpServer* server);

/**
 * 检查服务器是否在运行
 */
int http_server_is_running(HttpServer* server);

// ============ 请求/响应结构 ============

/**
 * HTTP 请求
 */
typedef struct {
    char* method;           // GET, POST, etc.
    char* path;             // /v1/chat/completions
    char* body;             // 请求体
    int body_len;           // 请求体长度
    char* content_type;     // Content-Type
} HttpRequest;

/**
 * HTTP 响应
 */
typedef struct {
    int status_code;        // 200, 400, 500, etc.
    char* status_text;      // OK, Bad Request, etc.
    char* content_type;     // application/json
    char* body;             // 响应体
    int body_len;           // 响应体长度
} HttpResponse;

/**
 * 释放请求
 */
void http_request_free(HttpRequest* req);

/**
 * 释放响应
 */
void http_response_free(HttpResponse* resp);

// ============ JSON 解析 (简单实现) ============

/**
 * 聊天补全请求
 */
typedef struct {
    char** roles;           // 消息角色数组
    char** contents;        // 消息内容数组
    int num_messages;       // 消息数量
    float temperature;      // 温度 (默认 1.0)
    int max_tokens;         // 最大 token 数 (默认 100)
    int top_k;              // Top-K (默认 40)
    float top_p;            // Top-P (默认 0.9)
} ChatCompletionRequest;

/**
 * 聊天补全响应
 */
typedef struct {
    char* role;             // assistant
    char* content;          // 回复内容
    int prompt_tokens;      // prompt token 数
    int completion_tokens;  // 补全 token 数
    int total_tokens;       // 总 token 数
} ChatCompletionResponse;

/**
 * 解析聊天补全请求
 * @param json JSON 字符串
 * @return 解析后的请求 (需要释放)
 */
ChatCompletionRequest* parse_chat_completion_request(const char* json);

/**
 * 释放聊天补全请求
 */
void chat_completion_request_free(ChatCompletionRequest* req);

/**
 * 生成聊天补全响应 JSON
 * @param resp 响应数据
 * @return JSON 字符串 (需要 free)
 */
char* format_chat_completion_response(ChatCompletionResponse* resp);

/**
 * 生成错误响应 JSON
 * @param error_msg 错误消息
 * @return JSON 字符串 (需要 free)
 */
char* format_error_response(const char* error_msg);

// ============ 辅助函数 ============

/**
 * URL 解码
 */
char* url_decode(const char* str);

/**
 * 从 JSON 中提取字符串值
 * @param json JSON 字符串
 * @param key 键名
 * @return 值 (需要 free), 或 NULL
 */
char* json_get_string(const char* json, const char* key);

/**
 * 从 JSON 中提取数字值
 */
float json_get_number(const char* json, const char* key, float default_val);

/**
 * 从 JSON 中提取整数值
 */
int json_get_int(const char* json, const char* key, int default_val);

#endif // HTTP_SERVER_H
