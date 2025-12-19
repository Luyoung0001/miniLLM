#include "http_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>

#define DEFAULT_PORT 8080
#define DEFAULT_MAX_CONNECTIONS 10
#define DEFAULT_TIMEOUT 30
#define DEFAULT_BUFFER_SIZE 65536

// ============ 服务器结构 ============

struct HttpServer {
    int socket_fd;
    int running;
    GPTModel* model;
    Tokenizer* tokenizer;
    ServerConfig config;
    ChatConfig chat_config;
};

// 全局服务器指针 (用于信号处理)
static HttpServer* g_server = NULL;

// ============ 配置 ============

ServerConfig default_server_config(void) {
    ServerConfig config;
    config.port = DEFAULT_PORT;
    config.max_connections = DEFAULT_MAX_CONNECTIONS;
    config.request_timeout = DEFAULT_TIMEOUT;
    config.buffer_size = DEFAULT_BUFFER_SIZE;
    return config;
}

// ============ 辅助函数 ============

static char* str_dup(const char* str) {
    if (str == NULL) return NULL;
    size_t len = strlen(str);
    char* result = (char*)malloc(len + 1);
    memcpy(result, str, len + 1);
    return result;
}

static char* str_trim(char* str) {
    if (str == NULL) return NULL;
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0') return str;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return str;
}

char* url_decode(const char* str) {
    if (str == NULL) return NULL;
    size_t len = strlen(str);
    char* result = (char*)malloc(len + 1);
    char* dst = result;

    while (*str) {
        if (*str == '%' && str[1] && str[2]) {
            char hex[3] = {str[1], str[2], 0};
            *dst++ = (char)strtol(hex, NULL, 16);
            str += 3;
        } else if (*str == '+') {
            *dst++ = ' ';
            str++;
        } else {
            *dst++ = *str++;
        }
    }
    *dst = '\0';
    return result;
}

// ============ JSON 解析 (简单实现) ============

char* json_get_string(const char* json, const char* key) {
    if (json == NULL || key == NULL) return NULL;

    // 构造搜索模式 "key":
    size_t key_len = strlen(key);
    char* pattern = (char*)malloc(key_len + 5);
    sprintf(pattern, "\"%s\"", key);

    const char* pos = strstr(json, pattern);
    free(pattern);

    if (pos == NULL) return NULL;

    // 跳到冒号后
    pos = strchr(pos, ':');
    if (pos == NULL) return NULL;
    pos++;

    // 跳过空白
    while (*pos && isspace((unsigned char)*pos)) pos++;

    if (*pos != '"') return NULL;
    pos++;  // 跳过引号

    // 找到结束引号
    const char* end = pos;
    while (*end && *end != '"') {
        if (*end == '\\' && end[1]) end++;  // 跳过转义
        end++;
    }

    size_t len = (size_t)(end - pos);
    char* result = (char*)malloc(len + 1);
    memcpy(result, pos, len);
    result[len] = '\0';

    return result;
}

float json_get_number(const char* json, const char* key, float default_val) {
    if (json == NULL || key == NULL) return default_val;

    size_t key_len = strlen(key);
    char* pattern = (char*)malloc(key_len + 5);
    sprintf(pattern, "\"%s\"", key);

    const char* pos = strstr(json, pattern);
    free(pattern);

    if (pos == NULL) return default_val;

    pos = strchr(pos, ':');
    if (pos == NULL) return default_val;
    pos++;

    while (*pos && isspace((unsigned char)*pos)) pos++;

    char* endptr;
    float val = strtof(pos, &endptr);
    if (endptr == pos) return default_val;

    return val;
}

int json_get_int(const char* json, const char* key, int default_val) {
    return (int)json_get_number(json, key, (float)default_val);
}

// ============ 请求解析 ============

ChatCompletionRequest* parse_chat_completion_request(const char* json) {
    if (json == NULL) return NULL;

    ChatCompletionRequest* req = (ChatCompletionRequest*)malloc(sizeof(ChatCompletionRequest));
    memset(req, 0, sizeof(ChatCompletionRequest));

    // 解析参数
    req->temperature = json_get_number(json, "temperature", 1.0f);
    req->max_tokens = json_get_int(json, "max_tokens", 100);
    req->top_k = json_get_int(json, "top_k", 40);
    req->top_p = json_get_number(json, "top_p", 0.9f);

    // 解析 messages 数组
    const char* messages_start = strstr(json, "\"messages\"");
    if (messages_start == NULL) {
        free(req);
        return NULL;
    }

    messages_start = strchr(messages_start, '[');
    if (messages_start == NULL) {
        free(req);
        return NULL;
    }
    messages_start++;

    // 计算消息数量
    int count = 0;
    int capacity = 16;
    req->roles = (char**)malloc(capacity * sizeof(char*));
    req->contents = (char**)malloc(capacity * sizeof(char*));

    const char* ptr = messages_start;
    while (*ptr) {
        // 找到下一个对象
        const char* obj_start = strchr(ptr, '{');
        if (obj_start == NULL) break;

        const char* obj_end = strchr(obj_start, '}');
        if (obj_end == NULL) break;

        // 提取这个对象
        size_t obj_len = (size_t)(obj_end - obj_start + 1);
        char* obj = (char*)malloc(obj_len + 1);
        memcpy(obj, obj_start, obj_len);
        obj[obj_len] = '\0';

        char* role = json_get_string(obj, "role");
        char* content = json_get_string(obj, "content");
        free(obj);

        if (role != NULL && content != NULL) {
            if (count >= capacity) {
                capacity *= 2;
                req->roles = (char**)realloc(req->roles, capacity * sizeof(char*));
                req->contents = (char**)realloc(req->contents, capacity * sizeof(char*));
            }
            req->roles[count] = role;
            req->contents[count] = content;
            count++;
        } else {
            free(role);
            free(content);
        }

        ptr = obj_end + 1;

        // 检查是否到达数组末尾
        while (*ptr && isspace((unsigned char)*ptr)) ptr++;
        if (*ptr == ']') break;
    }

    req->num_messages = count;
    return req;
}

void chat_completion_request_free(ChatCompletionRequest* req) {
    if (req == NULL) return;
    for (int i = 0; i < req->num_messages; i++) {
        free(req->roles[i]);
        free(req->contents[i]);
    }
    free(req->roles);
    free(req->contents);
    free(req);
}

// ============ 响应生成 ============

char* format_chat_completion_response(ChatCompletionResponse* resp) {
    if (resp == NULL) return NULL;

    // 转义内容中的特殊字符
    size_t content_len = resp->content ? strlen(resp->content) : 0;
    char* escaped_content = (char*)malloc(content_len * 2 + 1);
    char* dst = escaped_content;
    const char* src = resp->content ? resp->content : "";

    while (*src) {
        if (*src == '"') {
            *dst++ = '\\';
            *dst++ = '"';
        } else if (*src == '\\') {
            *dst++ = '\\';
            *dst++ = '\\';
        } else if (*src == '\n') {
            *dst++ = '\\';
            *dst++ = 'n';
        } else if (*src == '\r') {
            *dst++ = '\\';
            *dst++ = 'r';
        } else if (*src == '\t') {
            *dst++ = '\\';
            *dst++ = 't';
        } else {
            *dst++ = *src;
        }
        src++;
    }
    *dst = '\0';

    size_t buffer_size = strlen(escaped_content) + 512;
    char* json = (char*)malloc(buffer_size);

    snprintf(json, buffer_size,
        "{\n"
        "  \"id\": \"chatcmpl-miniLLM\",\n"
        "  \"object\": \"chat.completion\",\n"
        "  \"model\": \"miniLLM\",\n"
        "  \"choices\": [\n"
        "    {\n"
        "      \"index\": 0,\n"
        "      \"message\": {\n"
        "        \"role\": \"assistant\",\n"
        "        \"content\": \"%s\"\n"
        "      },\n"
        "      \"finish_reason\": \"stop\"\n"
        "    }\n"
        "  ],\n"
        "  \"usage\": {\n"
        "    \"prompt_tokens\": %d,\n"
        "    \"completion_tokens\": %d,\n"
        "    \"total_tokens\": %d\n"
        "  }\n"
        "}",
        escaped_content,
        resp->prompt_tokens,
        resp->completion_tokens,
        resp->total_tokens
    );

    free(escaped_content);
    return json;
}

char* format_error_response(const char* error_msg) {
    size_t buffer_size = strlen(error_msg) + 256;
    char* json = (char*)malloc(buffer_size);

    snprintf(json, buffer_size,
        "{\n"
        "  \"error\": {\n"
        "    \"message\": \"%s\",\n"
        "    \"type\": \"invalid_request_error\"\n"
        "  }\n"
        "}",
        error_msg
    );

    return json;
}

// ============ HTTP 解析 ============

static HttpRequest* parse_http_request(const char* raw, int len) {
    if (raw == NULL || len <= 0) return NULL;

    HttpRequest* req = (HttpRequest*)malloc(sizeof(HttpRequest));
    memset(req, 0, sizeof(HttpRequest));

    // 复制请求以便修改
    char* buffer = (char*)malloc(len + 1);
    memcpy(buffer, raw, len);
    buffer[len] = '\0';

    // 解析请求行
    char* line_end = strstr(buffer, "\r\n");
    if (line_end == NULL) {
        free(buffer);
        free(req);
        return NULL;
    }
    *line_end = '\0';

    // 解析 METHOD PATH HTTP/1.x
    char* method = buffer;
    char* path = strchr(method, ' ');
    if (path == NULL) {
        free(buffer);
        free(req);
        return NULL;
    }
    *path++ = '\0';

    char* version = strchr(path, ' ');
    if (version) *version = '\0';

    req->method = str_dup(method);
    req->path = str_dup(path);

    // 解析头部
    char* headers_start = line_end + 2;
    char* body_start = strstr(headers_start, "\r\n\r\n");

    if (body_start) {
        *body_start = '\0';
        body_start += 4;

        // 计算 body 长度
        req->body_len = len - (int)(body_start - buffer);
        if (req->body_len > 0) {
            req->body = (char*)malloc(req->body_len + 1);
            memcpy(req->body, raw + (body_start - buffer), req->body_len);
            req->body[req->body_len] = '\0';
        }
    }

    // 解析 Content-Type
    char* ct = strstr(headers_start, "Content-Type:");
    if (ct == NULL) ct = strstr(headers_start, "content-type:");
    if (ct) {
        ct += 13;  // 跳过 "Content-Type:"
        while (*ct && isspace((unsigned char)*ct)) ct++;
        char* ct_end = strstr(ct, "\r\n");
        if (ct_end) {
            size_t ct_len = (size_t)(ct_end - ct);
            req->content_type = (char*)malloc(ct_len + 1);
            memcpy(req->content_type, ct, ct_len);
            req->content_type[ct_len] = '\0';
        }
    }

    free(buffer);
    return req;
}

void http_request_free(HttpRequest* req) {
    if (req == NULL) return;
    free(req->method);
    free(req->path);
    free(req->body);
    free(req->content_type);
    free(req);
}

void http_response_free(HttpResponse* resp) {
    if (resp == NULL) return;
    free(resp->status_text);
    free(resp->content_type);
    free(resp->body);
    free(resp);
}

// ============ HTTP 响应发送 ============

static void send_http_response(int client_fd, int status_code, const char* status_text,
                               const char* content_type, const char* body) {
    char header[1024];
    int body_len = body ? (int)strlen(body) : 0;

    snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text,
        content_type,
        body_len
    );

    send(client_fd, header, strlen(header), 0);
    if (body && body_len > 0) {
        send(client_fd, body, body_len, 0);
    }
}

// ============ 请求处理 ============

static void handle_health_check(int client_fd) {
    const char* body = "{\"status\": \"ok\", \"model\": \"miniLLM\"}";
    send_http_response(client_fd, 200, "OK", "application/json", body);
}

static void handle_chat_completion(HttpServer* server, int client_fd, HttpRequest* req) {
    if (req->body == NULL || req->body_len == 0) {
        char* error = format_error_response("Request body is empty");
        send_http_response(client_fd, 400, "Bad Request", "application/json", error);
        free(error);
        return;
    }

    // 解析请求
    ChatCompletionRequest* chat_req = parse_chat_completion_request(req->body);
    if (chat_req == NULL || chat_req->num_messages == 0) {
        char* error = format_error_response("Invalid request: messages array is required");
        send_http_response(client_fd, 400, "Bad Request", "application/json", error);
        free(error);
        chat_completion_request_free(chat_req);
        return;
    }

    // 创建对话
    Conversation* conv = conversation_create();

    // 添加消息到对话
    for (int i = 0; i < chat_req->num_messages; i++) {
        conversation_add(conv, chat_req->roles[i], chat_req->contents[i]);
    }

    // 配置生成参数
    ChatConfig config = server->chat_config;
    config.gen_config.temperature = chat_req->temperature;
    config.gen_config.max_new_tokens = chat_req->max_tokens;
    config.gen_config.top_k = chat_req->top_k;
    config.gen_config.top_p = chat_req->top_p;

    // 获取最后一条用户消息
    const char* last_user_msg = "";
    for (int i = chat_req->num_messages - 1; i >= 0; i--) {
        if (strcmp(chat_req->roles[i], "user") == 0) {
            last_user_msg = chat_req->contents[i];
            // 移除最后一条消息（因为 chat() 会自动添加）
            free(conv->messages[conv->num_messages - 1].role);
            free(conv->messages[conv->num_messages - 1].content);
            conv->num_messages--;
            break;
        }
    }

    // 生成回复
    char* response = chat(server->model, server->tokenizer, conv, last_user_msg, &config);

    if (response == NULL) {
        char* error = format_error_response("Failed to generate response");
        send_http_response(client_fd, 500, "Internal Server Error", "application/json", error);
        free(error);
    } else {
        // 构造响应
        ChatCompletionResponse resp;
        resp.role = "assistant";
        resp.content = response;
        resp.prompt_tokens = 0;  // 简化实现
        resp.completion_tokens = (int)strlen(response);
        resp.total_tokens = resp.completion_tokens;

        char* json = format_chat_completion_response(&resp);
        send_http_response(client_fd, 200, "OK", "application/json", json);

        free(json);
        free(response);
    }

    conversation_free(conv);
    chat_completion_request_free(chat_req);
}

static void handle_options(int client_fd) {
    send_http_response(client_fd, 204, "No Content", "text/plain", NULL);
}

static void handle_not_found(int client_fd) {
    char* error = format_error_response("Endpoint not found");
    send_http_response(client_fd, 404, "Not Found", "application/json", error);
    free(error);
}

static void handle_request(HttpServer* server, int client_fd) {
    char* buffer = (char*)malloc(server->config.buffer_size);

    // 读取请求
    int total_read = 0;
    int bytes_read;

    while ((bytes_read = recv(client_fd, buffer + total_read,
                              server->config.buffer_size - total_read - 1, 0)) > 0) {
        total_read += bytes_read;

        // 检查是否读完 (简单检查 Content-Length)
        buffer[total_read] = '\0';
        if (strstr(buffer, "\r\n\r\n") != NULL) {
            // 检查是否有 body
            char* cl = strstr(buffer, "Content-Length:");
            if (cl == NULL) cl = strstr(buffer, "content-length:");

            if (cl == NULL) {
                break;  // 没有 body
            }

            int content_length = atoi(cl + 15);
            char* body_start = strstr(buffer, "\r\n\r\n");
            if (body_start) {
                int header_len = (int)(body_start - buffer + 4);
                int body_read = total_read - header_len;
                if (body_read >= content_length) {
                    break;  // body 读完
                }
            }
        }

        if (total_read >= server->config.buffer_size - 1) break;
    }

    if (total_read <= 0) {
        free(buffer);
        return;
    }

    // 解析请求
    HttpRequest* req = parse_http_request(buffer, total_read);
    free(buffer);

    if (req == NULL) {
        char* error = format_error_response("Invalid HTTP request");
        send_http_response(client_fd, 400, "Bad Request", "application/json", error);
        free(error);
        return;
    }

    printf("[HTTP] %s %s\n", req->method, req->path);

    // 路由
    if (strcmp(req->method, "OPTIONS") == 0) {
        handle_options(client_fd);
    } else if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/health") == 0) {
        handle_health_check(client_fd);
    } else if (strcmp(req->method, "POST") == 0 &&
               (strcmp(req->path, "/v1/chat/completions") == 0 ||
                strcmp(req->path, "/chat/completions") == 0)) {
        handle_chat_completion(server, client_fd, req);
    } else {
        handle_not_found(client_fd);
    }

    http_request_free(req);
}

// ============ 服务器管理 ============

static void signal_handler(int sig) {
    (void)sig;
    if (g_server != NULL) {
        g_server->running = 0;
    }
}

HttpServer* http_server_create(GPTModel* model, Tokenizer* tokenizer, ServerConfig* config) {
    if (model == NULL || tokenizer == NULL) return NULL;

    HttpServer* server = (HttpServer*)malloc(sizeof(HttpServer));
    server->socket_fd = -1;
    server->running = 0;
    server->model = model;
    server->tokenizer = tokenizer;

    if (config != NULL) {
        server->config = *config;
    } else {
        server->config = default_server_config();
    }

    server->chat_config = default_chat_config();

    return server;
}

void http_server_free(HttpServer* server) {
    if (server == NULL) return;
    if (server->socket_fd >= 0) {
        close(server->socket_fd);
    }
    free(server);
}

int http_server_start(HttpServer* server) {
    if (server == NULL) return -1;

    // 创建 socket
    server->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server->socket_fd < 0) {
        perror("socket");
        return -1;
    }

    // 设置 SO_REUSEADDR
    int opt = 1;
    setsockopt(server->socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 绑定
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(server->config.port);

    if (bind(server->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server->socket_fd);
        server->socket_fd = -1;
        return -1;
    }

    // 监听
    if (listen(server->socket_fd, server->config.max_connections) < 0) {
        perror("listen");
        close(server->socket_fd);
        server->socket_fd = -1;
        return -1;
    }

    // 设置信号处理
    g_server = server;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    server->running = 1;
    printf("miniLLM HTTP Server started on http://localhost:%d\n", server->config.port);
    printf("Endpoints:\n");
    printf("  GET  /health              - Health check\n");
    printf("  POST /v1/chat/completions - Chat completion (OpenAI compatible)\n");
    printf("Press Ctrl+C to stop.\n\n");

    // 主循环
    while (server->running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server->socket_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            if (!server->running) break;
            perror("accept");
            continue;
        }

        // 处理请求 (单线程，顺序处理)
        handle_request(server, client_fd);
        close(client_fd);
    }

    printf("\nServer stopped.\n");
    return 0;
}

void http_server_stop(HttpServer* server) {
    if (server == NULL) return;
    server->running = 0;
    if (server->socket_fd >= 0) {
        shutdown(server->socket_fd, SHUT_RDWR);
    }
}

int http_server_is_running(HttpServer* server) {
    if (server == NULL) return 0;
    return server->running;
}
