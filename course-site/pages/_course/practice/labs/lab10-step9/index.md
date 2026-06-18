---
title: 'Lab10 - HTTP 服务'
layout: 'course_page'
permalink: '/course/practice/labs/lab10-step9/'
lab: 'Lab10'
step: 'step9'
hours: '2h'
deliverable: '完成请求行解析与响应构建两个 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab10-step9/TASK.md'
---


> **实验编号** Lab10 **预计耗时** 2h **本节产出** 补完请求行解析与响应构建两个接口，并让验证输出相对当前实验基线继续转绿

## 实验目的

前 9 个 lab 我们都在跟"本机二进制"打交道——`./student` 给你一句、模型回一句。
Lab10 的目标是把模型**装进一个 HTTP 服务**里，让"别的地方"能调它。

具体说，做完本 lab 你应当：

1. 能解释一次 HTTP "请求-响应"里**有哪几段**（method / path / header / body / status / 响应体）。
2. 能手写两个函数：把 `"GET /health HTTP/1.1"` 拆成 method / path / version，以及把 `200 / "OK" / body` 拼成完整 HTTP 响应文本。
3. 明白 **OpenAI 兼容 API** 是个契约——客户端不关心你跑的是 miniLLM 还是 GPT-4，JSON 字段名对就行。
4. 能用 `make clean && make test` 验证请求行解析和响应构建都成立。

> **本 lab 不要求你写 socket 监听循环**——那是 `framework/http_server.c` 的事。
> 你只负责**协议层两个最小积木**：拆请求行、拼响应帧。

## 为什么聊天之后还要单独学 HTTP

到 `Lab09` 为止，模型已经能在本机终端里按“对话模板 -> 生成 -> 提取回复”的路径工作了。但这仍然只是一个本地程序。只要它还不能被别的进程、别的机器、别的客户端通过协议访问，它就还没有真正变成服务。

这就是 HTTP 这一章的作用。课程现在故意不让你一上来写 socket 循环，而是先把协议层两个最小积木单独拿出来：一是读懂请求行，二是拼出合法响应。这样做的原因很直接：服务化最容易出错的地方，往往不是模型本身，而是协议边界有没有被说清楚。

## 实验环境

- 工作目录：`minillm_lab/labs/lab10-step9/`
- 你要改的文件：`framework/student.c`（其中有 2 个 `TODO(student)` 函数）
- 头文件：`framework/student.h`（声明你要写的 2 个函数）
- 自动验证：`framework/verify.c` 跑 `verify_run_all()`，打印 5 条 `[PASS]` / `[FAIL]`
- 共享 framework 代码（**只读**、**不修改**）：
  - `../../framework/http_server.{h,c}`——HTTP 请求/响应结构 + JSON 解析
  - `../../framework/{tensor,math_ops,embedding,attention,layernorm,ffn,transformer,model,generate,chat}.{h,c}`
- 标准库可用：`<stdio.h> <stdlib.h> <string.h> <ctype.h>`
- **不可用**：不要 include `../framework/*.c`——所有 framework 代码走 Makefile 链接

**编译运行：**

```bash
cd labs/lab10-step9
make clean && make test
```

## 实验内容

本 lab 只做协议层的"两个最小积木"——**不要求你起 socket 监听、不要求你 curl**。
框架已经搭好 socket 循环和 JSON 解析；你要写的是协议的"原子函数"。

这两个 task 之所以值得单独拆出来，是因为它们分别对应 HTTP 的两个基本问题：客户端到底请求了什么，服务端又准备怎样清楚地把结果送回去。只要这两层边界稳住，后面把 chat 能力暴露成 `/v1/chat/completions` 这种接口，就只是往里塞业务逻辑，而不是重新发明协议。

### 1.5.1 读懂 `HttpRequest` / `HttpResponse` 结构

**动作**：打开 `../../framework/http_server.h`，看 `HttpRequest` 和 `HttpResponse` 两个结构体。

**重点**：

- `HttpRequest` 里有 `method` / `path` / `body` / `body_len` / `content_type` 五个字段。
- `HttpResponse` 里有 `status_code` / `status_text` / `content_type` / `body` / `body_len` 五个字段。
- `body` 字段都是 `char*`——用 `malloc` / `strdup` 申请，记得 `free`。

**约束**：**不动** framework 头文件。

这里建议你带着“结构体就是协议展开图”的视角去看。`HttpRequest` 和 `HttpResponse` 并不是随手凑出来的字段集合，而是在把 HTTP 文本协议投影成 C 里的可操作对象。后面你写任何服务端时，这种“先把文本协议映射成结构体”的习惯都很有价值。

### 1.5.2 写 `student_http_parse_request_line`（在 `framework/student.c` 的 TODO 1）

**函数签名**：

```c
int student_http_parse_request_line(const char* line,
                                    char* method, int method_cap,
                                    char* path,   int path_cap,
                                    char* version, int version_cap);
```

**做什么**：把 `"GET /health HTTP/1.1"` 拆成 `method="GET"`、`path="/health"`、`version="HTTP/1.1"`。

**输入**：

- `line`：原始请求行（**保证以 `\0` 结尾**；不含 `\r\n`）。
- `method` / `path` / `version`：调用方提供的输出 buffer。
- `method_cap` / `path_cap` / `version_cap`：对应 buffer 的容量（包含结尾 `\0`）。

**输出**：

- 返回 `0` = 成功；返回 `-1` = 解析失败（行里没两个空格、某个段太长）。

**要点**：

- 用 `strchr` 找**两个空格**，把它们改成 `\0`，再 `strncpy`。
- 段长 ≥ 对应 `_cap` 时返回 `-1`（截断不安全）。
- 三个输出 buffer 至少要 16 字节——`HTTP/1.1` 就 8 字节。

**自检**：调 `student_http_parse_request_line("POST /v1/chat/completions HTTP/1.1", ...)`，应当拿到 `method="POST"`、`path="/v1/chat/completions"`、`version="HTTP/1.1"`，返回 0。

这一步真正想让你建立的意识是：协议解析首先是边界切分问题，而不是业务问题。只有先准确切出 method / path / version，后面才谈得上路由、JSON、chat handler。也就是说，这一题是在给后面的所有“上层逻辑”准备地基。

### 1.5.3 写 `student_http_build_response`（在 `framework/student.c` 的 TODO 2）

**函数签名**：

```c
char* student_http_build_response(int status_code, const char* status_text,
                                  const char* content_type, const char* body);
```

**做什么**：拼出完整 HTTP 响应文本，例如：

```
HTTP/1.1 200 OK\r\n
Content-Type: application/json\r\n
Content-Length: 33\r\n
Connection: close\r\n
\r\n
{"status": "ok", "model": "miniLLM"}
```

**输入**：

- `status_code`：HTTP 状态码（200 / 400 / 404 / 500）。
- `status_text`：对应文本（"OK" / "Bad Request" / "Not Found"）。
- `content_type`：`"application/json"` 等。
- `body`：响应体（可空；为 `NULL` 时 `Content-Length: 0`）。

**输出**：

- 返回**堆上的字符串**（调用方 `free`）；失败返回 `NULL`。

**要点**：

- **行结束符必须是 `\r\n`**，不是 `\n`——HTTP 协议规定。
- 头部和 body 之间是**一个空行**（`\r\n\r\n`）。
- `Content-Length` 必须用**字节数**（`strlen(body)`），不是字符数。
- 至少要包含 4 个 header：`Content-Type` / `Content-Length` / `Connection: close`（可加 CORS）。
- `body == NULL` 时 `Content-Length: 0`，但仍要写空行结束 headers。

**自检**：调 `student_http_build_response(200, "OK", "application/json", "{\"status\":\"ok\"}")`，
应当返回一段以 `"HTTP/1.1 200 OK\r\n"` 开头、含 `Content-Length: 15`、空行后是 `{"status":"ok"}` 的字符串。

这一题最容易被低估，因为很多人会把“拼响应文本”看成字符串练习。其实它训练的是另一层东西：你是否真的理解一份合法 HTTP 响应的最小骨架。后面如果你接 Web 前端、写 API 适配层、或者做 OpenAI 兼容服务，本质上都逃不开这层协议拼装。

### 1.5.4 跑通 verify

**动作**：

```bash
make clean && make test
```

**预期**：打印

```
[TEST 1] request line parse: GET ...        [PASS]
[TEST 2] request line parse: POST ...       [PASS]
[TEST 3] request line parse: invalid ...    [PASS]
[TEST 4] response build: 200 OK ...         [PASS]
[TEST 5] response build: 404 Not Found ...  [PASS]
```

这 5 个测试里，前 3 个主要压 `student_http_parse_request_line`，后 2 个主要压 `student_http_build_response`。真正合理的目标不是“先过一半”，而是把两类测试都做通，因为它们共同定义了一条完整的协议往返链路。

### 1.5.5 (选做) 改成"短"模式 / 加 CORS 头

如果上面的 5 个测试都 PASS 了，你可以**扩展** `student_http_build_response`：
- 加一个 `int add_cors` 参数，开启时多写 `Access-Control-Allow-Origin: *`。
- 或把 `Content-Length` 改用 `<` 的更安全格式（带 `%-10d` 对齐）。

**check**：`./student` 输出与改前一致，并且能解释每一处改动的"HTTP 协议根据"。

## 现象

`Lab10` 当前实验第一次运行时，通常会先通过 1 条到 3 条测试。原因是请求行解析和响应构建被 verify 分开了：你可能先把解析部分做对，但响应头格式还没完全对齐。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 3] request line parse: invalid (no second space) ...   [PASS]
[TEST 4] response build: 200 OK with JSON body ...   [FAIL]
[TEST 5] response build: 404 Not Found ...   [FAIL]
Result: 1 / 5 passed
```

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
rm -rf build student
mkdir -p build
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tensor.c -o build/tensor.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/math_ops.c -o build/math_ops.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/embedding.c -o build/embedding.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/attention.c -o build/attention.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/layernorm.c -o build/layernorm.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/ffn.c -o build/ffn.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/transformer.c -o build/transformer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/model.c -o build/model.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/generate.c -o build/generate.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/chat.c -o build/chat.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/http_server.c -o build/http_server.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o \
    build/tensor.o build/math_ops.o build/embedding.o build/attention.o build/layernorm.o \
    build/ffn.o build/transformer.o build/model.o build/generate.o build/chat.o build/http_server.o -lm
[TEST 1] request line parse: GET /health HTTP/1.1                 [PASS]
[TEST 2] request line parse: POST /v1/chat/completions HTTP/1.1   [PASS]
[TEST 3] request line parse: invalid (no second space)            [PASS]
[TEST 4] response build: 200 OK with JSON body                   [PASS]
[TEST 5] response build: 404 Not Found                           [PASS]
All tests done.
```

## 思考题

### (必做)

1. **`Content-Length` 头为什么是必须的？** 提示：HTTP 是字节流协议，`recv()` 一次可能读不到全部 body——服务端怎么知道"这条请求读完了"？
2. **HTTP 的行结束符为什么是 `\r\n` 而不是 `\n`？** 提示：早期协议要在电传打字机上"回车+换行"两件事；现在的 Linux 上 `\n` 也行，但严格客户端会拒收纯 `\n` 的响应。你的 `student_http_build_response` 漏掉 `\r` 会怎样？

### (选做)

3. **OpenAI 兼容 API 的"契约"是什么？** 提示：客户端不知道这个服务端跑的是 miniLLM 还是 GPT-4，只要 JSON 字段名对就行。这给了什么好处？
4. **`student_http_parse_request_line` 里我用了 `strchr` 找空格、`\0` 替换、`strncpy` 复制。** 如果用 `sscanf(line, "%s %s %s", ...)`，哪里会出问题？提示：HTTP 路径里能不能有空格？`HTTP/1.1` 后能不能有空白？
5. **如果同时来 100 个请求会怎样？** 提示：看 `http_server.c` 的 `http_server_start` 主循环是单线程串行。生产环境会怎么改造？

## 上节 / 下节

- **上节**：[Lab09 - 多轮对话](/course/practice/labs/lab09-step8/)——你把模型装进了 `Conversation`，本 lab 把它装进 HTTP。
- **下节**：[Lab11 - KV Cache](/course/practice/labs/lab11-step10/)——HTTP 服务的每次请求都从 token 0 重算 attention，下一步用 KV Cache 把 O(N²) 压到 O(N)。

---

**完成条件**：请求行解析与响应构建两类测试都转绿，并完成 2 道必做思考题。
