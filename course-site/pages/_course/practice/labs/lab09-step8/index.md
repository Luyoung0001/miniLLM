---
title: 'Lab09 - 多轮对话'
layout: 'course_page'
permalink: '/course/practice/labs/lab09-step8/'
lab: 'Lab09'
step: 'step8'
hours: '2h'
deliverable: '完成 prompt 拼接与回复提取两个 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab09-step8/TASK.md'
---

> **实验编号** Lab09 **预计耗时** 2h **本节产出** 补完 prompt 拼接与回复提取两个接口，并让验证输出相对当前实验基线继续转绿

## 实验目的

到 step7 为止，模型只懂"接龙"——给一段 prompt，它往后写 token。这一章要把模型变成"聊天"：让它**区分谁在说话**（user / assistant / system）、**记得自己之前说过什么**、**在历史过长时自动裁剪**。

读完 step8 的 `chat.c` 你会知道，所谓的"多轮对话"本质是三件事：

1. 把 `Conversation` 里的一串 `Message` 拼成一段带 `<|user|>` / `<|assistant|>` / `<|system|>` 标签的字符串（**prompt 模板**）；
2. 把这段字符串丢给 `generate()`；
3. 从模型输出里截出最后一个 `<|assistant|>` 之后的内容（**响应提取**）。

这一章你**不是**重写 `chat.c`，而是用 `student.c` 里两个小函数重新实现"拼 prompt"和"取回复"——这是任何 chat 系统最核心的两个字符串操作。

## 为什么生成之后还要单独学“对话格式”

做完 `Lab08` 之后，模型已经可以从一段 prompt 继续往后采样 token 了。但“能续写”还不等于“能聊天”。聊天系统和普通文本生成最大的区别，在于模型必须分清谁在说话、前面已经说到了哪里、当前这一轮回复应该接在什么位置之后。

这就是本章的地位。它没有引入新的神经网络结构，却第一次把“模型周围的协议”写出来。对一个 chat 系统来说，prompt 模板和回复提取不是外围小修小补，而是决定多轮上下文能否被正确组织的关键层。后面的 HTTP 服务和 BPE 对话整合，其实都会默认你已经把这一层字符串协议想明白了。

## 实验环境

```bash
cd labs/lab09-step8

# 文件清单
TASK.md            # 本文件
Makefile           # 编译脚本
framework/
  ├── student.h    # 学员要实现的函数声明
  ├── student.c    # ★ 学员改这个文件（含 TODO 注释）
  ├── verify.c     # 自动验证（不需要改）
  └── verify.h
tests/             # 测试输入（部分 lab 有）

# 编译 + 跑
make clean && make test
make clean

# 你可调用的 framework API（声明在 framework/chat.h / framework/chat.c）
#   const char *CHAT_USER_TAG, CHAT_ASSISTANT_TAG, CHAT_SYSTEM_TAG
#   const char *CHAT_END_TAG
#   int starts_with(const char *s, const char *p)
#   int ends_with(const char *s, const char *p)
#   char *trim_string(const char *s)         # 申请新串，调用方 free
```

注意：本 lab **不加载模型**。模型/分词器/真实 `chat()` 在 Lab10+ 才会用到。Lab09 只验证"两个字符串函数"——这是把 chat 系统跑通前的**最小可验证核心**。

## 实验内容

这两个 task 看起来都只是字符串拼接和截断，但它们其实共同决定了一条很重要的边界：模型看见的 prompt 到底长什么样，以及模型吐出来的一长串文本里，哪一段才算当前这轮真正的回复。把这两层理清楚，后面你看任何 chat API 都会更有把握。

### 1.5.1 准备：读 chat.h 画出字段

打开 `framework/student.h` 看一下你**要写**的两个函数（`student_format_prompt` / `student_extract_response`），再扫一眼 `framework/chat.h` 顶部的 4 个标签宏。然后看一眼 `framework/student.c` 里的 TODO 注释和 `int main`。

这一步**不写代码**，只读。但你必须做完这一步才看得懂后面两个任务。

这一步的目的，是先把 `<|user|>`、`<|assistant|>`、`<|system|>`、`<|end|>` 读成一套数据协议，而不是几段随手拼的字符串。只要你把标签理解成协议边界，后面两个 TODO 就不会再显得像“机械字符串题”。

### 1.5.2 在 student_format_prompt 里拼一段带标签的 prompt

打开 `framework/student.c`，定位 `student_format_prompt`。当前文件里是一个占位 `return dup_cstr("");`，这只是为了让骨架能编译，并不代表逻辑已经成立。

你的任务是把传入的 `(role, content)` 数组（按 `count` 长度；`role[i]` ∈ `{"user","assistant","system"}`）拼成一段这样的字符串（每条占 2 行 = 标签行 + 内容行，**最后一条之后**再加一行尾标签）：

```
<|user|>
你好
<|assistant|>
<|user|>
<|assistant|>
```

提示：用 `malloc` + `strcpy` / `strcat` 或者你自己的字符串拼接方式都可以；注意 `count==0` 时直接返回 `dup_cstr("<|assistant|>\n")`。函数**申请新串**，调用方 `free`。5~15 行。

这一题最想让你获得的，不只是“会把数组拼成一段字符串”，而是知道 chat prompt 的本质是“把结构化对话历史投影成模型可读的一段线性文本”。以后你再看到任何大模型模板，本质上都还是在做这件事。

### 1.5.3 在 student_extract_response 里找"最后一个"标签

定位 `student_extract_response`。当前文件里同样保留着占位 `return dup_cstr("");`，作用只是让编译先通过。

你的任务是从 `output` 里**找最右一个** `<|assistant|>`，从它后面开始截，到下一个 `<|user|>` / `<|assistant|>` / `<|system|>` / `<|end|>` 之前停（如果都没有，就到字符串末尾）。最后用 `trim_string` 去首尾空白。

提示：`strstr(p, "<|assistant|>")` 一次只找第一个；要找最右的，要么用 `strrstr`（标准库没有），要么在循环里 `ptr = strstr(ptr + 1, ...)` 一直扫到尾。5~15 行。

这一题表面上是在做字符串截断，实质上是在回答一个更重要的问题：模型生成出来的一长段文本里，哪一部分才属于“当前这轮回答”。如果这个边界判断错了，多轮聊天就会出现最典型的坏现象：模型把历史角色标签也一起复述回来。

### 1.5.4 跑通 verify

```bash
make clean && make test
```

你应该看到 `=== Prompt 格式化 ===`、`=== 响应提取 ===` 几段里至少 3 个 `[PASS]`。`student.c` 末尾的 `int main` 调的是 `verify_run_all()`，不是手写的 demo——你**不要**在 main 里加 printf。

## 现象

这里原来最容易误导学员的地方，就是把“baseline”和“完成后输出”混在了一起。实际上 `Lab09` 当前实验第一次运行时是**0 通过, 6 失败**，这恰好说明两个字符串函数都还没写。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 5] last-of-two tags wins ... [FAIL]
[TEST 6] stops at <|user|> boundary ... [FAIL]
测试结果: 0 通过, 6 失败
```

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
rm -rf build student
mkdir -p build
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c  -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/chat.c -o build/chat.o
... (其它 framework .o)
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o \
    build/chat.o build/... -lm
=== Prompt 格式化 ===
[TEST 1] 1 turn 1 assistant tag ... [PASS]
[TEST 2] 3 turns 4 user/assistant ... [PASS]
[TEST 3] trailing assistant present ... [PASS]

=== 响应提取 ===
[TEST 4] no tag returns trimmed ... [PASS]
[TEST 5] last-of-two tags wins ... [PASS]
[TEST 6] stops at <|user|> boundary ... [PASS]

========================================
测试结果: 6 通过, 0 失败
========================================
```

这里不建议把目标理解成“先过几条就算完成”。更好的读法是：6 个测试分别对应 prompt 拼接和回复提取的几个边界，最好全部转绿。

这章结束时，最理想的状态不是“两个字符串函数都能过测试”，而是你已经能清楚说出：聊天系统之所以比单轮生成多了一层复杂度，根本原因并不在模型结构，而在于上下文要先被组织成一份稳定的文本协议。

## 思考题

1. **(必做)** 为什么 `student_extract_response` 找的是"**最后一个**" `<|assistant|>`，而不是第一个？提示：想一下模型"自己模仿 user 又说了一遍"会发生什么。
2. **(必做)** 如果把 `<|user|>` 改成 `<User>`（去掉竖线），你的 `student_format_prompt` 仍然能编过——但模型行为会变吗？为什么？提示：分词器在分词时遇到"训练分布外"的字符串会怎样。
3. **(选做)** `format_prompt`（`framework/chat.c:138`）和 `format_prompt_with_input`（`framework/chat.c:182`）几乎是同一份代码复制了两遍。这种"小复制"在你现在的项目里可以接受吗？什么时候应该合并？什么时候故意不合并？
4. **(选做)** 如果 `student_format_prompt` 的 `count` 传 0，应当返回什么字符串？`strdup("")` 还是 `strdup("<|assistant|>\n")`？为什么这个选择对应"没有任何上下文，模型空开一段"？

## 上节 / 下节

- **上节**：Lab08 ([lab08-step7](/course/practice/labs/lab08-step7/))——文本生成（`generate()` / `generate_stream()`）。本章直接调 step7 的 `generate()`。
- **下节**：Lab10 ([lab10-step9](/course/practice/labs/lab10-step9/))——HTTP 服务（`curl` 假装 ChatGPT 客户端）。Lab10 会在 Lab09 这两个字符串函数之上，加一层 socket 把 `chat()` 暴露成 `/v1/chat/completions`。
