---
title: 'Lab13 - BPE 对话整合'
layout: 'course_page'
permalink: '/course/practice/labs/lab13-end-to-end/'
lab: 'Lab13'
step: 'step11'
hours: '3h'
deliverable: '完成 BPE 对话整合相关 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab13-end-to-end/TASK.md'
---

> **实验编号** Lab13 **预计耗时** 3h **本节产出** 补完 BPE 编码、解码、单轮生成和循环接口，并让验证输出相对当前实验基线继续转绿


## 实验目的

前 12 章把 BPE 分词（Lab12）、KV Cache（Lab11）、生成采样（Lab08）、对话模板（Lab09）各做了一段，**但还没有一次把它们整合成一条能连续工作的 BPE 对话链路**。

本 lab 解决最后一公里：学员在 `framework/student.c` 里写三个 5~20 行的小函数，把它们拼成一条可交互的 BPE 对话回路。

完成本 lab 后你应当能：

1. 看一眼 `bpe_encode` / `bpe_decode` / `model_forward` 三个 API，就知道 demo 程序的整条数据流。
2. 写出一个最简 REPL：循环读 stdin → BPE 编码 → 模型前向 → 采样 → BPE 解码 → 打印。
3. 区分"训练语料配对"（`model_bpe.bin` 必须配 `bpe_vocab.txt`）和"运行时分词"（输入文本 → BPE 子词 ID）。
4. 用 `verify.c` 的 5 个自动测试证明你写的三个函数**接口对、行为对**。

## 为什么最后还要做一章“整合”

前 12 章你已经陆续做出了很多看起来独立的部件：BPE 编解码、KV Cache、采样策略、聊天模板、HTTP 服务、完整模型、训练循环。对项目型课程来说，最危险的时刻也恰恰在这里，因为学员很容易停留在“我分别做过这些东西”，却还没有真正把它们串成一条连续的数据流。

所以这最后一章的目标，不是再教一个新概念，而是逼你亲手完成一次总装。为什么输入一行文本之后，要先 BPE 编码；为什么第一次 `model_forward` 用整段 prompt，而后面只喂一个 token；为什么最后还得把生成出的 token ID 重新解码回字符串。只要这条链路被你自己接通，前面 12 章才真正闭环。

## 实验环境

涉及的文件：

```
labs/lab13-end-to-end/
├── TASK.md                # 本文件
├── Makefile
├── framework/
│   ├── student.c          # ★ 你要改的文件（含 TODO 注释）
│   ├── student.h
│   ├── verify.c           # 自动验证程序
│   └── verify.h
└── tests/                 # 留空（lab13 是综合章，不放输入数据）
```

可调用的 framework API（声明在 `minillm_lab/framework/*.h`）：

- `bpe_tokenizer.h`：`bpe_tokenizer_create()` / `bpe_tokenizer_free()` / `bpe_load_vocab()` / `bpe_encode()` / `bpe_decode()` / `bpe_vocab_size()`
- `model.h`：`GPTModel` 结构 + `model_create()` / `model_free()` / `model_init_random()` / `model_forward()` / `model_load()` / `model_cache_create()` / `model_cache_free()`
- `tensor.h` / `math_ops.h`：`tensor_create()` / `tensor_zeros()` / `tensor_free()` / `tensor_get()` / `tensor_set()`
- `config.h`：`ModelConfig` + `tiny_config()` / `default_config()` / `config_validate()`
- `generate.h`：`GenerateConfig` + `default_generate_config()` + `sample_top_k()` / `sample_greedy()`（可选）
- `chat.h`：`conversation_create()` / `conversation_free()` / `conversation_add()` / `format_prompt_with_input()`

> **注意**：本 lab 的 `student.c` **不**直接 include 任何 `framework/*.c`——所有 framework 代码由 Makefile 链接进来。

命令：

```bash
cd labs/lab13-end-to-end
make clean && make test
```

## 实验内容

本 lab 共有 4 个子任务，全部集中在 `framework/student.c` 里。每个子任务对应一个 TODO 函数。

这 4 个任务其实正好对应一条完整对话链路的 4 个断点：先把用户输入变成模型可读 token，再把模型产出的 token 变回文本，中间补上单轮生成逻辑，最后再包一层 REPL。你可以把它看成“综合章版的 explain -> practice -> verify -> reflect”，只是这次 explain 的对象已经不再是单个模块，而是模块之间的接缝。

### 任务 1.5.1：填 `student_bpe_chat_tokenize`（BPE 编码用户输入）

打开 `framework/student.c`，找到 `student_bpe_chat_tokenize` 函数。函数签名：

```c
int* student_bpe_chat_tokenize(BPETokenizer* tok,
                               const char* user_input,
                               int* out_len);
```

它要把 `user_input`（一段英文，比如 `"hello"`）通过 `bpe_encode` 转成 token ID 数组，并返回**带 BOS、不带 EOS** 的 ID 序列（BOS 让模型知道序列起点；EOS 在生成时由 `sample_top_k` 自己吐出）。

骨架：

```c
int* student_bpe_chat_tokenize(BPETokenizer* tok,
                               const char* user_input,
                               int* out_len) {
    // TODO(student):
    //   1. 检查 tok 和 user_input 是否为 NULL
    //   2. 调 bpe_encode(tok, user_input, out_len, /*add_bos=*/1, /*add_eos=*/0)
    //   3. 返回 ID 数组
    //   4. out_len 由 bpe_encode 填充
    (void)tok; (void)user_input; (void)out_len;
    return NULL;   /* 占位返回 */
}
```

**action**：写 4~6 行，**记得返回前** `bpe_encode` 已经帮你 malloc 了一块 ID 数组，你直接 return 它即可。

这一题的重要性在于，它让“BPE 是训练期概念还是运行时概念”这个问题第一次落地。答案是两者都是：训练时它决定词表怎么长出来，运行时它又决定用户输入怎样被映射到同一套词表上。只要这层关系搞清楚，后面你就不会把 `bpe_vocab.txt` 当成一份只在训练时出现的静态文件。

### 任务 1.5.2：填 `student_bpe_chat_decode`（BPE 解码助手回复）

函数签名：

```c
char* student_bpe_chat_decode(BPETokenizer* tok, int* token_ids, int len);
```

它把 `token_ids[0..len)` 通过 `bpe_decode` 转回可打印字符串。

骨架：

```c
char* student_bpe_chat_decode(BPETokenizer* tok, int* token_ids, int len) {
    // TODO(student):
    //   1. 检查 tok / token_ids / len <= 0
    //   2. 调 bpe_decode(tok, token_ids, len) 拿到堆上的字符串
    //   3. 返回
    (void)tok; (void)token_ids; (void)len;
    return NULL;   /* 占位返回 */
}
```

**action**：写 4~6 行。返回值是 `malloc` 出来的——谁调谁 `free`。

这一题和上一题放在一起，真正构成了一条“运行时分词闭环”。如果只会 encode 不会 decode，模型就永远只能吐出一串整数；如果只会 decode 不会 encode，用户输入又永远进不了模型。综合章把这两步放在最前面，是为了先把 I/O 两端都钉住。

### 任务 1.5.3：填 `student_bpe_chat_one_turn`（单轮对话：编码 → 模型 → 采样 → 解码）

函数签名：

```c
char* student_bpe_chat_one_turn(GPTModel* model,
                                BPETokenizer* tok,
                                GPTCache* cache,
                                const char* user_input,
                                int max_new_tokens);
```

它要做"半轮"对话：把 `user_input` 喂给模型，从最后一个位置 logits 用 `sample_top_k` 贪心采样，逐个 token 解码、累积，直到生成 `max_new_tokens` 个或遇到 EOS，最后返回解码出的字符串。

骨架：

```c
char* student_bpe_chat_one_turn(GPTModel* model,
                                BPETokenizer* tok,
                                GPTCache* cache,
                                const char* user_input,
                                int max_new_tokens) {
    // TODO(student):
    //   1. 调 student_bpe_chat_tokenize 拿 prompt_ids
    //   2. 申请 logits = tensor_zeros(2, (int[]){prompt_len, vocab_size})
    //   3. 调 model_forward(model, prompt_ids, prompt_len, cache, logits)
    //   4. 把最后一行 logits 拷贝到 last_logits (形状 [vocab_size])
    //   5. 循环 max_new_tokens 次:
    //        - sample_top_k(last_logits, k, temperature)
    //        - 把采到的 id 追加到 out_ids
    //        - 如果 id == tok->eos_id 跳出
    //        - 用刚采到的 id 再做一次 model_forward 拿下一位置 logits
    //   6. 调 student_bpe_chat_decode 把 out_ids 转成字符串返回
    //   7. 收尾 free 临时内存
    (void)model; (void)tok; (void)cache;
    (void)user_input; (void)max_new_tokens;
    return dup_cstr("(TODO)");   /* 占位返回 */
}
```

**action**：写 25~40 行。这是本 lab 的核心题。**注意**步骤 5 里第二次 `model_forward` 的输入只有 1 个 token（刚采到的那个），不是整个 prompt——这是 KV Cache 复用的关键。

这一题是整章里最值得慢下来做的一题，因为它第一次把前面若干章的部件真正串成了一条活的路径：编码、前向、采样、缓存复用、解码。你如果能把这题写顺，后面再看 step11 的 `chat_bpe`，就不会只把它当成“现成程序”，而能看出每一段调用背后到底对应课程前面的哪一章。

### 任务 1.5.4：填 `student_chat_loop`（主循环：读 stdin → 调 one_turn → 打印）

函数签名：

```c
void student_chat_loop(GPTModel* model,
                       BPETokenizer* tok,
                       GPTCache* cache);
```

骨架：

```c
void student_chat_loop(GPTModel* model,
                       BPETokenizer* tok,
                       GPTCache* cache) {
    // TODO(student):
    //   1. 打印欢迎语 "BPE Chat (type /quit to exit)"
    //   2. 循环:
    //        - fgets(buf, sizeof(buf), stdin)
    //        - 去掉行尾 '\n'
    //        - 如果是 "/quit" 跳出
    //        - 否则调 student_bpe_chat_one_turn 拿回复
    //        - printf("Bot: %s\n", reply)
    //        - free(reply)
    (void)model; (void)tok; (void)cache;
    printf("(student_chat_loop not implemented yet)\n");
}
```

**action**：写 15~25 行。这一关不需要你"训"任何模型——随机初始化的 GPT 也能跑通循环（输出会是乱码但**循环本身**能 `[PASS]`）。

这一题看起来像“最后再包一个 while 循环”，其实它在课程结构里承担的是“把所有部件真正变成一个可交互体验”的任务。只要 REPL 跑起来，学员就会非常直观地看到：前面做的那些局部函数不是孤立作业，而是在通向一个真实可玩的程序。

## 现象

这一章当前的实验起点不是全红，而是“前面几项能跑，真正的整合路径还没闭合”。这符合它作为综合章的定位：BPE 编解码的壳子和 REPL 入口都在，但真正的数据流还没被你接起来。

### 当前实验基线（2026-06-17 复核）

```text
BPE Chat (type /quit to exit)
(student_chat_loop not implemented yet)
[TEST] 全函数空指针守护                       [FAIL]
测试结果: 2 通过, 3 失败
```

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
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
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/kv_cache.c -o build/kv_cache.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/bpe_tokenizer.c -o build/bpe_tokenizer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o build/tensor.o ... build/bpe_tokenizer.o
========================================
      Lab13 - BPE 对话整合测试
========================================
[TEST 1] bpe_chat_tokenize 编码非空 ........ [PASS]
[TEST 2] bpe_chat_decode 解码可逆 .......... [PASS]
[TEST 3] bpe_chat_one_turn 单轮生成 ........ [PASS]
[TEST 4] chat_loop 走完 1 轮 stdin ......... [PASS]
[TEST 5] 全函数空指针守护 .................. [PASS]
========================================
测试结果: 5 通过, 0 失败
========================================
```

如果 `chat_loop` 段的 [TEST 4] 卡住不退，多半是 `fgets` 没读到 EOF 时返回 NULL 但你忘了 `break`。

课程走到这里，最值得你检查的已经不只是“某个函数值对不对”，而是你是否真的能用自己的话，把整条 BPE 对话数据流从 stdin 一直讲到 stdout。只要这条链能被你自己复述清楚，前面所有章节才真正从知识点变成了工程系统。

## 思考题

### 思考题 1（必做）

`student_bpe_chat_one_turn` 里**第二次** `model_forward` 的输入 `input_ids` 长度是 1（只有一个刚采到的 token），不是整个 prompt 序列。这是为什么？  
提示：第 N 次采样时，前 N-1 个位置的 K、V 已经在 `cache` 里了，重新算它们是浪费。`model_forward` 内部怎么知道"只用新 token"？

### 思考题 2（必做）

如果 `model_bpe.bin` 是用 A 词表训的，但 `chat_bpe` 加载时拿的是 B 词表，会发生什么？  
提示：token ID 在两个词表之间**没有对齐**关系——A 词表的 ID 17 可能是 `"ing"`，B 词表的 ID 17 可能是 `"猫"`。解码出来的文本会变成什么样？

### 思考题 3（选做）

为什么本 lab 让学员**自己写** `student_bpe_chat_one_turn`，而 step11 的 `chat_bpe.c` 是直接调 `chat()` API？  
提示：前者是"教学版"——逼你把 5 个 step 串起来；后者是"生产版"——封装好了省事。两者的**接口**差多少？

### 思考题 4（选做）

如果 `student_chat_loop` 跑了 50 轮，`cache->current_len` 会一直涨吗？什么时候该 reset？  
提示：看 `kv_cache.h` 里的 `current_len` 字段，再看 `max_seq_len` 限制。

## 上节 / 下节

- **上节**：[Lab12 - BPE 分词](/course/practice/labs/lab12-step11/)：你刚学会 BPE 编/解。
- **下节**：**课程结束**。下一步可以选（A）回到 `framework/` 改 BPE 实现（换成 WordPiece），（B）把 `student_chat_loop` 改成 HTTP endpoint（接 Lab10 的 `http_server`），（C）把模型做大（`vocab_size=5000, hidden_dim=128`）重新训。
