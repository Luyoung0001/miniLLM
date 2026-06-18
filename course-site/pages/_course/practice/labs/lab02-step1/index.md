---
title: 'Lab02 - 字符级 Tokenizer'
layout: 'course_page'
permalink: '/course/practice/labs/lab02-step1/'
lab: 'Lab02'
step: 'step1'
hours: '2h'
deliverable: '完成 framework/student.c 中本章 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab02-step1/TASK.md'
---


> **实验编号** Lab02 **预计耗时** 2h **本节产出** 补完本章 TODO，并让验证输出相对当前实验基线继续转绿

## 实验目的

完成本 lab 后，你应当能够：

1. 解释 miniLLM 字符级 tokenizer 的词汇表为什么是 260（4 特殊 + 256 ASCII）。
2. 区分 `<PAD>` / `<UNK>` / `<BOS>` / `<EOS>` 四个特殊 token 的用途。
3. 手动把 `'H'`、`'i'`、`' '` 这种字符算成 token ID（ASCII + 偏移 4）。
4. 写一个"编一次再解一次能拿回原串"的双向验证循环。

## 实验环境

- **涉及文件**：
  - `framework/student.c` —— 你**唯一**需要修改的文件（含 `TODO(student)` 注释）
  - `framework/student.h` —— 你要实现的函数声明
  - `framework/verify.c` —— 自动验证程序（**不要改**）
  - `../../framework/tokenizer.h` 与 `tokenizer.c` —— 共享字符级 tokenizer 实现（来自 step1）
  - `../../framework/tensor.h`、`math_ops.h` —— 链接进来但本 lab 不直接用
- **可执行命令**：
  ```bash
  make            # 编译 build/ 目录
  make test       # 编译 + 跑 ./student
  make clean      # 清掉 build/ 与 student
  ```
- **共享 API（你可调用）**：
  - `Tokenizer* tokenizer_create(void)` —— 构造一个字符级 tokenizer
  - `void tokenizer_free(Tokenizer* tok)` —— 释放
  - `int tokenizer_vocab_size(Tokenizer* tok)` —— 词汇表大小
  - `int tokenizer_is_special(Tokenizer* tok, int id)` —— 是否特殊 token
  - 你自己写的 3 个 `student_*` 函数（见下）

> 框架已经帮你实现了 `tokenizer_encode_char` / `tokenizer_decode_id`（内部用转义序列）。**你**要在 `student.c` 里**自己重写一份更朴素的版本**——为的是让你亲手摸一遍"加偏移"和"减偏移"。

## 实验内容

### 1.5.1 阅读 `framework/student.h`，列出 3 个待实现函数

打开 `framework/student.h`，确认你要实现的 3 个函数签名：

- `int  student_encode_char(Tokenizer* tok, char c);`
- `char student_decode_id (Tokenizer* tok, int id);`
- `int  student_roundtrip (Tokenizer* tok, const char* text, char* out, int out_size);`

确认你理解 `Tokenizer*` 是一个由 `tokenizer_create()` 返回的句柄——你**只**通过它读 vocab size / 调内置 `tokenizer_encode_char` 做对照，**不要**直接读结构体字段。

### 1.5.2 在 `student.c` 的 `student_encode_char` 里写 1~3 行

`student_encode_char` 要做的事：

- 如果 `tok` 为 NULL 或 `c` 是负数，返回 `TOKEN_UNK` (1)。
- 否则返回 `(unsigned char)c + NUM_SPECIAL_TOKENS`（也就是 ASCII + 4）。

提示：常量 `NUM_SPECIAL_TOKENS`、`TOKEN_UNK` 在 `framework/tokenizer.h` 里已经定义好。

### 1.5.3 在 `student.c` 的 `student_decode_id` 里写 4~6 行

`student_decode_id` 要做的事：

- 如果 `id < NUM_SPECIAL_TOKENS`（即 0~3 特殊 token），返回 `'\0'`。
- 如果 `id >= tok->vocab_size`，返回 `'\0'`。
- 否则返回 `(char)(id - NUM_SPECIAL_TOKENS)`。

> 注意：本函数只处理"普通字符 ID"（4~259）。遇到特殊 token 返回 `'\0'`，让上层区分"这是 PAD 还是真的 NUL"。

### 1.5.4 在 `student.c` 的 `student_roundtrip` 里写 8~12 行

`student_roundtrip` 要做"encode → decode → 写入 out 缓冲"：

1. 用 `tokenizer_encode(tok, text, &len, 0, 0)` 拿到 ID 数组（**不加** BOS/EOS，方便逐字符对照）。
2. 申请一个临时缓冲 `tmp` 长度 = `len + 1`。
3. 逐 ID 用你写的 `student_decode_id` 还原成字符，写到 `tmp`。
4. `tmp[len] = '\0'`。
5. 用 `strncpy(out, tmp, out_size - 1)` 复制到 `out`，并在末尾补 `'\0'`。
6. 返回实际写入的字符数 `len`。
7. **记得** `free(ids)` 和 `free(tmp)`。

> 边界：若 `text` 为空串或 `out` / `out_size` 非法，直接返回 0 并把 `out[0]='\0'`。

### 1.5.5 跑 `make test` 看 [PASS]

```bash
cd labs/lab02-step1
make clean && make test
```

预期：终端打印 4 行 `[TEST N] ... [PASS]`，最后一行 `All tests passed!`。

## 现象

这里也要先区分“初始实验状态”和“学员完成后的目标状态”。`Lab02` 的验证器里有一项词汇表大小检查不依赖你的 TODO，所以第一次运行时会出现“一项先绿，三项还红”的状态。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 1]  ... [PASS]
[TEST 2]  ... [FAIL] encode('H')=1 (want 76), encode('i')=1 (want 109)
[TEST 3]  ... [FAIL] decode(76)='\0' (want 'H'), decode(109)='\0' (want 'i')
[TEST 4]  ... [FAIL] roundtrip returned len=0 (want 5)
3 test(s) FAILED.
```

这说明共享 tokenizer 的基础框架已经连通，但 `student_encode_char`、`student_decode_id`、`student_roundtrip` 还没有写对。

### 完成本章 TODO 后的目标输出

```
========================================
       Lab02 - 字符级 Tokenizer 验证
========================================
[TEST 1] 词汇表大小 = 260 ... [PASS]
[TEST 2] 'H' 编码 = 76, 'i' 编码 = 109 ... [PASS]
[TEST 3] decode(76) = 'H', decode(109) = 'i' ... [PASS]
[TEST 4] "Hello" 往返一致 ... [PASS]
========================================
All tests passed!
```

如果哪一行动 `[FAIL]`，先看 `verify.c` 的预期值对不对，再回头查你的 TODO 函数。

## 思考题

1. **(必做)** 为什么 miniLLM 用 ASCII（0~255）做字符级，而不是直接用 UTF-8？提示：UTF-8 是变长 1~4 字节，"字符级"必须按 byte 算才简单。
2. **(必做)** 如果你写训练循环时把所有样本的 `add_bos=0, add_eos=0`，模型在生成阶段怎么知道"一个完整句子结束了"？请描述一个**不靠 EOS 也能停**的退路。
3. **(选做)** 词汇表大小从 260 涨到 5 万（BPE）会怎样影响：(a) 单条样本的 token 数；(b) embedding 表的参数量；(c) 单步 softmax 的算量。
4. **(选做)** 在 `student_roundtrip` 里你用 `strncpy` 复制结果到 `out`。如果输入 `"Hello"` 的 encode 长度是 5，但 `out_size=3`，会出现什么？你的代码现在能正确处理吗？解释为什么 `strncpy` 之后**还要**手动补 `'\0'`。

## 上节 / 下节

- **上节**：[Lab01 - 张量与数学运算](/course/practice/labs/lab01-step0/)：你刚学会 `Tensor` 和矩阵乘法；本 lab 拿到的 token ID 就是后面 embedding 层的输入。
- **下节**：[Lab03 - 词嵌入与位置编码](/course/practice/labs/lab03-step2/)：把 `id=76` 这种整数变成 64 维浮点向量。
