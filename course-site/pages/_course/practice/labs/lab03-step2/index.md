---
title: 'Lab03 - 词嵌入与位置编码'
layout: 'course_page'
permalink: '/course/practice/labs/lab03-step2/'
lab: 'Lab03'
step: 'step2'
hours: '2h'
deliverable: '完成 framework/student.c 中本章 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab03-step2/TASK.md'
---


> **实验编号** Lab03 **预计耗时** 2h **本节产出** 补完本章 TODO，并让验证输出相对当前实验基线继续转绿

## 实验目的

完成本 lab 后，你应当能够：

1. 解释"把整数 token ID 变成 64 维浮点向量"这一步是 LLM 中**第一处**有可训练参数的地方。
2. 手算 `PE(pos=0)` 的前 4 个维度是 `[0, 1, 0, 1]`，并解释为什么所有 pos=0 的偶数维 = 0、奇数维 = 1。
3. 在代码里区分"查表 (token embedding)"与"加位置 (position embedding)"——这两件事在 `embedding_forward` 里是**独立**的。
4. 验证"相同 token 在不同位置 → 输出向量不同"，从而解释 PE 不可缺。
5. 区分 sinusoidal (闭式公式、可外推) 与 learned (查表、不能外推) 两种位置编码的优缺点。

## 实验环境

- **涉及文件**：
  - `framework/student.c` —— 你**唯一**需要修改的文件（含 `TODO(student)` 注释）
  - `framework/student.h` —— 你要实现的函数声明
  - `framework/verify.c` —— 自动验证程序（**不要改**）
  - `../../framework/embedding.h` 与 `embedding.c` —— 共享的 Embedding 类型与实现（来自 step2）
  - `../../framework/tensor.h`、`math_ops.h` —— 共享 Tensor / 数学运算（链接但本 lab 不直接用）
- **可执行命令**：
  ```bash
  make            # 编译 build/ 目录
  make test       # 编译 + 跑 ./student
  make clean      # 清掉 build/ 与 student
  ```
- **共享 API（你可调用）**：
  - `Embedding* embedding_create(int vocab_size, int hidden_dim, int max_seq_len)`
  - `void embedding_free(Embedding* emb)`
  - `void embedding_init_random(Embedding* emb, float std)`
  - `void embedding_init_sinusoidal_position(Embedding* emb)`
  - `void embedding_init_learned_position(Embedding* emb, float std)`
  - `Tensor* tensor_create(int ndim, int* shape)` / `tensor_zeros` / `tensor_free`
  - `float tensor_get(Tensor* t, int* indices)` / `void tensor_set(...)`
  - 你自己写的 2 个 `student_*` 函数（见下）

> 框架已经帮你实现好 Embedding 的**结构体**和**辅助 init 函数**。你要在 `student.c` 里**自己写两个核心算子**——sinusoidal PE 的单元素计算，以及完整的 forward 拼装。**实现**就是这两块，框架代码一律不要动。

## 实验内容

### 1.5.1 阅读 `framework/student.h`，列出 2 个待实现函数

打开 `framework/student.h`，确认你要实现的 2 个函数签名：

- `float student_pe_sinusoidal(int pos, int dim, int hidden_dim);`
- `void  student_embedding_forward(Embedding* emb, int* token_ids, int seq_len, Tensor* output);`

确认你理解：
- `student_pe_sinusoidal` 是**单值函数**——只算 `PE(pos, dim)` 这一个数（不写循环，由调用方循环）。公式是 `sin(pos/10000^(2i/d))`（偶数维）和 `cos(...)`（奇数维），其中 `i = dim / 2`。
- `student_embedding_forward` 是**完整 forward**：先查 token embedding，再叠 PE，最后写到 `output` 张量。`output` 的 shape 是 `[seq_len, hidden_dim]`，由调用方预分配好。

### 1.5.2 在 `student.c` 的 `student_pe_sinusoidal` 里写 5~10 行

`student_pe_sinusoidal(pos, dim, hidden_dim)` 要做的事：

- 当 `pos == 0` 时，无论 `dim` 是奇是偶，所有 `i` 都让 `pos / 10000^(...) = 0`，因此偶数维 = `sin(0) = 0`，奇数维 = `cos(0) = 1`。
- 通用公式：先算 `div_term = powf(10000.0f, (float)(dim / 2) * 2.0f / (float)hidden_dim)`（注意：浮点除法，**先除再 `*2`** 跟框架里 `(i/2*2)` 整数除法的结果不同——你写浮点版即可）。
- 然后 `angle = (float)pos / div_term`。
- `dim % 2 == 0` 返回 `sinf(angle)`，否则返回 `cosf(angle)`。

提示：
- 框架代码里用的是整数除法 `(i/2*2)`，这等价于 `((i/2) * 2)`，效果是把 `2i` 截到偶数。你**也可以**直接用 `(dim / 2) * 2.0f`，更直观。
- 需要 `#include <math.h>` 才能用 `sinf` / `cosf` / `powf`。
- 写 5~10 行。

### 1.5.3 在 `student.c` 的 `student_embedding_forward` 里写 8~14 行

`student_embedding_forward(emb, token_ids, seq_len, output)` 要做的事：

1. `NULL` 保护：`emb` / `token_ids` / `output` 任一为 NULL → 直接 return。
2. 越界保护：若 `seq_len > emb->max_seq_len`，打印一条 `stderr` 警告并 return（**不**修改 output）。
3. 读 `int hidden_dim = emb->hidden_dim;`。
4. 双层循环：
   - 外层 `pos = 0 .. seq_len - 1`，从 `token_ids[pos]` 取 ID。
   - 越界 token：若 `id < 0 || id >= emb->vocab_size`，把 `id` 修正为 1（UNK）。
   - 内层 `d = 0 .. hidden_dim - 1`，把 `output[pos * hidden_dim + d] = token_emb[id*hidden_dim + d] + PE(pos, d)`。
5. 调你自己写的 `student_pe_sinusoidal(pos, d, hidden_dim)` 取 PE 值。

> 提示：访问 `emb->token_embedding->data[id * hidden_dim + d]`。**不要**写 `tensor_get`/`tensor_set`（那俩会越界检查，本 lab 用 flat 索引更直接）。

### 1.5.4 跑 `make test` 看 [PASS]

```bash
cd labs/lab03-step2
make clean && make test
```

预期：终端打印至少 3 行 `[TEST N] ... [PASS]`，最后一行 `All tests passed!`。

## 现象

`Lab03` 第一次运行时通常会是全红。这不是环境坏了，而是因为本章的 4 个测试都直接依赖你要写的 `student_pe_sinusoidal` 和 `student_embedding_forward`。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 1]  ... [FAIL] PE(0, 偶数)!=0 或 PE(0, 奇数)!=1
[TEST 2]  ... [FAIL] PE(1,0)=0.000000, want sin(1)=0.841471
[TEST 3]  ... [FAIL] 相同 token 不同位置 -> diff^2 = 0.000000 (应 > 0)
[TEST 4]  ... [FAIL] 不同 token 相同位置 -> diff^2 = 0.000000 (应 > 0)
4 test(s) FAILED.
```

### 完成本章 TODO 后的目标输出

```
========================================
       Lab03 - 词嵌入与位置编码 验证
========================================
[TEST 1] PE(pos=0) 偶数维=0, 奇数维=1 ... [PASS]
[TEST 2] PE(pos=1, dim=0) 接近 sin(1) ... [PASS]
[TEST 3] forward: 相同 token 不同位置 -> 输出不同 ... [PASS]
[TEST 4] forward: 不同 token 相同位置 -> 输出不同 ... [PASS]
========================================
All tests passed!
```

如果哪一行动 `[FAIL]`，先看 `verify.c` 里 TEST N 行的"预期值是多少"（注释里写好了手算值），再回头查你的 TODO 函数。

## 思考题

1. **(必做)** 在 `student_embedding_forward` 里**关掉 PE**（也就是把 `+ student_pe_sinusoidal(...)` 删掉，只查 token embedding），跑一遍 `make test`。**TEST 3 必然 FAIL**。请用"diff > 0 还是 = 0"这个数值证据，解释为什么 PE 对区分"AB"和"BA"是**必要**的。
2. **(必做)** 手算并写下来：`PE(0, 4)`、`PE(0, 5)`、`PE(0, 6)`、`PE(0, 7)` 的精确值（用 `sin(0)=0`, `cos(0)=1`, `sin(0/x)=0`, `cos(0/x)=1` 推）。为什么"pos=0 这一整行"都是 `[0, 1, 0, 1, ...]` 这种最平凡的形式？提示：所有 `pos/10000^(...)` 在 `pos=0` 时都是 0。
3. **(选做)** 把 `embedding_init_sinusoidal_position` 换成 `embedding_init_learned_position(emb, 0.02f)`，再跑 `make test`。观察 TEST 1 / TEST 2 是不是都还过？解释：为什么 learned position 也能让"相同 token 不同位置 → 不同输出"（hint：随机初始化的 PE[pos=0] 和 PE[pos=2] 几乎不可能完全相同）。
4. **(选做)** 假设 `vocab_size = 260`，`hidden_dim = 64`，`max_seq_len = 128`，这个 embedding 层**总共有多少参数**？其中 token embedding 占多少、position embedding 占多少？如果把 `hidden_dim` 加到 128，token embedding 部分的参数量会变成多少？提示：`token_embedding` 形状是 `[vocab_size, hidden_dim]`，按 float 4 字节算不算都行，本题只问**元素数**。

## 上节 / 下节

- **上节**：[Lab02 - 字符级 Tokenizer](/course/practice/labs/lab02-step1/)：你刚学会把 `"Hello"` 编成 `[76, 109, 108, 108, 111]`；本 lab 把这一串整数变成 64 维浮点向量。
- **下节**：[Lab04 - 多头自注意力](/course/practice/labs/lab04-step3/)：有了 `[seq_len, hidden_dim]` 的向量之后，下一步让每个位置"看"其他位置——进入 attention。
