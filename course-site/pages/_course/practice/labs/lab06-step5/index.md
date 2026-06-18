---
title: 'Lab06 - 完整 GPT 模型'
layout: 'course_page'
permalink: '/course/practice/labs/lab06-step5/'
lab: 'Lab06'
step: 'step5'
hours: '3h'
deliverable: '完成完整模型组装与保存相关 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab06-step5/TASK.md'
---

> **实验编号** Lab06 **预计耗时** 3h **本节产出** 补完默认配置、模型组装、模型保存三个接口，并让验证输出相对当前实验基线继续转绿


## 实验目的

完成 Lab05 后你手里有一块 transformer block——一个"能跑"的乐高单元。GPT 实际上就是 **N 块同样的乐高** 前后各加一些零件后拼出来的长链。本 lab 解决两件事：

1. **组装**：把 `Embedding` + N × `TransformerBlock` + `Final LayerNorm` + `LM Head` 拼成一个完整的 `GPTModel`。
2. **序列化**：用 `fwrite` 把全部权重落到磁盘成 `model.bin`，再用 `fread` 完整恢复，让前后两次前向输出**逐位相同**（bit-exact）。

读完之后你应当能口述：`model_forward` 内部做的四步 = `embedding_forward` → 循环 N 次 `transformer_block_forward` → `layernorm_forward`（final）→ `matmul(hidden, lm_head)`。

## 为什么组装和保存要放在同一章

到 `Lab05` 为止，你已经拥有了一个能工作的 Transformer block，但这还只是“很多零件已经准备好”，并不等于“完整模型已经存在”。如果这些零件没有被统一装配进一个 `GPTModel`，后面的训练、生成、聊天其实都没有稳定边界可言。

这就是为什么本章同时讲组装和保存。组装解决的是“完整模型对象到底包含哪些部分”；保存解决的是“这个对象怎样稳定地跨进程、跨命令、跨实验阶段被重新拿回来”。后面的 `Lab07` 到 `Lab13` 之所以能够按训练、生成、对话、HTTP、KV cache、BPE 分开推进，前提就是本章已经先把“完整 GPT 模型”的对象边界和文件边界都钉死了。

## 实验环境

涉及的文件：

```
labs/lab06-step5/
├── TASK.md                # 本文件
├── Makefile
└── framework/
    ├── student.c          # ★ 你要改的文件（含 TODO 注释）
    ├── student.h
    ├── verify.c           # 自动验证程序
    └── verify.h
```

可调用的 framework API（声明在 `minillm_lab/framework/*.h`）：

- `config.h`：`ModelConfig` 结构 + `default_config()` / `tiny_config()` / `medium_config()` / `config_validate()` / `config_print()`
- `embedding.h`：`embedding_create()` / `embedding_free()` / `embedding_init_random()` / `embedding_init_sinusoidal_position()` / `embedding_forward()`
- `transformer.h`：`transformer_block_create()` / `transformer_block_free()` / `transformer_block_init()` / `transformer_block_forward()`
- `layernorm.h`：`layernorm_create()` / `layernorm_free()` / `layernorm_init()` / `layernorm_forward()`
- `model.h`：`GPTModel` 结构 + `model_create()` / `model_free()` / `model_init_random()` / `model_forward()` / `model_save()` / `model_load()` / `model_num_params()` / `model_memory_size()` / `model_print_info()`
- `tensor.h` / `math_ops.h`：`tensor_zeros()` / `tensor_get()` / `tensor_set()` / `tensor_argmax()` / `matmul_inplace()`

命令：

```bash
cd labs/lab06-step5
make clean && make test
```

## 实验内容

本 lab 共有 3 个子任务，全部集中在 `framework/student.c` 里。

这三个任务分别对应三种不同层次的工程能力。`student_default_config` 是配置边界，`student_model_create` 是对象与内存边界，`student_model_save` 是磁盘格式边界。把这三种边界都明确下来，你手里才真正有了一个可训练、可保存、可重新加载的模型。

### 任务 1.5.1：填 `student_default_config`（6 字段）

打开 `framework/student.c`，找到 `student_default_config` 函数。它应当返回一个 `ModelConfig`，6 个字段对应 `vocab_size / hidden_dim / num_heads / num_layers / ffn_dim / max_seq_len`。

回忆默认值（参考 `framework/config.h`）：`260 / 64 / 4 / 4 / 256 / 128`。注意 `hidden_dim % num_heads == 0` 是 `config_validate` 唯一会算的合法性检查——你抄写时**先看一眼 64 ÷ 4 = 16 是否整除**。

**TODO 位置**：`student.c` 内 `student_default_config` 函数体。

**action**：填 6 个 `config.xxx = ...`，最后 `return config;`。

这一题看起来像是在“抄默认值”，其实很适合建立一个很重要的工程习惯：配置不是随手填数字，而是后续所有模块共享的契约。只要这里某个字段写错，比如 `hidden_dim`、`num_heads` 或 `max_seq_len` 不一致，后面看到的错误通常不会立刻出现在配置处，而会延迟到 attention、训练甚至保存阶段才爆出来。

### 任务 1.5.2：填 `student_model_create`（申请 5 个子对象）

打开 `framework/student.c`，找到 `student_model_create` 函数。它接收一个 `ModelConfig` config，要返回一个完整的 `GPTModel*`。

按依赖顺序要申请 5 个子对象（任一失败要清理前面的，**别泄漏**）：

1. `model = (GPTModel*)malloc(sizeof(GPTModel))` 并保存 `config`
2. `embedding_create(vocab_size, hidden_dim, max_seq_len)` 赋给 `model->embedding`
3. 一个 `malloc(num_layers * sizeof(TransformerBlock*))` 给 `model->layers`，再循环 `transformer_block_create` 申请每一层
4. `layernorm_create(hidden_dim, 1e-5f)` 赋给 `model->final_ln`
5. `tensor_zeros(2, (int[]){hidden_dim, vocab_size})` 赋给 `model->lm_head`

> **关键约束**：`lm_head` 必须是 `[hidden_dim, vocab_size]`，**不是反过来**。因为最后一步是 `matmul(hidden, lm_head)`，输入 `hidden` 形状是 `[seq_len, hidden_dim]`——只有右矩阵列数 = `vocab_size`，结果才会是 `[seq_len, vocab_size]`。`step5` 的 `model_num_params` 默认应当返回 `240512`，把这个数反过来就破功了。

**TODO 位置**：`student.c` 内 `student_model_create` 函数体。

**action**：按上面 5 步写，每步失败要 `free` 前面已申请的内存后 `return NULL;`。

这一题真正想让你掌握的，不只是“malloc 五次”，而是开始把 `GPTModel` 看成一个有明确生命周期的对象。谁创建它、哪些子对象归它拥有、任何一步失败时怎样回滚，这些都是后面扩展缓存、优化器状态甚至 HTTP 服务时必须具备的工程意识。

### 任务 1.5.3：填 `student_model_save`（写 magic + config + 权重）

打开 `framework/student.c`，找到 `student_model_save` 函数。它接收一个 `GPTModel* model` 和一个 `const char* path`，要把整个模型按以下顺序写到 `path` 指定的文件里：

```
[4B magic "MLLM" = 0x4D4C4C4D]
[4B version = 1]
[sizeof(ModelConfig) 字节]
[token_embedding float 数组]   [position_embedding float 数组]
[第 0 层: LN1(γ,β) + Attn(Wq,Wk,Wv,Wo) + LN2(γ,β) + FFN(W1,b1,W2,b2)]
...
[第 N-1 层]
[final LayerNorm: γ,β]
[lm_head]
```

骨架步骤：

1. `FILE* f = fopen(path, "wb");`，失败 `return -1;`
2. 写 magic：`uint32_t magic = 0x4D4C4C4D; fwrite(&magic, sizeof(uint32_t), 1, f);`
3. 写 version：`uint32_t ver = 1; fwrite(&ver, sizeof(uint32_t), 1, f);`
4. 写 config：`fwrite(&model->config, sizeof(ModelConfig), 1, f);`
5. 写 token embedding：`fwrite(model->embedding->token_embedding->data, sizeof(float), <vocab*hidden>, f);`
6. 写 position embedding：同理
7. 写每一层（建议先写一个**辅助函数** `static void save_tensor(FILE* f, Tensor* t)`，把 `fwrite(t->data, sizeof(float), t->size, f)` 包起来——后面要写 1+2+2+4+2+2+2+1 = 16 个张量，没有这个 helper 你的代码会又长又乱）
8. 写 final LN（`gamma` + `beta`）
9. 写 `lm_head` 的 data
10. `fclose(f); return 0;`

**TODO 位置**：`student.c` 内 `student_model_save` 函数体（建议加 `static void save_tensor(FILE* f, Tensor* t)` 帮助函数）。

**action**：按上面 10 步写。**警告**：写顺序必须和你 `model_load` 的读顺序严格一致——这是后续 Lab07 训练-加载闭环能工作的前提。

保存这一题表面上像“机械地 fwrite 一串数组”，但它实际上是在训练你做一个最小二进制协议。文件头写什么、版本号怎么放、权重按什么顺序排，都会直接决定后面能不能正确加载。也就是说，本章第一次让你意识到：模型文件本身也是一份需要设计的数据格式。

## 现象

从 `Lab06` 开始，当前实验第一次运行通常会是全红。原因很简单：这一章负责的是“完整模型能不能被正确组装出来”，而 verify 的 16 个断言几乎都直接依赖这 3 个 TODO。

### 当前实验基线（2026-06-17 复核）

```text
[FAIL] model_create ok for round-trip
测试结果: 0 通过, 16 失败
```

你第一次看到这个结果，不必惊慌。它的意义只是：框架已经能编译、验证器已经开始工作，但 `student_default_config`、`student_model_create`、`student_model_save` 还没有提供有效实现。

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tensor.c -o build/tensor.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/math_ops.c -o build/math_ops.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tokenizer.c -o build/tokenizer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/embedding.c -o build/embedding.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/attention.c -o build/attention.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/layernorm.c -o build/layernorm.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/ffn.c -o build/ffn.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/transformer.c -o build/transformer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/model.c -o build/model.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o build/tensor.o ...
========================================
      Lab06 - 完整 GPT 模型 测试
========================================
[TEST 1] default config fields ............ [PASS]
[TEST 2] model_create wires all 5 sub-objects  [PASS]
[TEST 3] model_save writes 240K+ floats ... [PASS]
[TEST 4] model.bin header is MLLM v1 ...... [PASS]
[TEST 5] save+load round-trip is bit-exact  [PASS]
========================================
测试结果: 5 通过, 0 失败
========================================
```

如果 `model_num_params` 不是 `240512` 或 `model_load` 之后前向输出对不上，回到 `student.c` 检查写顺序。

本章完成后，你手里第一次有了一个真正意义上的“项目核心资产”：不是某个 block，不是某份局部权重，而是一整个可创建、可保存、可重新加载的 GPT 模型。训练和推理之所以能在后续章节里分开讨论，正是因为这里先把总装边界做对了。

## 思考题

### 思考题 1（必做）

如果 `student_model_save` 里"写 magic"那一步被你删了（直接写 config + 权重），会发生什么？  
提示：想想加载时文件如果短了一截，`fread` 读 magic 位置会读到全 0。你能立刻发现自己拿错了文件吗？

### 思考题 2（必做）

`config_save` 写的是 `sizeof(ModelConfig)` 个字节 = 6 个 int = 24 字节。如果你之后给 `ModelConfig` 加一个 `float dropout = 0.1` 字段（结构变成 28 字节），但用**老**的二进制模型去加载，会发生什么？  
提示：想想 `fread` 读 config 用了 28 字节，但老文件只有 24 字节，**后面第一个权重矩阵**会被挪位 4 字节读取。`diff_count` 会是多少？

### 思考题 3（选做）

为什么 `lm_head` 是一个**普通 `Tensor`**，但 `embedding` 是一个**结构体 `Embedding*`**？  
提示：打开 `framework/embedding.h` 看 `Embedding` 里的成员——它至少有什么是 `Tensor` 没有的？

### 思考题 4（选做）

`model_num_params` 返回 `int`。如果模型有 30 亿参数（真的 LLM），`int` 装得下吗？`size_t` 会不会更合适？  
提示：在你的机器上 `int` 是几位？`size_t` 是几位？

## 上节 / 下节

- **上节**：[Lab05 - Transformer Block](/course/practice/labs/lab05-step4/)
- **下节**：[Lab07 - 训练循环](/course/practice/labs/lab07-step6/)：用 `model_save` 存的是"未训练"权重；下一步要让它真能学——`loss.c` 算 cross-entropy，`backward.c` 算梯度，`optimizer.c`（Adam）改权重。
