---
title: 'Lab05 - Transformer Block'
layout: 'course_page'
permalink: '/course/practice/labs/lab05-step4/'
lab: 'Lab05'
step: 'step4'
hours: '3h'
deliverable: '完成 framework/student.c 中本章 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab05-step4/TASK.md'
---

> **实验编号** Lab05 &nbsp;&nbsp; **预计耗时** 3h &nbsp;&nbsp; &nbsp;&nbsp; **本节产出** 补完 `student_layernorm`、`student_residual_add`、`student_block_forward`，并让验证输出相对当前实验基线继续转绿


## 实验目的

完成 Lab04 后，你已经能让一个 token "看到" 之前所有 token（多头因果注意力）。但只把 Attention 一层层堆起来，模型很快会数值崩溃——残差信号消失、激活爆炸、loss 飘成 NaN。

让 Transformer 能训练的关键是三件事：

1. **残差连接**：让梯度"有一条高速公路直接流回输入"
2. **Layer Normalization**：把每一层输出拉回"均值为 0、方差为 1"的尺度
3. **Feed-Forward Network (FFN)**：给每个位置加一个"独立思考"的两层 MLP

本 lab 完成后，你应当能：

- 口述 Pre-LN Block 的数据流：`x = x + Attn(LN(x))`，再 `x = x + FFN(LN(x))`。
- 自己写出 LayerNorm 公式里"减均值、除标准差、乘 γ 加 β"每一步在做什么。
- 解释 FFN 为什么要"扩展到 `4 * hidden_dim`，再压回 `hidden_dim`"。
- 算出"一个 block 约 5 万参数"是怎么来的。
- 从 `./student` 的输出里指认每条 [PASS] 对应 `student.c` 里的哪一行 TODO。

## 为什么 attention 之后还不够

很多同学第一次学 Transformer 时，会误以为“attention 写完，模型最关键的部分就已经结束了”。实际上还差得很远。attention 负责让一个位置读取其他位置的信息，但它并不自动解决深层网络里的数值稳定性问题，也不自动提供足够强的逐位置非线性变换能力。

这就是为什么真正的 Transformer block 一定还要带 LayerNorm、残差和 FFN。LayerNorm 负责把每一层输入重新拉回稳定尺度；残差负责让信息和梯度都有一条直接主干；FFN 则让每个位置在看完上下文之后，仍然能做一次“只属于自己”的局部加工。把这三件事接回 attention，模型才从“会做上下文加权”变成“能被一层层堆起来训练”的结构。

## 实验环境

### 涉及文件（学员只需改 `student.c`）

```
labs/lab05-step4/
├── TASK.md                  # 本文件
├── Makefile                 # 编译脚本
├── framework/
│   ├── student.h            # 学员要实现的函数声明
│   ├── student.c            # ★ 你要改的文件（含 TODO）
│   ├── verify.c             # 自动验证程序
│   └── verify.h
└── tests/
```

### 可用 API（全部来自 `../../framework/`）

| 头文件 | 关键函数 |
| --- | --- |
| `tensor.h` | `tensor_create / tensor_zeros / tensor_randn / tensor_get_flat / tensor_set_flat / tensor_free` |
| `math_ops.h` | `tensor_add / tensor_add_inplace / tensor_mean / tensor_var / tensor_std` |
| `layernorm.h` | `LayerNorm* / layernorm_create / layernorm_forward` |
| `ffn.h` | `FFN* / ffn_create / ffn_forward / FFNCache*` |
| `attention.h` | `MultiHeadAttention* / attention_forward / create_causal_mask` |
| `transformer.h` | `TransformerBlock* / transformer_block_create / transformer_block_forward / TransformerCache*` |

### 命令

```bash
cd labs/lab05-step4
make clean && make test
make clean     # 清理
```

## 实验内容

这一章的 3 个任务本身也构成一个递进关系。`student_layernorm` 解决的是数值尺度，`student_residual_add` 解决的是主干路径，`student_block_forward` 才是把 attention、FFN 和这两者真正接成一个可重复堆叠的模板。你如果只盯着最后一个 task，前两个没有吃透，verify 往往就会表现成“形状对了，但数值行为还不对”。

### 1.5.1 实现 `student_layernorm`

在 `framework/student.c` 的 `student_layernorm` 函数里写 LayerNorm 公式：

```
y_i = (x_i - mean) / sqrt(var + eps)
```

输入 `x` 是 1D 张量 `[hidden_dim]`，先在最后一维算均值与方差，再逐元素归一化。**注意**：本函数**只做归一化**，不乘 γ 加 β（把"加可学习参数"留给真正的 `LayerNorm`）。

**规模提示**：1D 向量长度通常 `hidden_dim = 64`。

这一题最该建立的意识是：LayerNorm 不是“为了公式完整所以补一项”，而是让后续计算始终待在一个更可控的数值区间里。后面训练章节如果你看到 loss 飘成 `NaN`，首先就应该回到这种“有没有在正确位置做稳定化”的问题上。

### 1.5.2 实现 `student_residual_add`

在 `framework/student.c` 的 `student_residual_add` 函数里做**原地**残差加法：`a += b`，shape 都是 `[seq_len, hidden_dim]`。可调用 `tensor_add_inplace(a, b)`，5 行内可完成。

代码上这一题最短，但概念上绝对不能轻视。`x = x + F(x)` 的含义不是“少开一个 buffer”，而是模型始终保留原输入主干，同时只学习增量修正。正因为有这一条主干，哪怕某个子层一开始学得很差，网络也不至于把已有信息整块破坏掉。

### 1.5.3 实现 `student_block_forward`

在 `framework/student.c` 的 `student_block_forward` 函数里把 LN、Attention、LN、FFN、两个 `+` 号按 Pre-LN 顺序拼起来：

```
h = input
h = h + attention(layer_norm(h))
h = h + ffn(layer_norm(h))
*output = h
```

`output` 张量已由 `verify.c` 预先分配，shape 与 `input` 相同。**注意输出张量是传入指针**，用 `*output = ...` 赋值；缓存张量（`cache->ln1_out` 等）也由 verify 创建，你直接用即可。

**数据流参考**：

```
input -> ln1_out = LN(input)
      -> attn_out = Attn(ln1_out)
      -> input + attn_out -> input
      -> ln2_out = LN(input)
      -> ffn_out = FFN(ln2_out)
      -> input + ffn_out -> output
```

这一题真正考你的，是“能不能把前面几章已经见过的零件，按一个严格顺序重新串起来”。attention、FFN、LayerNorm 单拿出来你都见过了，但 block 的价值就在于它把这些零件固定成了一个能重复堆叠的模板。后面的完整 GPT 模型，本质上就是把这个模板前后再接上 embedding 和输出头。

### 1.5.4 编译并跑通

```bash
cd labs/lab05-step4
make clean && make test
```

第一次运行时不要求直接过线。先确认当前实验代码能编译、verify 能运行，再逐个把 3 个 TODO 对应的测试点转绿。

## 现象

`Lab05` 当前不是全红状态，因为共享的 `attention_forward`、`ffn_forward`、缓存结构都已经由 framework 提供。你第一次运行时更可能看到“若干基础检查已经通过，但 block 组装还没有完全成立”。

### 当前实验基线（2026-06-17 复核）

```text
[PASS] output shape 与 input shape 一致
[FAIL] output 不全为零
[PASS] output 与 input 不同
[PASS] output 不含 NaN / Inf
测试结果: 7 通过, 4 失败
```

### 完成本章 TODO 后的目标输出

```
[TEST 1] student_layernorm: 1D vector mean ≈ 0            ... [PASS]
[TEST 2] student_layernorm: 1D vector var  ≈ 1            ... [PASS]
[TEST 3] student_residual_add: 原地 a += b shape 一致      ... [PASS]
[TEST 4] student_block_forward: output shape = input shape ... [PASS]
[TEST 5] student_block_forward: output 与 input 不同       ... [PASS]
[TEST 6] student_block_forward: 跑完无 NaN                ... [PASS]
================================
测试结果: 6 通过, 0 失败
================================
```

看到 `测试结果: 6 通过, 0 失败` 即合格。如果某一行为 `[FAIL]`，先看是哪个函数——回到 `framework/student.c` 对应 TODO 改。

这章收尾时，如果你只能说“block 跑通了”，其实还不够。更好的完成标准是：你已经能清楚解释为什么 Transformer 不是“attention 再随便加几层”，而必须是一个带 LayerNorm、残差和 FFN 的稳定模板。只要这点真懂了，下一章完整 GPT 的组装就会显得非常自然。

## 思考题

1. **(必做)** 为什么 Pre-LN 训练比 Post-LN 更稳？提示：从"残差路径上信号的方差"想。如果层数堆到 12 层以上，Post-LN 会出现什么现象？
2. **(必做)** FFN 为什么要先扩到 `4 * hidden_dim` 再压回？提示：如果扩到 `hidden_dim`（不扩张）或扩到 `16 * hidden_dim`，会发生什么？
3. **(选做)** 残差连接的"高速公路"在反向传播时对应什么？提示：写出 `y = x + F(x)`，对 `x` 求梯度。
4. **(选做)** 如果把 `ffn_dim` 设成 0 会怎样？这个 block 还能跑吗？loss 还能下降吗？
5. **(选做)** Lab04 做出的 attention 是"加权和"——它本质是个线性组合。那 FFN 给模型加了什么 attention 没有的能力？提示：激活函数 + 维度扩张。

## 上节 / 下节

- **上节**：[Lab04 — Multi-Head Attention](/course/practice/labs/lab04-step3/)：做出多头因果注意力。本 lab 直接调用 `attention_forward` 把它装进 Block。
- **下节**：[Lab06 — 完整 GPT 模型](/course/practice/labs/lab06-step5/)：把 N 个 Block 串起来，加 embedding 和 LM Head，做成完整 GPT。
