---
title: 'Lab04 - 多头自注意力'
layout: 'course_page'
permalink: '/course/practice/labs/lab04-step3/'
lab: 'Lab04'
step: 'step3'
hours: '3h'
deliverable: '完成 framework/student.c 中本章 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab04-step3/TASK.md'
---

> **实验编号** Lab04 **预计耗时** 3h **对应 step** [step3](https://github.com/Luyoung0001/miniLLM/tree/main/step3) **本节产出** 补完本章 TODO，并让验证输出相对当前实验基线继续转绿


## 实验目的

完成本 lab 后，你应当：

1. 能用一句话解释 Q / K / V 在 attention 里各自的角色（谁问、谁答、内容是什么）。
2. 能口述 `Attention(Q,K,V) = softmax(Q @ K^T / sqrt(head_dim)) @ V` 中**每一步在做什么**。
3. 能独立写出一个**单头** attention：`scores = Q @ K^T * scale` → `+ mask` → `softmax` → `@ V`。
4. 能验证 softmax 之后 attn 矩阵的**每行和 = 1**、上三角（被 mask 的位置）权重为 0。
5. 能解释为什么 `head_dim` 出现在缩放因子分母上，以及用 `-1e9f` 而不是 `-INFINITY` 的实际原因。

本 lab 不要求你训练权重，不要求写反向传播；也不要求实现 `attention_forward` 整个多头循环——你只需要把 attention 的"三步核心"用 student_xxx 函数独立写出来。完整的多头前向 `attention_forward` 由 `framework/attention.c` 提供，你调用即可。

## 为什么现在轮到 attention

走到这一章之前，你已经做完了两件很关键的准备工作。`Lab02` 把原始文本变成了离散 token，`Lab03` 又把这些 token 变成了带位置信息的向量。可问题在于，这些向量虽然已经不再是“裸字符”，却依然主要在表达“我自己是谁”。如果模型永远只能看当前位置本身，它还是不知道一句话里别的位置已经说了什么，也就谈不上真正理解上下文。

attention 的作用，就是让每个位置在更新自己之前，先看一眼序列中的其他位置。换句话说，这一章不是在额外增加一个复杂公式，而是在第一次把“上下文交互”真正写进模型里。后面所有 block、完整 GPT、聊天循环，都会默认这一层能力已经存在。

课程没有一上来就让你写完整多头前向，而是先把 attention 拆成三个 student 函数：算分数、加 mask、做 softmax。这样做的原因很直接：你先把最核心的三段数学看透，后面的多头包装和线性投影就只是组合问题，而不会继续显得像黑箱。

## 实验环境

涉及的文件：

```
labs/lab04-step3/
├── TASK.md                # 你正在读的
├── Makefile               # 编译入口
├── framework/
│   ├── student.h          # 学员实现的函数声明
│   ├── student.c          # ★ 你要改的文件（含 TODO 注释）
│   ├── verify.c           # 自动验证（你不需要改）
│   └── verify.h
├── tests/                 # （空，本 lab 用纯函数验证）
```

可以调用的 framework API（来自 `framework/tensor.h`、`framework/math_ops.h`、`framework/attention.h`）：

- `tensor_zeros / tensor_create_2d / tensor_get / tensor_set / tensor_fill_data / tensor_free`
- `matmul / matmul_transposed_b`
- `softmax / softmax_inplace`
- `create_causal_mask(seq_len)` 返回 `[seq_len, seq_len]` 的下三角 mask

命令：

```bash
cd labs/lab04-step3
make clean && make test
make clean      # 清理 build/ 和 student
```

> `framework/attention.c` 已经提供了完整的 `attention_forward`。你这一章只需要把 attention 里 3 个最核心的数学片段写清楚。

## 实验内容

本 lab 含 3 个子任务，对应 student.c 里的 3 个 TODO 函数。每个函数都要靠 `framework/` 的公开 API 实现——不依赖任何 framework 内部状态。

这三个 task 不是三道彼此无关的小题，而是一条完整 attention 链路的拆解。第一步决定“谁应该看谁”，第二步决定“谁现在还不能看未来”，第三步才把分数变成真正可用的概率权重。你写每一个 TODO 时，都应该知道它在整条链里的位置。

### 1.5.1 单头 attention scores：`student_attention_scores`

**对应 TODO**：`student.c` → `student_attention_scores`

**action**：实现 `Q @ K^T * scale`，输入 Q、K 都是 `[seq_len, head_dim]` 的 2D 张量，输出 `out` 也是 `[seq_len, seq_len]`。

```c
// 单头 attention 分数: out[i][j] = scale * sum_d Q[i][d] * K[j][d]
int student_attention_scores(Tensor* Q, Tensor* K, float scale, Tensor* out);
```

**context**：这是 attention 公式的第一步。Q 的第 i 行代表"位置 i 在问什么"，K 的第 j 行代表"位置 j 的标签"；它们的点积就是"匹配度"。最后乘以 `scale = 1/sqrt(head_dim)`。

**check**：用 `Q = [[1,0], [0,1]]`、`K = [[1,0], [0,1]]`、`scale = 0.5` 跑一次（见 verify 的 [TEST 1]），应得到 `[[0.5, 0], [0, 0.5]]`（允许 1e-4 浮点误差）。

**hint**：
- 循环三层：`for i in seq_len: for j in seq_len: for d in head_dim:`。
- 不要把 K 转置——按 `K[j][d]` 读即可（行 j 相当于"被问的那个位置"）。
- `Q->shape[0]` = seq_len，`Q->shape[1]` = head_dim；索引用 `data[i * head_dim + d]` 这种一维偏移更简单。

这一题真正想让你建立的直觉是：`Q @ K^T` 不是一条抽象矩阵恒等式，而是“位置 i 正在给所有候选位置 j 打分”。以后你再看到这条公式，脑子里应该先出现“谁在问、谁在被比较”，而不只是符号推导。

### 1.5.2 因果掩码：`student_apply_mask`

**对应 TODO**：`student.c` → `student_apply_mask`

**action**：把 mask 加到 scores 上（**原地**修改 scores）。mask[i][j] = 0 表示"保留"，mask[i][j] 是个很大的负数表示"屏蔽"。

```c
// 原地: scores[i][j] += mask[i][j]
void student_apply_mask(Tensor* scores, Tensor* mask);
```

**context**：因果掩码让位置 i 只能"看"位置 0..i。加在 softmax **之前**——这样 softmax 之后被 mask 的位置权重自然变成 0。

**check**：[TEST 2] 先 `student_attention_scores` 得到全零矩阵（Q=K=全零），再 `student_apply_mask` 加一个 `seq_len=3` 的因果掩码，应得到 `[0, -1e9, -1e9; 0, 0, -1e9; 0, 0, 0]`。

**hint**：
- 一个循环搞定：`for (int i = 0; i < scores->size; i++) scores->data[i] += mask->data[i];`
- 注意是**原地**修改 scores，不要 new tensor。

这里最值得记住的一点，是 mask 必须加在 softmax 之前。只有这样，被屏蔽的位置才会在概率化之前就被压到极小，softmax 后自然接近 0。如果你把它放到 softmax 之后，那些非法位置其实已经参与过归一化了，语义就完全不一样。

### 1.5.3 沿最后一维 softmax：`student_softmax`

**对应 TODO**：`student.c` → `student_softmax`

**action**：对每一行（最后一维）做 softmax，使得每行和 = 1。

```c
// 沿最后一维 softmax: out[row][j] = exp(in[row][j]) / sum_k exp(in[row][k])
void student_softmax(Tensor* out, Tensor* in);
```

**context**：softmax 把"匹配度"变成"概率"。需要先减去最大值（数值稳定 trick），再 exp，再归一化。

**check**：[TEST 3] 给定 `seq_len=3` 的输入，全 0 的 in → 每行全 1/3 的 out（允许 1e-5 误差）。[TEST 4] 串起来：scores → mask → softmax，验证每行和 = 1、上三角权重 < 1e-6。

**hint**：
- 先用 `in->shape[in->ndim - 1]` 拿到最后一维长度 = `last_dim`。
- 外层循环每一行；对每行：找 max → 算 `exp(x - max)` → 求和 → 归一化。
- `out->data[i] = exp(in->data[i] - max) / sum;`

softmax 这一步虽然在 `Lab01` 已经作为数值稳定练习出现过，但现在它第一次回到了真实模型链路里。这里每一行和为 1，不只是为了通过一个数学检查，而是为了让后续 `attn @ V` 真正可以被解释成“按概率权重做加权求和”。

## 现象

`Lab04` 的 verify 比前几章更细，它不是简单的 4 个大测试，而是把每个阶段拆成很多小断言。所以你第一次运行时看到的很可能不是“全红”或“全绿”，而是**一半通过、一半失败**。这正说明框架已经把 attention 的各个观察点拆开了。

### 当前实验基线（2026-06-17 复核）

```text
row 2 sum = 0.000000
row 3 sum = 0.000000
[FAIL] attn 行和 = 1
[PASS] attn 上三角 = 0（mask 生效）
结果: 11 失败 / 11 通过
```

### 完成本章 TODO 后的目标输出

```text
========================================
   Lab04 多头自注意力 — verify
========================================

[TEST 1] student_attention_scores 基础形状与缩放 ...
  [PASS] 输出形状为 [3, 3]
  [PASS] Q @ K^T * 0.5 在 (0,0) 等于 0.5

[TEST 2] student_apply_mask 因果掩码 ...
  [PASS] mask 后 [0][0] = 0
  [PASS] mask 后 [0][1] = -1e9
  [PASS] mask 后 [1][1] = 0
  [PASS] mask 后 [2][2] = 0

[TEST 3] student_softmax 行和 = 1 ...
  [PASS] 全 0 输入 softmax 后每行和 ≈ 1.0
  [PASS] softmax([1, 1, 1]) 每行 = 1/3

[TEST 4] 端到端: scores → mask → softmax ...
  [PASS] attn 行和 = 1
  [PASS] attn 上三角 = 0（mask 生效）

========================================
结果: 4 个 TEST 全 PASS
========================================
```

这里真正的目标不是“凑够 3 个 PASS”，而是把每个小断言都逐步转绿，直到 `attn` 的行和和 mask 都成立。

这一章结束时，最理想的状态不是“我记住了 3 个函数名”，而是你已经能指着 verify 的每一项输出，说清楚它究竟在证明 attention 链路中的哪一步已经成立。只有做到这一点，后面进入 Transformer Block 时你才不会再次把 attention 当成黑箱。

## 思考题

1. **(必做)** 把 `student_attention_scores` 里的 `scale = 1.0f` 跑一次（即不缩放），观察 softmax 后的权重矩阵——是不是几乎变成 one-hot？为什么？再改回 `1/sqrt(head_dim)`。
2. **(必做)** 在 `student_apply_mask` 里把 `-1e9f` 改成 `-INFINITY`（注意加 `#include <math.h>`），观察 softmax 后的行为。NaN 出现了吗？为什么用 `-1e9f` 更安全？
3. **(选做)** 把 `student_attention_scores` 用 `framework/math_ops.h` 里的 `matmul_transposed_b` 实现一次（2~3 行就能搞定），对比你手写三层循环的版本——速度上能差几倍？提示：用 `clock()` 包住测试循环。
4. **(选做)** 解释：`hidden_dim = 64, num_heads = 4` 时，参数总量是 `4 * 64 * 64 = 16384`。如果 `num_heads = 1`（`head_dim = 64`），参数总量变吗？表达力变吗？

## 上节 / 下节

- **上节**：[Lab03 - 词嵌入与位置编码](/course/practice/labs/lab03-step2/)：你做出了 `[seq_len, hidden_dim]` 的输入矩阵；每个位置只知道"自己"。
- **下节**：[Lab05 - Transformer Block](/course/practice/labs/lab05-step4/)：在 attention 的外面再包一层 FFN + LayerNorm + 残差连接，就成了 Transformer Block。
