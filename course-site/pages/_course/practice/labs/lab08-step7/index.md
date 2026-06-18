---
title: 'Lab08 - 文本生成'
layout: 'course_page'
permalink: '/course/practice/labs/lab08-step7/'
lab: 'Lab08'
step: 'step7'
hours: '2h'
deliverable: '完成三种采样策略 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab08-step7/TASK.md'
---

> **实验编号** Lab08 **预计耗时** 2h **本节产出** 补完 greedy、temperature、top-k 三个采样函数，并让验证输出相对当前实验基线继续转绿


## 实验目的

Lab07 跑完训练后，你手里应该有一份 `model.bin`——里面装着 24 万个 float，看起来"懂一点事"。但这堆数字自己不会说话。

本 lab 要解决"**怎么把一串 logits 变成一个 token**"——也就是采样。`generate.c` 里 4 个采样函数（greedy / temperature / top-k / top-p）和自回归循环一起，构成了"让模型开口说话"的全部。

你会亲手写出 3 个核心采样函数：

1. **Greedy 采样** `student_sample_greedy`：永远挑 logits 里最大的那个。
2. **Temperature 采样** `student_sample_temperature`：先把 logits 除以 T 拉尖或摊平，再 softmax 抽样。
3. **Top-K 采样** `student_sample_top_k`：只在前 K 个最高 logit 里掷骰子。

完成后你应当能口述：

- 自回归循环：`encode(prompt)` → 循环里 `model_forward` → 取最后一行 → `sample(...)` → 拼到末尾 → 直到 EOS 或上限。
- 温度 < 1 让分布"拉尖"（更确定），> 1 让分布"摊平"（更随机）。
- Top-K = 0 等于关闭，Top-K = vocab_size 等于不限制。
- "未训练就 generate 出乱码"是预期——模型没学，logits 几乎均匀。

## 为什么训练完还不能直接聊天

`Lab07` 跑完之后，模型终于不再只是随机权重了。但这并不意味着它已经自动拥有“说一句完整话”的能力。训练阶段产出的，是对整个词表的一组 logits，也就是“每个候选 token 现在各有多大分数”。分数本身不会自动变成答案，答案要靠采样策略真正挑出来。

这就是本章存在的理由。模型前向负责给出“它倾向于说什么”，采样负责决定“这一轮到底说哪一个”。同一组 logits，用 greedy、temperature、top-k 三种策略去消费，得到的文本风格会明显不同。所以从这一章开始，你要把“模型质量”和“输出策略”视为两层不同的问题，而不是混在一起判断。

## 实验环境

涉及的文件：

```
labs/lab08-step7/
├── TASK.md                # 本文件
├── Makefile
└── framework/
    ├── student.c          # ★ 你要改的文件（含 TODO 注释）
    ├── student.h
    ├── verify.c           # 自动验证程序
    └── verify.h
```

可调用的 framework API（声明在 `minillm_lab/framework/*.h`）：

- `tensor.h`：`tensor_create_1d()` / `tensor_zeros()` / `tensor_set_flat()` / `tensor_get_flat()` / `tensor_free()` / `tensor_argmax()`
- `math_ops.h`：`softmax()` / `softmax_inplace()` / `tensor_mul_scalar()` / `tensor_max()` / `tensor_argmax()`
- `generate.h`：`default_generate_config()` / `greedy_config()` / `creative_config()` —— **不要直接调 framework 的 `sample_*`**，你写自己的版本

命令：

```bash
cd labs/lab08-step7
make clean && make test
```

## 实验内容

本 lab 共有 3 个子任务，全部集中在 `framework/student.c` 里。骨架已写在文件中，你只填 TODO 即可。

这三题的顺序是有意设计的。greedy 先帮你建立“完全确定地挑最大值”的基线；temperature 再告诉你“同一组 logits 可以被拉尖或摊平”；top-k 最后才补上“并不是所有尾部 token 都值得进入抽样候选集”。每往后一步，都是在前一步的基础上增加一个更细的控制旋钮。

### 任务 1.5.1：填 `student_sample_greedy`（贪心 = argmax）

打开 `framework/student.c`，找到 `student_sample_greedy` 函数。

它接收一个 `[vocab_size]` 的 logits 张量，要返回**最大 logit 对应的 token id**。

```
argmax(logits) = i 满足  logits[i] >= logits[j]  对所有 j
```

**TODO 位置**：`student.c` 内 `student_sample_greedy` 函数体。

**action**：写一个 for 循环——遍历 `vocab_size` 个位置，记录当前最大值 `best_val` 和对应的 `best_idx`，遇到更大的就更新。最后 `return best_idx;`。提示：可以直接 `tensor_get_flat(logits, i)` 取值；或者用 framework 提供的 `tensor_argmax(logits)`（但**自己写**更利于理解）。

**预计 5~10 行**。

greedy 虽然最朴素，却是理解后两种策略的基准。如果你对“完全不掷骰子、直接选最大分数”这件事没有牢靠直觉，后面再谈温度和 top-k 只会变成参数游戏。所以这一题做完后，建议你先在脑子里固定一句话：greedy 的本质就是“完全确定地消费 logits”。

### 任务 1.5.2：填 `student_sample_temperature`（温度 = logits / T 再 softmax）

打开 `framework/student.c`，找到 `student_sample_temperature` 函数。

数学（两步）：

```
1. scaled[i] = logits[i] / temperature        # 数值上等价于 softmax 的"放大/缩小"
2. probs = softmax(scaled)                     # 数值稳定版（framework 的 softmax 已自带）
3. 按 probs 抽样（用累积分布 + rand()/RAND_MAX）
```

温度的物理含义：

- `T = 0.1` → 分布"拉尖"成接近 one-hot，几乎必选最大那个。
- `T = 1.0` → 原始分布。
- `T = 2.0` → 分布"摊平"，更随机。
- `T <= 0` → 直接当 greedy（early return `student_sample_greedy(logits)`）。

**TODO 位置**：`student.c` 内 `student_sample_temperature` 函数体。

**action**：

1. 先 malloc 一份 scaled（用 `tensor_create_1d(vocab_size)` + `tensor_set_flat` 写入 `logits[i] / T`）。
2. 申请 probs 张量，调 `softmax(probs, scaled)`。
3. 计算累积和 `cum[i] = sum_{j<=i} probs[j]`，抽 `u = rand() / (float)RAND_MAX`，找到第一个 `cum[i] >= u` 的 i 返回。
4. 释放 scaled、probs。

**预计 12~18 行**。

这一题真正想教你的，是“随机性也是可以被设计的”。温度不会改变 logits 的排序，但会改变排序之间的差距被看得有多重。以后如果你觉得模型回复太死板，第一反应应该想到“是不是分布太尖”，而不是先怀疑模型一定没学会。

### 任务 1.5.3：填 `student_sample_top_k`（保留前 K 个最高 logit）

打开 `framework/student.c`，找到 `student_sample_top_k` 函数。

数学（两步）：

```
1. 找 top-K：保留 logits 里最大的 K 个，其余置为 -infinity
2. 在过滤后的 logits 上做温度采样（复用 1.5.2 的思路：softmax + 累积抽样）
```

边界：

- `k <= 0` 或 `k >= vocab_size`：等价于不过滤，直接做温度采样（再退化到 greedy 当 T 极小）。
- 用 `tensor_set_flat(t, i, -INFINITY)` 屏蔽掉非 top-K。

**TODO 位置**：`student.c` 内 `student_sample_top_k` 函数体。

**action**：

1. **找 top-K 阈值**——可以用"第 K 大"作为门槛：循环 `k` 次，每次在剩余 logits 里找 argmax，记下阈值 `threshold`（最后那个被选中的最小值就是）。
2. 复制 logits 到 `filtered`，对每个 `i`：若 `filtered[i] < threshold` 则置 `-INFINITY`，相等时**只**保留前 K 个（用计数 trick）。
3. 在 `filtered` 上调 1.5.2 的"softmax + 累积抽样"逻辑。

或者用更直接的策略：复制 logits → 排序（用 `qsort`）→ 前 K 保留、其余 `-INF` → 抽样。

**预计 12~20 行**。

> 三个任务**不是孤立的**：1.5.3 应当复用 1.5.2 的 softmax+累积抽样，1.5.2 应当复用 1.5.1 的 argmax。把代码组织成可重用的样子，是这章的隐形考点。

top-k 的意义在于，它第一次把“不要让明显不靠谱的长尾 token 参与竞争”明确写进了策略。对小模型尤其如此：模型本来就不够强，如果还让大量极低分候选进入抽样，输出质量会被尾部噪声明显拖低。也就是说，这一题不是在追求更复杂，而是在追求更可控。

## 现象

`Lab08` 是一个很典型的“半绿半红”实验。共享的采样框架、部分边界处理和基础张量操作已经连通，所以第一次运行时你会看到不少测试先通过，但真正和 `student_sample_*` 相关的行为还没全部成立。

### 当前实验基线（2026-06-17 复核）

```text
测试结果: 8 通过, 12 失败
```

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tensor.c -o build/tensor.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/math_ops.c -o build/math_ops.o
...
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o \
    build/tensor.o build/math_ops.o build/embedding.o build/attention.o build/layernorm.o \
    build/ffn.o build/transformer.o build/model.o build/generate.o
========================================
       Lab08 - 文本生成 测试
========================================

=== TEST 1: student_sample_greedy ===
  [TEST 1] argmax picks position of max logit ........ [PASS]
  [TEST 1] argmax handles tied values (first wins) .... [PASS]
  [TEST 1] argmax works on a uniform distribution ..... [PASS]

=== TEST 2: student_sample_temperature ===
  [TEST 2] T=0.1 always returns max token ............. [PASS]
  [TEST 2] T=1.0 ~ original distribution .............. [PASS]
  [TEST 2] T=2.0 spreads probability (low mass) ...... [PASS]
  [TEST 2] T=0 falls back to greedy ................... [PASS]

=== TEST 3: student_sample_top_k ===
  [TEST 3] K=0 falls back to temperature sample ...... [PASS]
  [TEST 3] K>=V equivalent to no filter .............. [PASS]
  [TEST 3] K=2 only samples within top-2 .............. [PASS]
  [TEST 3] deterministic on tied top .................. [PASS]

========================================
测试结果: 11 通过, 0 失败
========================================
```

如果哪一行 `[FAIL]`，对照检查：

- `[TEST 2] T=0.1 always returns max token` 失败 → 检查 `softmax` 后 probs 是否数值稳定（不应该出现 `probs = NaN`）。framework 的 `softmax()` 用减 max 技巧，但你的 `logits[i] / T` 在 `T=0.1` 时也可能溢出——必要时用 `softmax_inplace`。
- `[TEST 3] K>=V equivalent to no filter` 失败 → 检查 `k` 边界是不是 `k >= vocab_size` 早 return，而不是落到 top-K 过滤。
- 累积抽样总返回 0 → 检查 `cum[i] >= u` 是 `<` 还是 `<=`；以及 `rand() / RAND_MAX` 是否可能等于 1.0（不可能，但 `u=1.0` 会让所有 i 都不满足——所以最后一个 i 用特判 `if (i == vocab_size - 1) return i;` 比较稳）。

本章收尾时，你最好已经能清楚分开两层判断：权重决定 logits 的大致形状，采样决定你怎样消费这份形状。把这两层分开看，后面进入聊天章节时，你才不会把“回复风格怪”一股脑都归因到模型没学会。

## 思考题

### 思考题 1（必做）

为什么 `T=0.1` 时模型几乎必选 logits 最大的那个 token？写出数学：`logits = [0, 0, 2]`、`T=0.1`，scaled = `[0, 0, 20]`，softmax 后 token 2 的概率约多少？  
提示：`softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)`，`max=20`，所以 `exp(0-20) ≈ 1.4e-9`、`exp(20-20) = 1`——总和几乎就是 1。

### 思考题 2（必做）

如果 model.bin **完全没训过**（随机权重），`generate` 出来是 `?#$%` 这种乱码——这是 bug 还是预期？为什么？  
提示：看 `model_init_random(..., 0.02f)` 的 std——初始权重很小，logits 几乎均匀；采样 = 在 vocab 上"扔骰子"。这不是 bug，是"模型还没学"的可观察信号。

### 思考题 3（选做）

Top-K 跟 Top-P 的本质区别是什么？为什么"长尾分布"（一个 token 占 0.9、其余每个 0.0001）下 Top-P=0.9 只选 1 个，而 Top-K=10 会把 1 个大概率 + 9 个几乎不可能的混在一起？  
提示：Top-P 是"按概率**累积**选"，Top-K 是"按排名选"。前者动态，后者固定。

### 思考题 4（选做）

为什么 `sample_combined`（framework 的实现）的顺序是"先 K 后 P 再温度"，不能反过来？  
提示：想象先做温度 `T=0.1` 把分布拉成 one-hot，再做 Top-K=10——此时前 10 名里只有 1 个概率为 1，其余几乎为 0；再做 Top-P=0.9 就只剩 1 个 token。这等价于 greedy。顺序决定"组合后还剩下多少候选"。

## 上节 / 下节

- **上节**：[Lab07 - 训练循环](/course/practice/labs/lab07-step6/)：loss 收敛、`model.bin` 写盘了——但它自己不会说话，本 lab 让它开口。
- **下节**：[Lab09 - 多轮对话](/course/practice/labs/lab09-step8/)：把"单轮 prompt"升级成"多轮聊天"——加上下文、加模板、加长度截断。
