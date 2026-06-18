---
title: 'Lab07 - 训练循环'
layout: 'course_page'
permalink: '/course/practice/labs/lab07-step6/'
lab: 'Lab07'
step: 'step6'
hours: '3h'
deliverable: '完成交叉熵、梯度、Adam 三个 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab07-step6/TASK.md'
---

> **实验编号** Lab07 **预计耗时** 3h **本节产出** 补完交叉熵、softmax+CE 梯度、Adam 更新三个接口，并让验证输出相对当前实验基线继续转绿


## 实验目的

Lab06 你把模型装好了、还能 save/load。但 `model.bin` 里装的是**随机权重**——前向出来的 logits 完全是"瞎猜"，loss ≈ log(vocab_size) ≈ 5 到 6。

本 lab 要让模型"学会"点什么。你会亲手写出 3 段在 `framework/loss.c` 和 `optimizer.c` 里看起来像"魔法"、其实是固定数学公式的代码：

1. **交叉熵损失** `cross_entropy_loss`：模型在 vocab 上的猜测分布和"正确答案"之间的"距离"。
2. **Softmax + 交叉熵梯度** `softmax_cross_entropy_grad`：一个比"对 softmax 内部指数求导"短得多的公式。
3. **Adam 一步** `adam_step`：维护 m / v 两个移动平均做"自适应学习率"。

完成后你应当能口述：

- 训练循环四步：**forward → loss → backward → optimizer.step**。
- 为什么 `loss` 的下界是 0、初始值大致是 `log(vocab_size)`。
- 为什么 `learning_rate=1.0` 时 loss 会发散成 `NaN`（`grad_clip=1.0` 是安全网）。
- Adam 维护**两个**移动平均 `m`、`v` 的原因（对比 SGD 只维护一个 velocity）。

## 为什么现在要把训练数学亲手写一遍

前一章你已经能创建并保存完整模型了，但那个模型仍然只是“一个会前向的随机函数”。随机函数当然可以吐出 logits，可它不会因为你希望它学会语言，就自动往正确方向改变。训练章节的作用，就是把“模型存在”推进到“模型开始被数据推动着改变自己”。

很多初学者会把这一章的三个名字看得很散：交叉熵像损失函数，梯度像反向传播，Adam 像优化器，仿佛三件互不相关的事情。其实课程现在把它们放在一起，正是为了让你看到它们是一条连续链。交叉熵告诉你“错了多少”，梯度告诉你“朝哪边改”，Adam 则把“该往哪边改”真正落实成参数数组上的移动。

## 实验环境

涉及的文件：

```
labs/lab07-step6/
├── TASK.md                # 本文件
├── Makefile
└── framework/
    ├── student.c          # ★ 你要改的文件（含 TODO 注释）
    ├── student.h
    ├── verify.c           # 自动验证程序
    └── verify.h
```

可调用的 framework API（声明在 `minillm_lab/framework/*.h`）：

- `loss.h`：`cross_entropy_loss()` / `cross_entropy_loss_with_grad()` / `cross_entropy_grad()` / `softmax_cross_entropy_grad()`（这里更适合把公式自己写一遍）
- `optimizer.h`：`AdamOptimizer` 结构 + `adam_create()` / `adam_step()` / `adam_free()`（**不要直接调 `adam_step`**——你写自己的）
- `tensor.h` / `math_ops.h`：`tensor_zeros()` / `tensor_set_flat()` / `tensor_get_flat()` / `tensor_print()`
- `model.h` / `config.h`：`GPTModel` + `default_config()`

命令：

```bash
cd labs/lab07-step6
make clean && make test
```

## 实验内容

本 lab 共有 3 个子任务，全部集中在 `framework/student.c` 里。骨架已写在文件中，你只填 TODO 即可。

你可以把这三题看成训练循环的最小剖面图。第一题给出损失值，第二题给出损失对 logits 的方向，第三题再把这个方向变成真正的参数更新。把它们连起来，你才真正拥有了 `forward -> loss -> backward -> update` 这条主线。

### 任务 1.5.1：填 `student_cross_entropy_loss`（序列平均 CE）

打开 `framework/student.c`，找到 `student_cross_entropy_loss` 函数。

它接收一个 `[seq_len, vocab_size]` 的 logits 张量、一个 `[seq_len]` 的目标 token id 数组，要返回**整个序列的平均**交叉熵损失（标量 float）。

回忆数学：

```
对单个位置 s:
  L_s = -log( P(target_s) )
      = -log( softmax(logits_s)[target_s] )
      = -logits_s[target_s] + log_sum_exp(logits_s)   # 数值稳定版

对整个序列:
  L = mean(L_s, s = 0..seq_len-1)
```

数值稳定版：先找 `max_val = max(logits_s)`，再算 `log_sum_exp = max_val + log(sum(exp(logits_s - max_val)))`。

**TODO 位置**：`student.c` 内 `student_cross_entropy_loss` 函数体。

**action**：写一个嵌套循环——外层对 seq_len，内层对 vocab_size 计算 `log_sum_exp`，再叠 `-logits[target] + log_sum_exp` 到累加器。最后 `return total / seq_len;`。

这一题最值得体会的一点是：交叉熵并不神秘，它只是把“模型给正确答案分了多大概率”重新写成了一个适合数值稳定计算的形式。以后你再看 loss 曲线时，应该能立刻把它翻译成“正确 token 的概率有没有被推高”。

### 任务 1.5.2：填 `student_softmax_ce_grad`（单位置 softmax - one_hot）

打开 `framework/student.c`，找到 `student_softmax_ce_grad` 函数。

它对**单个位置**算梯度。输入是 `logits[vocab_size]` 和 `target`（一个 int），输出 `grad[vocab_size]`。

数学上有个**特别简洁**的形式（这就是为什么大家喜欢 softmax + cross-entropy 一起算）：

```
grad[i] = softmax(logits)[i] - (i == target ? 1 : 0)
```

实现步骤（参考 `framework/loss.c` 的 `softmax_cross_entropy_grad`，但**自己写**）：

1. 找 `max_val = max(logits)`（数值稳定）。
2. 对每个 i 算 `exp(logits[i] - max_val)` 存到 `grad[i]`，同时累加 `sum`。
3. 把 `grad[i] /= sum`（这是 softmax 的输出）。
4. 如果 `i == target`，`grad[i] -= 1`（减 one-hot）。

**TODO 位置**：`student.c` 内 `student_softmax_ce_grad` 函数体。

**action**：写 3 个 for 循环（找 max → exp+sum → 归一化并减 1）。提示：可以**复用**任务 1.5.1 里的 `max` 和 `log_sum_exp` 思路——把 `exp(x - max) / sum` 算出来再判断 `i == target` 减 1。

这一步很适合建立一个更健康的判断：不少看起来吓人的训练数学，最终落进代码时反而极其短。softmax 单独求导确实麻烦，但和交叉熵绑在一起之后，结果直接收缩成 `softmax - one_hot`。课程故意让你亲手写它，就是为了让你把这种“复杂理论在实现里会简化”的经验真正拿到手。

### 任务 1.5.3：填 `student_adam_step`（对单个张量应用 Adam）

打开 `framework/student.c`，找到 `student_adam_step` 函数。

**这是单个张量版本**——比 `framework/optimizer.c` 的 `adam_step`（要处理 16+ 个张量）简单很多，但**核心数学公式完全一样**。

签名：

```c
/* 对单个 param 张量应用一次 Adam 更新
 * param / m / v / grad 都是同 shape 的 [size] 一维数组
 * t 是当前时间步（从 1 开始）
 * 写 8~12 行
 */
int student_adam_step(
    float* param, float* m, float* v, const float* grad,
    int size, float lr, float beta1, float beta2,
    float eps, float weight_decay, int t
);
```

Adam 五行公式（按顺序）：

```
m_i = beta1 * m_i + (1 - beta1) * grad_i        # 一阶矩
v_i = beta2 * v_i + (1 - beta2) * grad_i²       # 二阶矩
m_hat = m_i / (1 - beta1^t)                      # 偏差校正 1
v_hat = v_i / (1 - beta2^t)                      # 偏差校正 2
param_i -= lr * m_hat / (sqrt(v_hat) + eps)     # 参数更新
param_i -= lr * weight_decay * param_i          # AdamW 权重衰减
```

**TODO 位置**：`student.c` 内 `student_adam_step` 函数体。

**action**：写一个 for 循环对每个 i 依次执行上面六步（最后两步都是"减"，可以合成一行）。提示：t 是一次调用的**常数**，偏差校正 `1 - beta1^t` 和 `1 - beta2^t` 在循环外算好再传进去；用 `sqrtf()` 不是 `sqrt()`。

Adam 这一题是本章最偏工程实现的一部分。它提醒你：训练从来不只是“数学上知道梯度存在”就结束了，真正好不好训，还取决于你怎样把梯度转成稳定的参数移动。后面你调学习率、加权重衰减、看 loss 是否震荡，本质上都还在和这一题里的几个量打交道。

## 现象

`Lab07` 的初始状态不是全红。因为 verify 里有一部分检查只依赖共享框架或非常宽松的数值性质，所以第一次运行时你会先看到少量通过、再看到大部分失败。

### 当前实验基线（2026-06-17 复核）

```text
测试结果: 3 通过, 11 失败
```

这说明训练循环的数学核心还没补上，但编译链、verify 入口和部分外层检查已经正常。

### 完成本章 TODO 后的目标输出

```
$ make clean && make test
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/loss.c -o build/loss.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/optimizer.c -o build/optimizer.o
...
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o build/tensor.o build/loss.o build/optimizer.o ...
========================================
       Lab07 - 训练循环 测试
========================================

=== TEST 1: student_cross_entropy_loss ===
  [TEST 1] cross_entropy > 0 ........................ [PASS]
  [TEST 1] loss ~ log(vocab_size) for uniform logits  [PASS]
  [TEST 1] loss decreases as target logit grows ..... [PASS]

=== TEST 2: student_softmax_ce_grad ===
  [TEST 2] grad sums to ~0 .......................... [PASS]
  [TEST 2] grad at target is negative ............... [PASS]
  [TEST 3] grad matches framework reference ......... [PASS]

=== TEST 3: student_adam_step ===
  [TEST 3] adam decreases loss ...................... [PASS]
  [TEST 3] adam updates m and v ..................... [PASS]

========================================
测试结果: 8 通过, 0 失败
========================================
```

如果某一行 `[FAIL]`，对照检查：

- `cross_entropy_loss` 算出来是负数？检查 `log_sum_exp` 里 `logf` 的参数是正的。
- `softmax_ce_grad` 求和不≈0？检查 `softmax / sum` 步骤里 `sum` 是否被忘加。
- `adam_step` 不收敛？检查 `t` 是否从 1 开始算 `1 - beta1^t`（不要从 0 算——0 会导致除 0）。

这章过关之后，训练这件事就不该再是一个模糊的大词。你应该已经能把它拆成三件具体而可执行的事：先量化错得有多厉害，再算出该往哪边改，最后让参数真的挪一步。下一章之所以能讨论生成，正是因为从这里开始，模型已经不再只是随机权重集合，而是会被数据塑形的对象。

## 思考题

### 思考题 1（必做）

为什么 `loss_confident`（target 位置 logit 远大于其他）→ 损失接近 0，而 `loss_uncertain`（所有 logit 接近相等）→ 损失接近 `log(vocab_size)`？  
提示：从 `L = -log P(target)` 出发，P=1 和 P=1/V 各会得到什么 L？

### 思考题 2（必做）

Adam 的 `m` 和 `v` 是不是"越多越好"？为什么要存**两个**移动平均，而不是只存一个？  
提示：对比 `framework/optimizer.c` 的 `SGD` 那部分——它的 `velocity` 是一阶还是二阶？Adam 多存的那个 `v` 让"学习率"变成什么样（per-parameter 自适应）？

### 思考题 3（选做）

如果 `grad_clip=1.0` 改成 `grad_clip=0`（不裁剪），小学习率（如 1e-3）下 loss 是不是就**一定**不会 NaN？为什么？  
提示：梯度范数偶尔可以很大（特别是某 batch 出现罕见 token），单次 step 的更新量 `lr * m_hat / sqrt(v_hat)` 就可能把参数推到数值不稳定区。

### 思考题 4（选做）

为什么 `BackwardCache` 一定要在前向时存 `embed_output`？不存会怎样？  
提示：反向时算 `d_token_embedding` 需要知道"哪些 token 出现过"以及"它们的 hidden 向量是什么"——这两个信息都来自前向。

## 上节 / 下节

- **上节**：[Lab06 - 完整 GPT 模型](/course/practice/labs/lab06-step5/)：模型能跑、能 save/load，但权重是随机的——下一步让它真能学。
- **下节**：[Lab08 - 文本生成](/course/practice/labs/lab08-step7/)：训练完权重后，从一个 prompt 开始一个 token 一个 token 续写（贪婪、Top-K、Top-P 采样）。
