---
title: 'Chapter 8 — step7 文本生成'
layout: 'course_page'
permalink: '/course/chapters/ch08-step7-generation/'
lab: 'Lab08'
step: 'step7'
hours: '2h'
deliverable: 'Lab08 全部测试通过'
prev_page:
  title: 'Chapter 7 — step6 训练'
  url: '/course/chapters/ch07-step6-training/'
next_page:
  title: 'Chapter 9 — step8 多轮对话'
  url: '/course/chapters/ch09-step8-chat/'
source_path: 'course/chapters/ch08-step7-generation.md'
---

> **对应实践**：[`course/practice/labs/lab08-step7/`](/course/practice/labs/lab08-step7/)  
> **主要修改文件**：`course/practice/labs/lab08-step7/framework/student.c`  
> **验证命令**：`make clean && make test`

到了这里，模型已经具备了一个重要前提：  
它不再只是随机把一堆模块拼起来，而是已经有了“通过 logits 表达偏好”的能力。

但 logits 还不是文本。  
一个语言模型真正“开口说话”的瞬间，不在 `model_forward` 结束的时候，而在你决定：

> 这一拍到底选哪一个 token 作为下一个输出。

这就是生成章的核心。

本章不会再引入新的大模型结构。它要处理的是另一类非常工程化、但又直接决定模型表现的问题：

1. 是永远选分数最高的那个 token，还是允许一点随机性？
2. 如果允许随机性，随机性的范围该怎样控制？
3. 如果词表很大，是否应该只在最有希望的少数候选里抽样？

这些问题的答案，最终都会落到三个需要你实现的函数上：

- `student_sample_greedy`
- `student_sample_temperature`
- `student_sample_top_k`

这三种策略加在一起，就构成了后面所有生成行为的基本决策层。

## 8.1 为什么生成章不是“附属技巧”

很多初学者会把采样策略误看成模型主体之外的一点“调味料”，好像只是在前向后面多接几行代码。  
但从用户实际看到的效果来说，采样几乎直接决定了模型表现的性格。

同样一组 logits：

- 用 greedy，模型会更稳定，也更容易复读；
- 用高温度采样，模型会更发散，也更容易出怪话；
- 用 top-k，模型会只在高分候选里活动，表现更受约束。

所以这一章不是在教一个外围技巧，而是在教：

> 模型内部的概率分布，怎样被转化成外部可见的文本行为。

## 8.2 本章你要建立哪些判断

这一章结束后，你应当能够清楚说出：

1. greedy 为什么本质上就是 `argmax`；
2. 温度为什么等价于“先把 logits 除以一个系数，再做 softmax”；
3. 为什么温度越小，分布越尖锐；温度越大，分布越平坦；
4. top-k 为什么能够砍掉大量低概率尾部候选；
5. 为什么“未训练模型生成乱码”是预期，而不是 bug。

如果这些直觉建立起来，后面无论是 REPL、HTTP 服务还是 BPE 对话，你都能更清楚地判断“问题到底出在模型学得差，还是出在采样策略不合适”。

## 8.3 先看 practice target：这章改哪里

本章实践目录是：

```text
course/practice/labs/lab08-step7/
├── TASK.md
├── Makefile
└── framework/
    ├── student.c      <- 主要修改这里
    ├── student.h
    ├── verify.c       <- 自动验证，不改
    └── verify.h
```

这一章需要你实现的函数，正好对应三种采样策略：

- `student_sample_greedy`
- `student_sample_temperature`
- `student_sample_top_k`

它们的关系不是三道孤立小题，而是一层层往上搭：

1. greedy 是最简单的基线；
2. temperature 是在完整词表上按概率抽样；
3. top-k 是先缩小候选范围，再调用温度采样。

这也是一个很好的工程设计范例：更复杂的策略最好建立在更简单的策略之上，而不是每一题都从头另写一遍。

## 8.4 greedy 为什么是最好的第一步

greedy 采样的定义非常直接：

```text
选 logits 中最大的那个索引
```

也就是 `argmax`。

它的好处是确定性强、最容易调试，也最适合作为采样逻辑的起点。  
因为只要 greedy 都选不对，后面的 temperature 和 top-k 就更没有可信度。

这也是为什么 `Lab08` 的第一组测试会优先检查：

- 最大值位置是否被正确选出；
- 平局时如何处理；
- logits 本身是否被无意修改。

这些看起来简单，但正是在守住最基本的决策层。

## 8.5 温度真正改变的是什么

温度不是“直接改概率”，而是先改 logits 的尺度。

如果写成公式，就是：

```text
scaled_logits[i] = logits[i] / temperature
```

然后再对 `scaled_logits` 做 softmax。

这里最值得讲清楚的是直觉：

- `temperature < 1` 时，所有差距都被放大，分布更尖；
- `temperature = 1` 时，保持原始分布；
- `temperature > 1` 时，差距被压缩，分布更平。

所以温度本质上是在调“模型有多犹豫”。  
温度越低，模型越像在说“我只认最可能那个”；温度越高，模型越像在说“几个候选都可以试试”。

## 8.6 为什么 top-k 是对尾部分布的约束

语言模型的词表通常很大。即使前几个 token 概率已经远高于其它候选，完整 softmax 仍然会给大量低概率 token 留下一点点质量。

如果你让抽样直接在完整词表上发生，就会出现一个问题：  
那些几乎不可能、但仍不为零的候选，仍然有机会偶尔被抽中。

top-k 的思路就是先粗暴地做一层裁剪：

1. 只保留 logits 最高的前 `k` 个；
2. 其余候选全部视作不可选；
3. 再在剩下的这些候选里做 temperature 抽样。

它的作用不是让模型“更聪明”，而是让生成空间更受约束、更不容易掉进词表尾部噪声。

## 8.7 为什么“模型学得差”和“采样策略差”要分开看

本章一个非常重要的课程目标，是让学员别把所有生成问题都归因给模型本身。

如果模型没训练好，那么：

- 不管你怎么调采样，输出都可能很差。

但反过来，如果模型其实已经学到一点东西，而采样策略选得极端，例如：

- 温度过高；
- top-k 太大；
- 没有任何约束；

那么输出同样可能显得混乱。

所以从 Chapter 8 开始，课程要学员建立一个新的调试分层：

1. 先问 logits 有没有学到东西；
2. 再问采样策略是不是把好分布抽坏了。

## 8.8 本章实践步骤

### task 8.1：先读 `student.c` 和 `verify.c`

进入：

```bash
cd course/practice/labs/lab08-step7
```

建议先读：

- `framework/student.h`
- `framework/student.c`
- `framework/verify.c`

当前验证器会分别检查：

1. greedy 是否真的选出最大位置；
2. 温度采样在低温和高温下的统计行为是否合理；
3. top-k 是否真的只在前 `k` 个候选里抽样；
4. 一些边界条件，例如 `temperature <= 0`、`k <= 0` 或 `k >= vocab_size` 时怎样退化。

这说明本章不是只看单次输出，而是在看抽样行为是不是符合概率直觉。

### task 8.2：实现 `student_sample_greedy`

这一题建议完全手写一次 `argmax`。

重点不是代码长短，而是形成习惯：  
采样函数不应该偷偷改输入 logits，本质上它只是读 logits，然后做决策。

### task 8.3：实现 `student_sample_temperature`

这一题的关键步骤有三个：

1. 按温度缩放 logits；
2. 做 softmax；
3. 按累积分布抽样。

建议你在写的时候不断提醒自己：  
temperature 的真正作用发生在 softmax 之前，而不是 softmax 之后。

### task 8.4：实现 `student_sample_top_k`

这一题的重点不是排序技巧，而是清楚：

1. 哪些候选应该保留；
2. 哪些候选应该彻底屏蔽；
3. 筛完以后，仍然要回到 temperature 抽样那套逻辑。

也就是说，top-k 的本质不是另一种全新抽样，而是“先裁剪，再抽样”。

### task 8.5：运行当前真实基线

在还没补完三个函数前，先运行：

```bash
make clean && make test
```

当前 `Lab08` 的真实初始状态是：

- **8 通过，12 失败**。

已经通过的部分，主要来自一些边界测试和“空实现也不会破坏输入”的检查。  
失败的部分则集中在真正与采样决策相关的逻辑上，例如：

- greedy 没有选对最大值；
- 温度采样没有体现出预期的分布趋势；
- top-k 没有把采样范围约束住。

这正说明这章当前的实验起点是健康的：  
框架可以跑，但采样核心逻辑还在等学员完成。

### task 8.6：完成后重新验证

当你补完三个函数后，再执行：

```bash
make clean && make test
```

如果实现正确，你应该看到：

- greedy 的位置选择正确；
- 低温时几乎总选最大项；
- 高温时非最大项获得更多采样质量；
- top-k 只在规定候选里活动。

到这一章结束，模型才真正具备“把内部概率转成外部文本选择”的能力。

## 8.9 常见错误与排查顺序

最常见的错误通常是：

1. greedy 写成了取最小值或平局规则错误；
2. temperature 忘了除温度，或者温度为 0 时没有退化到 greedy；
3. softmax 概率没正确归一化；
4. top-k 没有真正屏蔽词表尾部候选；
5. 抽样函数修改了输入 logits。

建议排查顺序是：

1. 先看 greedy；
2. 再看 temperature 分布；
3. 最后看 top-k 的筛选范围。

因为 top-k 本质上依赖前面两层直觉。

## 8.10 思考题

1. 为什么温度趋近于 0 时，采样行为越来越像 greedy？
2. 为什么温度升高时，低分 token 会获得更多机会？
3. 为什么 top-k 能抑制尾部噪声，但也可能让模型更容易陷入复读？

## 8.11 本章小结

Chapter 8 把课程重心从“模型内部怎样算”推到了“模型外部怎样表现”。

从现在开始，学员应该逐渐形成一个重要判断：  
模型输出的文本，不只是由权重决定，也同样由采样策略决定。

这一步铺好以后，下一章才有意义。因为多轮对话本质上就是把“单次生成”放进带上下文的连续交互里。
