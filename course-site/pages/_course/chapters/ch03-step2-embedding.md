---
title: 'Chapter 3 — step2 Embedding 与位置编码'
layout: 'course_page'
permalink: '/course/chapters/ch03-step2-embedding/'
lab: 'Lab03'
step: 'step2'
hours: '2h'
deliverable: 'Lab03 全部测试通过'
prev_page:
  title: 'Chapter 2 — step1 字符级 Tokenizer'
  url: '/course/chapters/ch02-step1-tokenizer/'
next_page:
  title: 'Chapter 4 — step3 多头自注意力'
  url: '/course/chapters/ch04-step3-attention/'
source_path: 'course/chapters/ch03-step2-embedding.md'
---

> **对应实践**：[`minillm_lab/labs/lab03-step2/`](/course/practice/labs/lab03-step2/)
> **主要修改文件**：`minillm_lab/labs/lab03-step2/framework/student.c`
> **验证命令**：`make clean && make test`

前两章里，输入已经经历了两次关键翻译。

在 Chapter 1，你学会了如何把“多维结构”落实成一块连续内存加一套索引规则。  
在 Chapter 2，你又把“人能读的文本”翻译成了“模型能处理的离散 ID”。

但对模型来说，这还不够。因为注意力、前馈网络、层归一化这些后续模块都工作在浮点向量空间里，而不是工作在离散整数空间里。`76` 这个 token ID 对人来说可以表示 `'H'`，对模型来说却只是一个整数。模型不会直接“乘以一个 token ID”，它只能读取一个向量、变换一个向量、再输出另一个向量。

这就是 embedding 的职责：把离散编号系统，接到连续向量空间上。

## 3.1 本章你真正要建立的直觉

如果只看表面，这章的代码比 attention、transformer block 都简单很多。它无非是在做“查表”和“相加”。  
但从结构上说，这章非常重要，因为它回答了一个贯穿整个模型的问题：

> 一个 token 到底是怎样第一次变成模型内部的数值表示的？

完成本章后，你应当具备下面这些判断：

1. 明白 token embedding 本质上就是一张 `vocab_size × hidden_dim` 的查表矩阵。
2. 明白 position embedding 不是装饰，而是为了让模型知道“同一个 token 出现在第几个位置”。
3. 能解释为什么 `embedding_forward` 不是一种神秘操作，而是“取 token 向量 + 加位置向量”。
4. 知道 sinusoidal position embedding 的基本公式和最容易验证的边界情形：`pos = 0`。
5. 能在本章实验代码中完成 `student_pe_sinusoidal` 和 `student_embedding_forward`。

和前两章一样，这里既不是纯理论，也不是纯填空。它真正要你建立的是一种组合直觉：一层模型经常并不做“复杂算法”，而是在把前面已经准备好的几块结构，按有意义的方式拼起来。

## 3.2 先看 practice target：这章改哪里

本章的主工作区仍然在本章 lab 目录里。

你应该进入：

```text
minillm_lab/labs/lab03-step2/
├── TASK.md
├── Makefile
└── framework/
    ├── student.c      <- 主要修改这里
    ├── student.h
    ├── verify.c       <- 自动验证，不改
    └── verify.h
```

当前 `student.c` 里只要求你完成两个函数：

- `student_pe_sinusoidal`
- `student_embedding_forward`

这两个函数的组合非常有代表性。第一个函数负责“位置编码的单元素公式”，第二个函数负责“把 token embedding 和 position embedding 装配成整层前向传播”。  

也就是说，这一章不是在让你实现完整 Embedding 模块的所有初始化和管理逻辑，而是在让你抓住它最核心的两步：

1. 位置编码这一行上的某个值到底怎么算；
2. 一个 token 序列怎样被逐位置写成 `[seq_len, hidden_dim]` 的输出矩阵。

## 3.3 为什么光有 token ID 还不够

上一章结束时，你已经有了 `"Hi"` 对应的一组整数 ID。看起来模型离“能处理输入”已经很近了，但其实还差一个关键层次。

token ID 的本质只是标签，不是表示。

例如，`76` 和 `109` 这两个数字并不天然带有“字符之间的相似性”或“语义接近性”。从模型角度看，如果你直接把这些整数当作数值去算，会产生非常奇怪的含义：难道 `109` 比 `76` 大，就说明 `'i'` 比 `'H'` 更重要吗？显然不是。

这就是 embedding 层存在的原因。它要做的不是保留这些整数的算术大小，而是为每个 token 分配一个可学习的连续向量表示。  

你可以把它想成一张很大的表：

```text
token_embedding[vocab_size][hidden_dim]
```

其中每一行对应一个 token，每一列对应这个 token 表示向量的某个维度。给定 token ID 后，embedding 做的事情非常朴素：直接去取那一行。

这就是为什么 embedding 通常被叫做“查表层”。它在最基本的意义上，确实就是查表。

## 3.4 但仅有 token embedding，模型仍然分不清顺序

到这里会出现一个新的问题。

如果你把一句话里的每个 token 都变成一个向量，那么模型确实有了连续输入表示；但同一个 token 在不同位置，查出来的 token embedding 是完全相同的。  
这就意味着，对模型来说，序列 `"AB"` 和 `"BA"` 在某种意义上会非常像：它看到了同样两种 token，只是换了位置，而单独的 token embedding 并不会告诉它“这个 token 现在在第 0 位还是第 1 位”。

这件事对 RNN 来说不那么致命，因为 RNN 的递推结构天然带有顺序。  
但对 Transformer 来说，它后面依赖的是 attention，而 attention 本身并不会自动记住“先后次序”。如果没有额外位置信息，模型看到的会更像是一袋 token，而不是一个有顺序的序列。

所以 position embedding 不是为了“锦上添花”，而是为了把序列顺序重新带回模型里。

## 3.5 sinusoidal position embedding 到底在做什么

本章默认关注的是 sinusoidal 位置编码，也就是一组固定的、由公式直接生成的位置向量。

最常见的写法是：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d))
```

这里：

- `pos` 表示当前位置；
- `i` 表示第几对维度；
- `d` 表示整个隐藏维度，也就是 `hidden_dim`。

这个公式第一次看时很像“数学人为构造”，但你可以先抓住它最直观的一层意义：

- 不同位置会得到不同向量；
- 不同维度变化的快慢不同；
- 因此模型能同时从某些维度里感知短距离位移，也从另一些维度里感知长距离位移。

你现在还不需要深入相对位置的推导，也不需要比较 RoPE、ALiBi 等变种。对本章来说，更重要的是：你能先把这个公式作为一个可验证、可写进代码、可实际打印出来的结构去理解。

## 3.6 为什么 `pos = 0` 是最好的第一个检查点

课程里最应该优先利用的，不是复杂公式本身，而是它们最简单的边界情况。

在 sinusoidal PE 里，`pos = 0` 恰好就是这样一个边界点。因为无论分母是什么，只要分子是 0，就有：

- `sin(0) = 0`
- `cos(0) = 1`

因此对任意维度，`pos = 0` 这一整行的位置编码都会呈现出非常规则的形状：

```text
[0, 1, 0, 1, 0, 1, ...]
```

这非常适合作为第一层验证。因为你不需要先理解整张位置编码矩阵，只要先确认第 0 个位置的偶数维是不是 0、奇数维是不是 1，就能知道自己的公式主干有没有跑偏。

这也是为什么 `Lab03` 的第一个测试就盯住这个现象。教学上最稳的方式，就是先从最容易手算、最不容易被误解的边界点入手。

## 3.7 `embedding_forward` 本质上只是两件事叠加

很多初学者第一次看到 embedding forward，会把它想得很复杂。其实把结构拆开后，它非常直接。

对序列里第 `pos` 个 token，embedding forward 做的事情就是：

1. 用 `token_ids[pos]` 去 token embedding 表里查一行；
2. 再取当前位置 `pos` 的 position embedding 那一行；
3. 把这两行逐元素相加，写到输出的第 `pos` 行。

换句话说：

```text
output[pos, d] = token_embedding[token_id, d] + position_embedding[pos, d]
```

或者在本 lab 的 student 实现里：

```text
output[pos, d] = token_embedding[token_id, d] + student_pe_sinusoidal(pos, d, hidden_dim)
```

这类前向传播最值得学的，不是“会不会写循环”，而是看懂结构：  

- token embedding 负责“这个 token 是谁”；
- position embedding 负责“它现在在第几个位置”；
- forward 把这两种信息压进同一个向量。

这个模式以后会反复出现。后面的很多层，都是把“不同来源的信息”加到一起，或者按某种结构组合起来。

## 3.8 本章实践步骤

### task 3.1：先看 `student.c` 和 `verify.c`

进入：

```bash
cd labs/lab03-step2
```

先读三份文件：

- `framework/student.h`
- `framework/student.c`
- `framework/verify.c`

你会发现这一章的测试在问四个非常明确的问题：

1. `pos = 0` 时，位置编码是否呈现 `0/1/0/1` 的交替；
2. `PE(1, 0)` 和 `PE(1, 1)` 是否接近 `sin(1)` 和 `cos(1)`；
3. 相同 token 放在不同位置时，输出向量是否真的不同；
4. 不同 token 放在相同位置时，输出向量是否也不同。

这四个问题很有代表性，因为它们分别覆盖了：

- 公式对不对；
- 数值算得像不像；
- 位置信息有没有真的加进去；
- token 信息有没有真的保留下来。

### task 3.2：实现 `student_pe_sinusoidal`

先做最小单位，也就是“位置编码的一个值”。

推荐先把逻辑拆成三步：

1. 计算当前维度对应的 `i = dim / 2`；
2. 计算分母 `10000^(2i / hidden_dim)`；
3. 决定当前维度是用 `sin` 还是 `cos`。

这里最值得留心的是：你不是在“背公式”，而是在把一个可验证的数值结构落到代码里。写完以后，第一时间就该拿 `pos = 0` 去检查，而不是先赌大样本行为。

### task 3.3：实现 `student_embedding_forward`

这个函数比上一章的 roundtrip 更像一个真正的“层前向传播”。

你要处理的核心事情有：

1. NULL 和长度边界；
2. 越界 token 用 UNK 兜底；
3. 外层按 `pos` 遍历；
4. 内层按 `d` 遍历；
5. 把 token embedding 和 position encoding 写进输出。

这里建议你刻意体会一个工程细节：  
本章虽然在讲“embedding”，但 student 实现并没有让你直接调用完整 `embedding_forward`。课程故意把最核心的结构拆给你自己写，就是为了让你亲手经历一次“这一层前向传播其实只是怎么组织几块数据”的过程。

### task 3.4：运行当前基线并理解它

在你还没完成这两个函数时，执行：

```bash
make clean && make test
```

当前这个 lab 的真实基线是：

- `TEST 1` FAIL
- `TEST 2` FAIL
- `TEST 3` FAIL
- `TEST 4` FAIL

这不是坏消息，反而说明 lab 现在正处在正确的“待学员完成”状态。  
因为当前 `student_pe_sinusoidal` 永远返回 `0.0f`，`student_embedding_forward` 也还什么都没写，所以四个测试都必然失败。

你应该把这次输出当作 Chapter 3 的 baseline，而不是当作环境异常。

### task 3.5：完成后重新验证

当你完成两个函数后，再执行：

```bash
make clean && make test
```

理想结果应当是：

```text
[TEST 1] ... [PASS]
[TEST 2] ... [PASS]
[TEST 3] ... [PASS]
[TEST 4] ... [PASS]
All tests passed!
```

如果 `TEST 1` 和 `TEST 2` 过了，但 `TEST 3`、`TEST 4` 没过，通常说明你的单元素 PE 算对了，但 forward 里没有正确把 token 信息和位置信息组合起来。

## 3.9 常见错误与排查顺序

| 现象 | 更可能的问题 | 优先检查 |
| --- | --- | --- |
| `TEST 1` 失败 | `pos = 0` 的边界没对齐 | `student_pe_sinusoidal` |
| `TEST 2` 失败 | `sin/cos` 公式或分母实现有误 | `student_pe_sinusoidal` |
| `TEST 3` 失败 | 没把位置信息真正加进去 | `student_embedding_forward` |
| `TEST 4` 失败 | token embedding 读取或写回逻辑有误 | `student_embedding_forward` |
| 全部仍为 0 | forward 根本没有写 output | `student_embedding_forward` 主循环 |

这一章最重要的排查思路是：先把“单元素公式”查清，再查“整层装配”。不要一上来就同时怀疑所有地方。

## 3.10 思考题

1. 如果彻底关掉 position embedding，模型为什么会更难区分 `"AB"` 和 `"BA"`？
2. 为什么 `pos = 0` 这么适合作为 sinusoidal PE 的第一个验证点？它揭示的是哪一类工程思路？
3. `student_embedding_forward` 里为什么要对越界 token ID 回退到 UNK，而不是直接崩溃？
4. 这一章的“查表 + 相加”，和上一章的“编码 + 解码”相比，虽然形式不同，但都体现了哪种“从协议层过渡到计算层”的结构？

## 3.11 本章小结

这一章的关键，不只是你第一次写出了 embedding forward，而是你第一次看见模型内部表示是如何生成的。

tokenizer 负责建立离散编号系统；embedding 则把这个离散系统投影到连续向量空间里。再加上 position encoding，模型终于拥有了一种既知道“这个 token 是谁”、又知道“它现在在哪”的输入表示。

从后面的角度看，这正是 attention 的起点。因为 attention 不会直接处理字符，也不会直接处理 token ID，它处理的正是这一章得到的 `[seq_len, hidden_dim]` 向量序列。

继续阅读：[Chapter 4](/course/chapters/ch04-step3-attention/)  
对应实践：[Lab04](/course/practice/labs/lab04-step3/)
