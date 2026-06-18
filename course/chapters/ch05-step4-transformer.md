---
title: Chapter 5 — step4 Transformer Block
lab: Lab05
step: step4
hours: 3h
deliverable: Lab05 全部测试通过
---

# Chapter 5：step4 — Transformer Block

> **对应实践**：[`course/practice/labs/lab05-step4/`](../practice/labs/lab05-step4/TASK.md)  
> **主要修改文件**：`course/practice/labs/lab05-step4/framework/student.c`  
> **验证命令**：`make clean && make test`

到上一章为止，你已经让每个位置具备了读取上下文的能力。  
这意味着模型终于不再只是“一串并排的向量”，而开始形成真正的上下文交互。

但 attention 还不是 Transformer block。

如果你只把 attention 一层一层硬堆起来，模型很快会遇到两个非常现实的问题：

1. 数值越来越不稳；
2. 表达能力还不够完整。

第一个问题需要 LayerNorm 和残差来稳住。  
第二个问题需要一个逐位置的前馈网络（FFN）来补足。

本章真正要做的，就是把前面已经学过的几种结构第一次装进一个标准的、能被重复堆叠的 block 里。

## 5.1 这一章为什么是“组装”，但又不是简单拼接

从表面看，这一章很像组装题。

你已经见过：

- 张量和 softmax；
- embedding；
- attention；
- 一些基础数值结构。

那么把它们装进一个 block，似乎只是把几段 API 连起来而已。

但真正的难点恰恰在于：  
**这些模块不只是要能连起来，还要按正确的顺序连起来，并且在数值上保持稳定。**

Transformer block 的价值，不在于它发明了新的基本算子，而在于它找到了一个非常高效的组织方式，让：

- 每个位置既能读上下文；
- 又能保留原始信息；
- 还可以通过逐位置 MLP 做更强的非线性变换；
- 最后还能稳定堆很多层。

所以这一章真正要建立的，是“结构编排能力”。这会是你后面读完整 GPT 模型时最重要的前置直觉。

## 5.2 本章你要建立哪些判断

完成本章后，你应当能够：

1. 用自己的话口述 Pre-LN block 的主数据流：`h = h + Attn(LN(h))`，再 `h = h + FFN(LN(h))`。
2. 明白 LayerNorm 解决的是哪类数值问题，以及它为什么总是和残差一起出现。
3. 明白 FFN 不是“attention 的重复”，而是给每个位置增加更强的逐位置非线性变换能力。
4. 知道为什么 block 不是“attention 后直接结束”，而必须把这几种结构串在一起。
5. 在本章实验代码中完成：
   - `student_layernorm`
   - `student_residual_add`
   - `student_block_forward`

你会发现，这一章和上一章有一个很强的延续关系。上一章是把 attention 拆成三段；这一章则是把整个 block 看成一个更高层的组合结构。

## 5.3 先看 practice target：这章改哪里

本章的主工作区是：

```text
course/practice/labs/lab05-step4/
├── TASK.md
├── Makefile
└── framework/
    ├── student.c      <- 主要修改这里
    ├── student.h
    ├── verify.c       <- 自动验证，不改
    └── verify.h
```

当前 student 文件里要求你完成三个函数：

- `student_layernorm`
- `student_residual_add`
- `student_block_forward`

这三个函数覆盖的正是 block 最核心的三层组织逻辑：

1. 先把输入归一化；
2. 再把子层输出加回原输入；
3. 最后按 Pre-LN 的顺序把 attention 和 FFN 都串起来。

从教学上说，这种拆法很合理。因为如果一上来就让学员直接照着 `transformer_block_forward` 抄一遍，很容易只看见“很多函数调用”，却看不见其中哪一层是在解决什么问题。

## 5.4 LayerNorm 为什么会在这里出现

Attention 已经能让位置之间交互了，那为什么还要在它前后引入归一化？

原因在于，一旦网络开始堆叠，多层输出的分布会越来越容易漂移。  
如果每一层都在上一层已经发生漂移的数值上继续做线性变换、softmax、非线性激活，那么模型很快就会出现：

- 某些维度值越来越大；
- 某些维度梯度越来越不稳定；
- 激活范围越来越难控制。

LayerNorm 做的事情很直接：对一个位置的整行向量，先减均值，再除以标准差，把它拉回一个更稳定的尺度。

最基础的形式就是：

```text
y_i = (x_i - mean) / sqrt(var + eps)
```

本 lab 里的 `student_layernorm` 甚至还刻意简化了：  
它不让你先处理可学习的 `gamma` / `beta`，而是先把“归一化本身”搞明白。

这是一种非常好的课程切法。因为对当前阶段来说，真正要先学会的是：

> 如何把一个向量拉回均值约为 0、方差约为 1 的尺度。

而不是一开始就把所有参数化细节都压进来。

## 5.5 残差连接为什么不是装饰

残差连接有时会被初学者误解成“就是多写一个加号”。  
但这个加号在深模型里极其关键。

如果没有残差，当前层就必须完全依赖子层输出，原始输入一旦在中间被破坏或放大，很难有稳定路径保留下去。  

而有了：

```text
h = h + F(h)
```

模型就获得了一条非常重要的结构特性：

- 子层可以负责“增量修正”；
- 原始输入不会轻易丢失；
- 堆层数时，信息和梯度都更容易保持可传递性。

这就是为什么本章会单独把 `student_residual_add` 作为一个 learner function 留出来。  
它从代码量看很小，但从结构意义看非常大。因为它代表的是 Transformer 这类深层模型之所以能稳定工作的核心设计之一。

## 5.6 FFN 在 block 里到底补了什么

很多人第一次学 Transformer 时，会误以为 attention 已经“足够强”，FFN 只是一个附带模块。  
其实不是。

attention 主要擅长做的是：  
**在不同位置之间建立信息流动和加权聚合。**

但它并不擅长做每个位置自己的更强非线性变换。

FFN 正是在补这个空缺。  
它对每个位置单独地做两层 MLP：

```text
FFN(x) = W2 * GELU(W1 * x + b1) + b2
```

常见结构里，它会先把维度从 `hidden_dim` 扩到更大的 `ffn_dim`，再压回来。当前 miniLLM 的典型设置就是扩到 `4 * hidden_dim`。

这一步的意义在于：

- 给每个位置更高维的中间表示空间；
- 引入更强的逐位置非线性；
- 让模型不只是“会看别人”，也会“自己加工已经读到的信息”。

所以 Transformer block 的直觉不应该是“attention + 一些收尾逻辑”，而应该是：

> 先让位置之间交换信息，再让每个位置自己消化这些信息。

## 5.7 为什么是 Pre-LN，而不是别的顺序

这一章明确要求的是 Pre-LN 结构：

```text
h = h + Attn(LN(h))
h = h + FFN(LN(h))
```

这和另一种常见写法 Post-LN 的区别，不在于模块有没有变，而在于 LayerNorm 放在了子层之前还是之后。

当前课程阶段不需要你完整比较两者所有理论差异，但至少要记住一个非常实用的判断：

Pre-LN 通常更容易在深层训练中保持稳定。

所以这章不是在给你一个任意顺序的拼装练习，而是在让你熟悉一个后来在实际实现里非常常见的稳定结构。

## 5.8 本章实践步骤

### task 5.1：先读 `student.c` 和 `verify.c`

进入：

```bash
cd course/practice/labs/lab05-step4
```

建议先读：

- `framework/student.h`
- `framework/student.c`
- `framework/verify.c`

你会看到这一章的验证器分别检查：

1. `student_layernorm` 输出均值是否接近 0；
2. `student_layernorm` 输出方差是否接近 1；
3. `student_residual_add` 是否真的把 `a += b` 做对；
4. `student_block_forward` 是否保持 shape；
5. 输出是不是全零；
6. 输出里是否含 NaN 或 Inf。

这说明本章验证器既看结构，也看数值状态。它不是只问“函数调没调用”，而是在问“你装出来的 block 是否真的保留了最基本的数值健康性”。

### task 5.2：实现 `student_layernorm`

这一章最适合作为第一步的，仍然是最小功能单元。

你要做的是对一维向量：

1. 计算均值；
2. 计算方差；
3. 再按 `sqrt(var + eps)` 做归一化。

这里建议你刻意体会上一章和这一章之间的联系：

- 上一章 softmax 里你已经学过“数值稳定”；
- 这一章 LayerNorm 则是在学“数值尺度控制”。

两者虽然不一样，但都属于“模型正确工作不只靠公式，还靠数值状态被约束在合理范围内”。

### task 5.3：实现 `student_residual_add`

这是本章代码量最小、但结构意义最强的一步。

实践里它可能就是调用一次 `tensor_add_inplace(a, b)`。  
但你不应该因此低估它。

因为这里这个小函数代表的是整个 block 保留主路径信息的机制。  
以后你再看到更大模块里的残差，最好都能第一时间意识到：这不是语法上的“加一下”，而是网络结构上的信息保留和梯度通路设计。

### task 5.4：实现 `student_block_forward`

这是本章真正的主任务。

你需要把 LayerNorm、Attention、FFN 和两次残差串起来，顺序必须正确。  
一个简单但有用的思考方式是：不要把它当成“一个长函数”，而是当成两段平行结构的重复：

第一段：

```text
LN -> Attention -> residual add
```

第二段：

```text
LN -> FFN -> residual add
```

当你这样看，Transformer block 就不再像一团混合逻辑，而会显得很有节奏感。

### task 5.5：运行当前真实基线

在 student 实现还没补完之前，执行：

```bash
make clean && make test
```

当前这个 lab 的真实基线是：

- `student_layernorm` 的返回值和均值检查可能 PASS；
- 但方差接近 1 的检查 FAIL；
- `student_residual_add` 的返回值可能 PASS；
- 但实际数值加法行为 FAIL；
- `student_block_forward` 的 shape 检查和部分健康性检查可能 PASS；
- 但 output 非零等关键行为 FAIL。

这说明当前基线不是“整块都坏”，而是“框架把调用壳子和部分结构线已经搭好，但核心数学行为还没有真正成立”。

这种基线很适合 block 这种组合模块，因为它会提醒你：

- 框架级对象和缓存已经创建好了；
- 函数签名和调用路径也是通的；
- 但你还没有把 block 的真正行为建立起来。

### task 5.6：完成后重新验证

补完三个函数后再执行：

```bash
make clean && make test
```

理想结果应当是：

- `student_layernorm` 的均值和方差检查都 PASS；
- `student_residual_add` 的两次求和检查都 PASS；
- `student_block_forward` 的 shape、非零、非 NaN 等检查都 PASS。

到这时，课程前五章就形成了一个非常清晰的累积链：

- Chapter 1：你会处理底层张量和 softmax；
- Chapter 2：你把文本变成 token ID；
- Chapter 3：你把 token ID 变成向量；
- Chapter 4：你让位置之间彼此读取信息；
- Chapter 5：你把这些部件装成一个可堆叠的 Transformer block。

## 5.9 常见错误与排查顺序

| 现象 | 更可能的问题 | 优先检查 |
| --- | --- | --- |
| 均值接近 0，但方差不接近 1 | LayerNorm 方差或除法路径不对 | `student_layernorm` |
| 残差测试 sum 不对 | 没真正原地加，或 shape 路径错 | `student_residual_add` |
| block 输出全零 | 子层输出没写回，或残差没加进去 | `student_block_forward` |
| block 输出含 NaN | LayerNorm / FFN / Attention 组合顺序或数值路径有误 | `student_block_forward` |
| shape 对，但行为不对 | 调用了模块，但没有正确组织数据流 | `student_block_forward` |

这一章最重要的排查原则是：  
先确认每个基础积木自己单独成立，再去查 block 级组装。不要在 `student_block_forward` 里盲目猜所有问题。

## 5.10 思考题

1. 为什么 attention 之后还需要 FFN？如果没有 FFN，block 的表达能力会缺什么？
2. 为什么残差连接和 LayerNorm 往往一起出现？它们分别在稳定性上承担什么角色？
3. 本章的 Pre-LN 结构和上一章 attention 的三段式拆分相比，体现了哪种“先理解局部，再理解组合”的学习顺序？
4. 如果把 `student_block_forward` 看成一个“更大粒度的前向传播模板”，你觉得下一章完整 GPT 模型会在这个模板外面再包哪几层结构？

## 5.11 本章小结

这一章最关键的收获，不是“会写一个 block 函数”，而是第一次看清 Transformer block 为什么是一种稳定、可堆叠的模块单位。

attention 解决的是上下文读取；FFN 解决的是逐位置非线性加工；LayerNorm 和残差则保证这些子层组合后仍然足够稳定。  
当这几种结构按正确顺序组织在一起时，模型才真正拥有了反复堆叠的基本积木。

下一章，课程会把“单个 block”进一步提升成“完整 GPT 模型”：embedding 在前，若干 block 在中间，最后再接上输出头和保存加载逻辑。

继续阅读：[Chapter 6](ch06-step5-gpt.md)  
对应实践：[Lab06](../practice/labs/lab06-step5/TASK.md)
