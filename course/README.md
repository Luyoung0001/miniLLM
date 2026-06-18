---
title: miniLLM 课程主页
---

# miniLLM 课程：从零写一个能聊天的迷你语言模型

这门课的目标很直接：带你从零写出一个纯 C 语言的小型语言模型。你不会只停在“能编译几个算子”这一步，而是会一路把张量、Tokenizer、Embedding、Attention、Transformer Block、完整 GPT、训练、生成、对话、HTTP 接口、KV Cache 和 BPE 串起来，最后做出一个真正能聊天的程序。

如果你以前看过很多大模型材料，却总觉得它们停在概念图和公式上，没有真正落到代码里，那么这门课就是为这个问题准备的。这里不会先把你扔进一个庞大框架，也不会让你先背一堆抽象定义。课程的节奏很固定：先讲清楚这章要解决的具体问题，再进入对应 lab，把这一章真正该写的代码补出来，然后用验证程序确认结果。

## 你会学到什么

学完整条主线后，你应当能独立回答下面这些问题：

- 一个张量在 C 语言里究竟是什么，它为什么离不开 `shape` 和 `stride`。
- 为什么字符级 Tokenizer 足够作为第一版模型入口，它的局限又在哪里。
- Embedding 和位置编码为什么能把“离散 token”变成可计算的向量。
- Attention 为什么需要 `Q`、`K`、`V`，为什么还要做缩放和 mask。
- Transformer Block 为什么不是“attention 写完就结束了”，而还需要残差、LayerNorm 和 FFN。
- 一个 GPT 模型从结构上怎样由前面这些部件拼起来。
- 为什么训练不是“再加几个循环”这么简单，而要同时关心 loss、梯度和优化器。
- 文本生成、多轮对话、HTTP 服务、KV Cache、BPE 分词各自解决的到底是哪一层问题。

更重要的是，你不会只在纸面上知道这些东西，而是会在 lab 里亲手把它们逐章补出来。

## 怎么开始

第一次进入课程时，不要急着跳到后面的章节。按这个顺序走：

1. 先读 [Chapter 0](00-orientation.md)，把环境和第一轮 smoke test 跑通。
2. 再读 [Chapter 1](chapters/ch01-step0-tensor.md)，进入正式课程。
3. 读完一章，就去对应 lab 完成这一章的代码。

如果你只想记一句最短路径，那就是：

```text
Chapter 0 -> Chapter 1 -> Lab01 -> Chapter 2 -> Lab02 -> ...
```

## 主线章节

| 阶段 | Chapter | 对应 Lab | 你会解决的问题 |
| --- | --- | --- | --- |
| 起步 | [Chapter 0](00-orientation.md) | 准备阶段 | 跑通环境，进入第一章前的真实工作状态 |
| 基础 | [Chapter 1](chapters/ch01-step0-tensor.md) | `lab01-step0` | 张量、stride、softmax |
| 基础 | [Chapter 2](chapters/ch02-step1-tokenizer.md) | `lab02-step1` | 字符级 Tokenizer |
| 基础 | [Chapter 3](chapters/ch03-step2-embedding.md) | `lab03-step2` | Embedding 与位置编码 |
| 模型 | [Chapter 4](chapters/ch04-step3-attention.md) | `lab04-step3` | 多头自注意力 |
| 模型 | [Chapter 5](chapters/ch05-step4-transformer.md) | `lab05-step4` | Transformer Block |
| 模型 | [Chapter 6](chapters/ch06-step5-gpt.md) | `lab06-step5` | 完整 GPT 模型 |
| 训练 | [Chapter 7](chapters/ch07-step6-training.md) | `lab07-step6` | 损失、反向传播、优化器 |
| 应用 | [Chapter 8](chapters/ch08-step7-generation.md) | `lab08-step7` | 文本生成 |
| 应用 | [Chapter 9](chapters/ch09-step8-chat.md) | `lab09-step8` | 多轮对话 |
| 工程 | [Chapter 10](chapters/ch10-step9-http.md) | `lab10-step9` | HTTP 服务 |
| 优化 | [Chapter 11](chapters/ch11-step10-kv-cache.md) | `lab11-step10` | KV Cache |
| 工程 | [Chapter 12](chapters/ch12-step11-bpe.md) | `lab12-step11` | BPE 分词 |
| 综合 | [Chapter 13](chapters/ch13-end-to-end.md) | `lab13-end-to-end` | 把 BPE 对话链路真正跑起来 |

## 附录

附录不是主线的一部分，只有在你遇到具体问题时再回来看：

- [A1：工具与前置知识](appendices/A1-prerequisites.md)
- [A2：常见陷阱与调试](appendices/A2-pitfalls.md)
- [A3：参考资料与延伸阅读](appendices/A3-references.md)

## 现在就开始

如果你已经准备打开第一章，那么下一步就是：

1. 进入 [Chapter 0](00-orientation.md)
2. 跑通里面的 smoke test
3. 继续进入 [Chapter 1](chapters/ch01-step0-tensor.md)
