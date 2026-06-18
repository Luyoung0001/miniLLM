---
title: 'Chapter 2 — step1 字符级 Tokenizer'
layout: 'course_page'
permalink: '/course/chapters/ch02-step1-tokenizer/'
lab: 'Lab02'
step: 'step1'
hours: '2h'
deliverable: 'Lab02 全部测试通过'
prev_page:
  title: 'Chapter 1 — step0 张量与数学运算'
  url: '/course/chapters/ch01-step0-tensor/'
next_page:
  title: 'Chapter 3 — step2 Embedding 与位置编码'
  url: '/course/chapters/ch03-step2-embedding/'
source_path: 'course/chapters/ch02-step1-tokenizer.md'
---

> **对应实践**：[`course/practice/labs/lab02-step1/`](/course/practice/labs/lab02-step1/)  
> **主要修改文件**：`course/practice/labs/lab02-step1/framework/student.c`  
> **验证命令**：`make clean && make test`

在 Chapter 1 里，你第一次建立了一个很底层但非常关键的认识：神经网络代码最终都要落实成“怎么从内存里把数取出来，再按规则写回去”。那一章解决的是“数字如何在张量里存在”的问题。

但模型真正要处理的输入并不是矩阵，也不是浮点向量，而是人写出来的文本。文本和张量之间隔着一道很硬的边界：计算机当然可以把 `'H'`、`'i'`、空格这些字符存在内存里，但模型并不会直接对“字符”这个概念做矩阵乘法。模型只接受数字，尤其只接受离散 ID 序列，再由 embedding 把这些 ID 变成浮点向量。

这就是 tokenizer 出场的原因。

所以这一章要回答的问题是：为什么 `"Hello"` 这样一个字符串，必须先变成 `[76, 105, 112, ...]` 这样的整数序列，后面的 embedding、attention、loss 才有工作对象。

## 2.1 本章真正要建立什么直觉

如果只从练习代码看，这一章似乎很简单：写几个字符和 token ID 之间的转换函数，再做一次 encode/decode 的 roundtrip。  
但从课程推进上看，这一章非常关键，因为它把“人类文本”第一次送入了“模型可处理的数据管道”。

这章结束后，你应当具备下面几个判断：

1. 知道 miniLLM 当前字符级 tokenizer 的词汇表为什么是 `4 + 256 = 260`。
2. 知道 `<PAD>`、`<UNK>`、`<BOS>`、`<EOS>` 为什么必须预留在最前面。
3. 明白字符级 tokenizer 不是在理解语言，而是在建立一种稳定、可逆、可喂给模型的离散编码方式。
4. 明白“编一次再解一次能拿回原串”为什么是 tokenizer 最基础也最重要的正确性检查。
5. 能在本章实验代码中完成 `student_encode_char`、`student_decode_id` 和 `student_roundtrip`。

和上一章一样，这里前四点是理解，第五点是实践。没有前四点，后面的练习会沦为背规则；没有第五点，这一章又会停留在纸面上。

## 2.2 先看 practice target：这一章改哪里

这一章的工作区同样不在 `step1/src/tokenizer.c` 里，而在本章 lab 目录中：

```text
course/practice/labs/lab02-step1/
├── TASK.md
├── Makefile
└── framework/
    ├── student.c      <- 主要修改这里
    ├── student.h
    ├── verify.c       <- 自动验证，不改
    └── verify.h
```

这一章真正需要你改的文件依然只有一个：`framework/student.c`。这说明本章仍然在坚持同一个原则：让你在一个边界明确、反馈直接的最小工作区里完成任务。

当前这个 lab 只要求你实现 3 个函数：

- `student_encode_char`
- `student_decode_id`
- `student_roundtrip`

看起来都不长，但它们恰好构成了一条完整数据链：

```text
字符 -> token ID -> 字符
```

这条链一旦稳定，下一章 embedding 才有可输入的整数序列。

## 2.3 为什么模型不能直接吃字符

这个问题对初学者很值得认真解释，因为它决定你会不会把 tokenizer 当成一个“多余前处理”。

模型之所以不能直接吃字符，不是因为字符“不够高级”，而是因为模型所有核心运算都是数值运算。一个 decoder-only 模型要做的是：

- 查表；
- 线性变换；
- attention；
- 归一化；
- 采样。

这些操作都要求输入先变成一个稳定的离散编号系统。你必须先告诉模型：“`H` 是哪个 ID，空格是哪个 ID，句子结束又是哪个 ID。”如果没有这层编号，后面 embedding 就不知道该查哪一行，loss 也不知道目标 token 对应哪一维。

从这个角度说，tokenizer 的工作不是“理解语言意义”，而是“把文本搬运进模型计算图”。

## 2.4 当前 miniLLM 的字符级词表为什么是 260

miniLLM 当前这一阶段还没有上 BPE，所以 tokenizer 采用的是最直接的字符级设计。

它的基本想法很简单：

- ASCII 普通字符一共有 256 个可能值（0 到 255）；
- 但在正式字符之前，还要预留几个特殊 token；
- 因此最终词表大小 = `4 + 256 = 260`。

这四个特殊 token 分别是：

- `TOKEN_PAD = 0`
- `TOKEN_UNK = 1`
- `TOKEN_BOS = 2`
- `TOKEN_EOS = 3`

然后从 `4` 开始，才轮到普通字符：

- ASCII 0 -> token 4
- ASCII 1 -> token 5
- ...
- ASCII 72 (`'H'`) -> token 76
- ASCII 105 (`'i'`) -> token 109

这里最关键的不是背这些数字，而是理解这个设计背后的秩序：

> 先把协议性、控制性的 token 预留出来，再把普通字符整体往后平移一个固定偏移量。

一旦你理解了“固定偏移量”这个想法，这一章的编码和解码逻辑其实就很直接了。

## 2.5 特殊 token 为什么必须单独存在

这一步很容易被初学者当成“为了好看加的常量”，其实不是。

特殊 token 的存在，是为了让模型和训练流程能表达普通字符之外的结构信号。

### `<PAD>`

它负责填充。以后当不同样本长度不一致、需要凑成固定长度 batch 时，就需要这个 token 占位。

### `<UNK>`

它负责兜底。遇到当前词表无法表示的内容时，至少要有一个合法 ID 可以退回，而不是直接崩掉。

### `<BOS>`

它告诉模型“一个序列从这里开始”。对生成模型来说，起点并不是无意义的，很多训练和推理流程都需要明确知道序列边界。

### `<EOS>`

它告诉模型“一个序列到这里结束”。没有它，模型在训练阶段就很难学习“何时停”，在推理阶段也很难自然结束输出。

即便这一章的 roundtrip 暂时可以不引入 BOS/EOS，你也应该知道它们为什么在词表里提前占了位置。因为这说明 tokenizer 并不是单纯的字符映射表，它还是整个序列协议的一部分。

## 2.6 编码和解码到底是在做什么

这一章最容易让人误以为“逻辑太简单，不值得讲”。但越简单的规则，越适合讲清楚它的本质。

### 编码

编码做的事是：

```text
char -> int
```

对普通字符来说，规则就是：

```text
id = (unsigned char)c + NUM_SPECIAL_TOKENS
```

这条公式有两个细节值得注意。

第一，为什么要转成 `(unsigned char)`？因为 `char` 在 C 里是否带符号，和编译器实现有关。如果你不先转成无符号范围，某些高位字符就可能被解释成负数。

第二，为什么要加 `NUM_SPECIAL_TOKENS`？因为前 0 到 3 已经留给特殊 token 了。普通字符不能再占这些位置。

### 解码

解码做的事是反过来：

```text
int -> char
```

对普通字符来说，就是：

```text
char = id - NUM_SPECIAL_TOKENS
```

但这里不能简单地对所有 ID 都这么做。因为前 0 到 3 根本不是普通字符；超出词表上界的 ID 也不能乱转。所以这一章的 `student_decode_id` 实际上是在做两件事：

1. 先过滤非法或特殊输入；
2. 再对合法普通字符做逆映射。

这种“先判边界，再做主逻辑”的写法，以后几乎每章都会出现。

## 2.7 为什么 roundtrip 是 tokenizer 的最低正确性标准

这一章第三个函数 `student_roundtrip` 看上去只是个小工具，但它实际上在验证 tokenizer 最重要的性质之一：**可逆性**。

如果一套 tokenizer 连最基本的 `decode(encode(text)) == text` 都做不到，那后面的 embedding、训练、生成就没有可靠输入输出边界。你甚至无法确认模型看到的到底是不是你以为的那串文本。

当然，真实世界里的 tokenizer 不一定对所有情况都完全严格可逆，尤其涉及规范化、空白折叠或更复杂分词时会有例外。但对当前这个字符级最小实现来说，roundtrip 就是最应该被牢牢守住的基线。

这也是为什么 `Lab02` 把第三个 learner function 设计成 `student_roundtrip`：它迫使你把前两个单点规则连成一条真正的数据路径，而不是停留在“单步看起来都对”。

## 2.8 这一章和下一章的关系是什么

如果说当前章解决的是“文本怎么变成离散编号”，那下一章要解决的就是“离散编号怎么变成浮点向量”。

这两个步骤不能调换。

embedding 从来不是直接作用在字符上，而是作用在 token ID 上。你可以把它理解成一张矩阵，每个 token ID 对应矩阵里的一行。  
那么自然地，只有先有 token ID，embedding 才知道该取哪一行。

也就是说，这一章不是一个独立的小插曲，而是把输入管道铺到 embedding 门口。

## 2.9 本章实践步骤

### task 2.1：先看一遍 `student.c` 和 `verify.c`

进入 practice 目录：

```bash
cd course/practice/labs/lab02-step1
```

先读：

- `framework/student.c`
- `framework/student.h`
- `framework/verify.c`

你会发现这一章的验证器很直白，它主要检查四件事：

1. 词表大小是不是 260；
2. `'H'` 和 `'i'` 编码后是不是 76 和 109；
3. 76 和 109 解码后是不是 `'H'` 和 `'i'`；
4. `"Hello"` 做 roundtrip 后能不能完整回来。

这种验证器的价值在于，它把“这一章真正要会什么”写得非常明确。

### task 2.2：实现 `student_encode_char`

这个函数的核心逻辑非常短，但不要因为短就草率。

你需要处理两类情况：

1. `tok == NULL` 或输入字符异常时，返回 `TOKEN_UNK`；
2. 普通字符时，返回 `(unsigned char)c + NUM_SPECIAL_TOKENS`。

这里建议你刻意留意“异常路径”和“主路径”是如何分开的。后面做更复杂模块时，这种结构会一直出现。

### task 2.3：实现 `student_decode_id`

这个函数的关键不是减去偏移本身，而是知道哪些 ID 根本不该被当成普通字符解码。

优先检查：

- `tok` 是否为空；
- `id < NUM_SPECIAL_TOKENS` 是否成立；
- `id >= tok->vocab_size` 是否成立。

只有这些边界都过了，才去做：

```c
(char)(id - NUM_SPECIAL_TOKENS)
```

如果你没有先做这些边界处理，很多输入在“数学上看似有值”，但在协议上是无意义的。

### task 2.4：实现 `student_roundtrip`

这一章最完整的实践就在这里。你需要把前两个局部规则串起来。

推荐的思路是：

1. 先对 `text`、`out`、`out_size` 做边界处理；
2. 调 `tokenizer_encode(tok, text, &len, 0, 0)` 得到 ID 数组；
3. 分配一个临时字符缓冲；
4. 逐个 ID 调 `student_decode_id` 写回字符；
5. 末尾补 `'\0'`；
6. 再复制到 `out`；
7. 释放中间内存。

这一步有一个很典型的工程点：即使逻辑很短，也不能忘记内存释放。课程越往后走，这类“不是算法主体、但会决定程序是否健康”的细节会越来越重要。

### task 2.5：运行验证

完成后执行：

```bash
make clean && make test
```

当前理想结果应当是：

```text
[TEST 1] ... [PASS]
[TEST 2] ... [PASS]
[TEST 3] ... [PASS]
[TEST 4] ... [PASS]
All tests passed!
```

如果还没有全部通过，不要在多个函数里同时乱改。先对照哪一条测试失败，再回到它对应的那一小段逻辑。

## 2.10 常见错误与排查顺序

| 现象 | 更可能的问题 | 优先检查 |
| --- | --- | --- |
| `TEST 1` 失败 | 词表大小或框架状态异常 | `tokenizer_vocab_size` 与框架状态 |
| `TEST 2` 失败 | 编码时偏移量没加对 | `student_encode_char` |
| `TEST 3` 失败 | 特殊 token / 上界判断有问题 | `student_decode_id` |
| `TEST 4` 失败 | roundtrip 路径没有完整串起来 | `student_roundtrip` |
| 编译不过 | 当前 lab 框架或你本地修改有问题 | 先看 `make` 第一处报错 |

这里有一个很值得养成的习惯：先分清“编译错误”和“验证失败”。

- 编译错误说明程序根本还没进入语义层；
- 验证失败说明程序能跑，但行为和预期不一致。

这两种问题的处理方式完全不同。不要把它们混在一起看。

## 2.11 思考题

1. 为什么字符级 tokenizer 在教学上很合适，但在真实大模型里通常不够高效？
2. 如果没有 `<EOS>`，模型在生成时还能靠什么机制停下来？这种停法有什么局限？
3. `student_decode_id` 遇到特殊 token 返回 `'\0'`。这只是一个最小课程选择。如果以后你想保留特殊 token 的文本形态，更合理的接口应该长什么样？
4. 这一章里“固定偏移量 + 可逆映射”的思想，和上一章“stride + 可逆定位”有什么共通点？

最后这个问题尤其重要。因为课程真正想训练的，不是你背住两个局部技巧，而是看到它们背后同一类设计结构。

## 2.12 本章小结

这一章你第一次把“人类文本”接进了模型管道。

你看到 tokenizer 并不神秘，它首先是在建立一个稳定的离散编号系统；你也看到特殊 token 并不是装饰，而是整个序列协议的一部分。更重要的是，你开始接触一类非常常见的工程模式：一条数据路径不仅要能正向走通，还要能通过 roundtrip 或其它方式证明它在边界上足够稳定。

下一章，离散 ID 将不再停留在整数层面，而会进入浮点表示世界。也就是：embedding。

继续阅读：[Chapter 3](/course/chapters/ch03-step2-embedding/)  
对应实践：[Lab03](/course/practice/labs/lab03-step2/)
