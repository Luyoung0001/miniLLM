---
title: Chapter 11 — step10 KV Cache
lab: Lab11
step: step10
hours: 3h
deliverable: Lab11 全部测试通过
---

# Chapter 11：step10 — KV Cache

> **对应实践**：[`course/practice/labs/lab11-step10/`](../practice/labs/lab11-step10/TASK.md)  
> **主要修改文件**：`course/practice/labs/lab11-step10/framework/student.c`  
> **验证命令**：`make clean && make test`

到上一章为止，模型已经不仅能生成文本，还已经可以被包装成一个对外服务。  
这时新的问题自然会出现：

> 如果同一个请求在生成过程中不断追加 token，模型是不是每一拍都在把整段历史重新算一遍？

答案是：如果没有缓存，那确实会这样。

这就是 KV cache 章节存在的原因。  
它不改变模型的数学输出目标，而是优化模型在自回归推理时的重复工作量。

需要注意的是，`Lab11` 的实践范围依然控制得很小。课程没有要求学员从头重写整套带 cache 的 attention，而是先把最小接口层留出来：

- `student_kv_cache_alloc`
- `student_kv_cache_append`
- `student_kv_cache_len`

也就是说，这章真正要学的第一步不是“我已经能写高性能推理”，而是：

> 我先得清楚 cache 在结构上到底是什么，它保存什么，它怎样被追加和读取。

## 11.1 为什么 KV cache 解决的是“重复计算”

在自回归生成里，第 `t` 次生成时，模型已经知道前面 `0..t-1` 个位置的历史。  
如果没有缓存，模型通常会把整段历史从头到尾再跑一次 attention。

但注意力里有一类量有一个非常关键的性质：

- 历史位置的 K 和 V 一旦算出来，在后续新 token 到来时并不会改变。

这意味着，真正每一步都必须重新算的，主要是新位置相关的那一小部分。  
而历史 K/V 完全可以缓存下来复用。

所以 KV cache 的核心价值并不是“变了一个更聪明的公式”，而是：

> 把本来会重复做的历史计算结果留下来，下次别再重算。

## 11.2 本章你要建立哪些判断

这一章结束后，你应当能够清楚说出：

1. 为什么 K 和 V 可以缓存，而 Q 不适合用同样方式处理。
2. KV cache 的容量为什么与 `num_layers`、`max_seq_len`、`hidden_dim` 成正比。
3. 为什么 append 一个新位置，本质上是在每层的 cache 里写入一行 K 和一行 V。
4. 为什么 `current_len` 是 cache 状态的一部分，而不是临时变量。
5. 为什么 KV cache 是典型的“用内存换时间”策略。

这些判断建立之后，后面你读完整 `step10` 实现时，才会知道哪些是主逻辑，哪些只是包装。

## 11.3 先看 practice target：这章改哪里

本章实践目录是：

```text
course/practice/labs/lab11-step10/
├── TASK.md
├── Makefile
└── framework/
    ├── student.c      <- 主要修改这里
    ├── student.h
    ├── verify.c       <- 自动验证，不改
    └── verify.h
```

本章需要你实现的函数有三个：

- `student_kv_cache_alloc`
- `student_kv_cache_append`
- `student_kv_cache_len`

课程这里故意没有要求你直接碰到底层 stride 或 attention 核心公式，而是先把 cache 生命周期最关键的三个接口做清楚：

1. 先能建出来；
2. 再能往里写；
3. 最后能知道自己写到了哪里。

## 11.4 为什么 cache 首先是一个数据布局问题

很多人第一次接触 KV cache 时，会直接想到“推理加速”。  
这当然没错，但如果只盯住速度，往往反而看不清它在代码里到底长什么样。

从数据结构角度看，KV cache 最核心的事情其实很朴素：

1. 对每一层，都留一块能按序列位置存 K 的区域；
2. 再留一块同样按序列位置存 V 的区域；
3. 记录当前已经写到了第几个位置。

这说明 KV cache 不是一个抽象“开关”，而是一套明确的内存布局。

## 11.5 为什么 append 这一题有教学价值

`student_kv_cache_append` 看起来可能只是包一层调用，代码并不长。  
但它的教学价值很高，因为它让学员亲手经历一个非常重要的推理流程：

1. 新 token 来了；
2. 当前层的新 K/V 算出来了；
3. 要把它们放进 cache 的哪一层、哪一个位置。

只要这件事在脑子里清楚了，后面去理解 prefill、decode、cache 命中这些概念时，就会容易很多。

## 11.6 为什么 `current_len` 不是可有可无

cache 如果只是有一大块内存，却不知道已经写到了哪里，那它就没有办法安全参与下一轮生成。  
所以 `current_len` 不是装饰字段，而是 cache 状态的一部分。

它回答的是：

> 现在这个 cache 里，前多少个位置已经是有效历史。

这也是为什么 `student_kv_cache_append` 不只是写数据，还要记得同步更新长度。

## 11.7 本章实践步骤

### task 11.1：先读 `student.c` 和 `verify.c`

进入：

```bash
cd course/practice/labs/lab11-step10
```

建议先读：

- `framework/student.h`
- `framework/student.c`
- `framework/verify.c`

当前验证器会检查：

1. cache 是否能被成功创建；
2. cache 占用字节数是否符合公式；
3. append 进去的 K/V 能否从对应位置正确读回；
4. `current_len` 是否和追加次数一致。

这说明本章虽然没有直接测完整推理速度，但它已经抓住了 KV cache 最重要的结构行为。

### task 11.2：实现 `student_kv_cache_alloc`

这一题最重要的不是代码量，而是理解：

1. cache 不是普通临时数组；
2. 它和模型配置直接相关；
3. 一旦创建失败，就不应该继续假装后面的状态存在。

### task 11.3：实现 `student_kv_cache_append`

这一题最值得建立的直觉是：  
append 一个位置，实际上就是在 cache 的“层 + 序列位置”二维坐标上写入一行。

当前课程为了降低负担，已经把底层 stride 和写入逻辑封装好了。  
你现在真正要抓住的是“接口语义”，也就是：

- 我往哪层写；
- 我往第几个位置写；
- 写完后长度该怎样更新。

### task 11.4：实现 `student_kv_cache_len`

这一题最短，但不要忽略它的意义。  
它其实是在把内部状态暴露成一个稳定接口，方便验证器和后续代码去观察 cache 当前走到哪里。

### task 11.5：运行当前真实基线

在还没补完三个函数前，先执行：

```bash
make clean && make test
```

当前 `Lab11` 的真实初始状态是：

- **0 通过，4 失败**。

这说明：

1. 框架和链接已经可以跑起来；
2. 但 cache 的最小生命周期接口还没有真正实现。

这是一种很好的教学起点，因为当前失败都集中在本章自己的实现范围里，没有混进过多外围噪声。

### task 11.6：完成后重新验证

当你补完三个函数后，再运行：

```bash
make clean && make test
```

如果实现正确，你应该看到：

- cache 创建成功；
- 容量公式匹配；
- append 后读回数据正确；
- `current_len` 正确推进。

到这里，KV cache 在课程里的第一层结构直觉就建立起来了。

## 11.8 常见错误与排查顺序

最常见的错误通常是：

1. 创建 cache 时参数没传对；
2. append 时没有做空指针保护；
3. 追加成功后忘了更新长度；
4. 写入了错误的层或错误的位置。

建议排查顺序是：

1. 先看 alloc 是否成功；
2. 再看 memory size 是否符合公式；
3. 再看 append 后的具体读回值；
4. 最后看 `current_len`。

## 11.9 思考题

1. 为什么 K 和 V 适合缓存，而 Q 不适合照搬同样模式？
2. 为什么说 KV cache 是典型的“空间换时间”？
3. 如果 `num_layers` 或 `max_seq_len` 翻倍，cache 占用会怎样变化？

## 11.10 本章小结

Chapter 11 把课程带进了推理优化这个层次。

它的重点不是让学员马上写出最快的实现，而是先把 cache 作为一个明确数据结构看清楚：

- 它存什么；
- 它怎样扩展历史；
- 它怎样让后续推理避免重复工作。

这一步打通后，下一章就顺理成章了。因为一旦推理路径更像真实系统，词表和分词方案的问题就会变得更加显眼，而 BPE 正是下一步要解决的输入表示升级。
