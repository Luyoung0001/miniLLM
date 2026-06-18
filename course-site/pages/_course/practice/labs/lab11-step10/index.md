---
title: 'Lab11 - KV Cache'
layout: 'course_page'
permalink: '/course/practice/labs/lab11-step10/'
lab: 'Lab11'
step: 'step10'
hours: '3h'
deliverable: '完成 KV Cache alloc/append 相关 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab11-step10/TASK.md'
---


> **实验编号** Lab11 **预计耗时** 3h **对应 step** [step10](https://github.com/Luyoung0001/miniLLM/tree/main/step10) **本节产出** 补完 KV Cache 分配与追加接口，并让验证输出相对当前实验基线继续转绿

## 实验目的

上一阶段（Lab10）你把模型塞进 HTTP 服务端，让它能"边喝边吐"。这一章你**不**改 HTTP，**不**改 attention，**不**改 generate——你做的事是**自己造一个 KVCache**：一个用来存放每一层"已算过"的 K 和 V 的大数组，并暴露 `alloc / append` 两个最基本的接口；剩下的"自回归推理加速"细节由 `framework/kv_cache.c` 帮你跑通。

做完本实验，你会：

1. 能口述 KV Cache 解决"重复计算"的根因：K、V 的"右下文不变"性质。
2. 能写 `student_kv_cache_alloc`：按 (num_layers, max_seq_len, hidden_dim) 申请一片 float 内存。
3. 能写 `student_kv_cache_append`：把新算出来的那一行的 K、V 写到 cache 的第 `pos` 个位置。
4. 能心算 cache 容量公式：`2 × num_layers × max_seq_len × hidden_dim × 4` 字节。
5. 能在不翻文档的前提下，区分 **prefill**（一次性算完整 prompt）和 **decode**（每步追加一个位置）。

## 为什么服务跑起来之后还要回头做缓存

`Lab10` 让模型第一次变成了可以被外部调用的服务，但这还不代表它“用起来合理”。如果每生成一个新 token，都把前面整段 prompt 从头再算一遍，那么随着上下文越来越长，推理时间会越来越难看。也就是说，服务化解决了“能不能被调用”，但没有解决“能不能高效地连续生成”。

KV Cache 这一章，就是在给推理链路补上一块真正的性能基础设施。它没有改变模型输出的数学结果，却改变了你为了拿到同一个结果，需要重复做多少无谓计算。从教学上看，这也是课程第一次正式引入“优化不是锦上添花，而是执行路径本身的一部分”这个视角。

## 实验环境

| 项 | 路径 / 命令 |
| --- | --- |
| 共享 KV Cache 头 | `../../framework/kv_cache.h` |
| 共享 Tensor 头 | `../../framework/tensor.h` |
| 学员改的文件 | `./framework/student.c` |
| 学员写的函数声明 | `./framework/student.h` |
| 自动验证 | `./framework/verify.c`（打印 `[PASS] / [FAIL]`） |
| 参考 step | `../../step10/` |
| 配套讲义 | `../../chapters/ch11-step10-kv-cache.md` |

```bash
cd labs/lab11-step10
make clean && make test
```

可用的关键 framework API（你**不**应该再写一遍）：

- `KVCache* kv_cache_create(int num_layers, int max_seq_len, int hidden_dim);`
- `void kv_cache_update_pos(KVCache*, int layer_idx, int pos, float* k_data, float* v_data);`
- `void kv_cache_set_len(KVCache*, int len);`
- `size_t kv_cache_memory_size(KVCache*);`

## 实验内容

这章的任务故意写得很窄，只让你补 `alloc / append / len` 三个接口，而不是让你重写完整 decode 流程。原因很明确：课程现在要你先把“缓存对象本身的边界”和“写入一行 K/V 的动作”吃透，至于如何在真实生成循环里调用它，framework 已经替你演示过了。

### 1.1 任务 1.5.1：在 `student.c` 里实现 `student_kv_cache_alloc`（15 行）

打开 `framework/student.c`，定位 `student_kv_cache_alloc`。它要做的事：给定 `(num_layers, max_seq_len, hidden_dim)`，返回一个非空 `KVCache*`。

提示：直接调 `framework/kv_cache_create` 即可——**不要**自己写 `malloc`。本任务的重点是让你在 verify 里**只靠**这个函数能造出一个 cache，再去验证 `kv_cache_memory_size` 与公式吻合。

这一题虽然代码量很小，但它第一次把“优化结构也要有清晰对象边界”这件事摆到了台面上。KV Cache 不是一块随手塞进全局变量的数组，而是一个有层数、长度、容量约束的正式对象。后面任何推理优化，如果没有这种对象化边界，最终都会越来越难维护。

### 1.2 任务 1.5.2：在 `student.c` 里实现 `student_kv_cache_append`（15 行）

定位 `student_kv_cache_append`。它要做的：把给定的 `k_row / v_row`（长度 = `hidden_dim` 的 `float*`）写到 `cache` 的第 `layer_idx` 层的第 `pos` 个位置。

提示：直接调 `framework/kv_cache_update_pos` 即可。本任务的重点是让你看到"append 一个位置 → 再 append 下一个 → 读回 → 验证内容一致"这条路径。**不要**自己 `memcpy` 写底层 `data[]`，因为 `kv_cache_update_pos` 已经帮你处理了 stride。

这一题真正想让你意识到的是：decode 阶段和 prefill 阶段最大的不同，并不是“调用了不同函数”，而是“现在每次只需要往缓存里追加一个新位置”。只要这个追加动作的语义被你彻底看懂，后面“为什么第二次 `model_forward` 输入长度可以是 1”这件事就会变得非常自然。

### 1.3 任务 1.5.3：在 `student.c` 里实现 `student_kv_cache_len`（5 行，可选）

定位 `student_kv_cache_len`。它要做的：把 `cache->current_len` 写到 `*out_len` 里并返回 0。

提示：这一题是 bonus，主要训练你"把 framework 的字段"包装成 verify 能调的稳定接口。

### 1.4 任务 1.5.4：跑通 verify

```bash
make clean && make test
```

完成后目标输出：

```
[TEST 1] kv_cache_alloc returns non-NULL              ... [PASS]
[TEST 2] kv_cache memory size matches formula         ... [PASS]
[TEST 3] kv_cache_append writes back same data        ... [PASS]
[TEST 4] kv_cache_len reflects appended count         ... [PASS]
[ALL PASS]
```

## 现象

这一章第一次运行时通常是全红，因为 `student_kv_cache_alloc` 和 `student_kv_cache_append` 正好卡在 verify 的主路径上；只要这两个接口没接上，后面的容量检查和读回检查都不会通过。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 1] kv_cache_alloc returns non-NULL               ... [FAIL]
[TEST 2] kv_cache memory size matches formula          ... [FAIL]
[TEST 3] kv_cache_append writes back same data         ... [FAIL]
[TEST 4] kv_cache_len reflects appended count          ... [FAIL]
[HAS FAIL]
```

### 完成本章 TODO 后的目标输出

```
$ cd course/practice/labs/lab11-step10
$ make clean && make test
rm -rf build student
mkdir -p build
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tensor.c -o build/tensor.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/math_ops.c -o build/math_ops.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/embedding.c -o build/embedding.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/attention.c -o build/attention.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/layernorm.c -o build/layernorm.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/ffn.c -o build/ffn.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/transformer.c -o build/transformer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/model.c -o build/model.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/generate.c -o build/generate.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/kv_cache.c -o build/kv_cache.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o build/tensor.o build/math_ops.o build/embedding.o build/attention.o build/layernorm.o build/ffn.o build/transformer.o build/model.o build/generate.o build/kv_cache.o -lm
[TEST 1] kv_cache_alloc returns non-NULL              ... [PASS]
[TEST 2] kv_cache memory size matches formula         ... [PASS]
[TEST 3] kv_cache_append writes back same data        ... [PASS]
[TEST 4] kv_cache_len reflects appended count         ... [PASS]
[ALL PASS]
```

这一章没有必要把目标压到“3 个 PASS 就够”。更合理的目标就是 4 项都过，因为它们实际上都围绕同一个缓存接口族。

本章结束时，你最好已经把 KV Cache 看成“推理里的状态存储层”，而不是“某个为了加速临时加上的数组”。只有把它当作正式状态对象，后面 BPE 对话整合时你才不会把 cache 复用和上下文管理混为一谈。

## 思考题

1. **（必做）** 解释为什么 Q 不能被缓存，而 K 和 V 可以。想一下 `Q[t]` 这一行要不要"看到自己"以及它依赖什么输入。
2. **（必做）** 任务 1.5.2 里的 `student_kv_cache_append` 一次写一个位置。在真实 `step10` 的 decode 循环里，每生成一个新 token 都要调一次本函数；如果 `seq_len` 翻倍到 256，这个函数的"被调用次数"和"cache 占的内存"分别怎么变？
3. **（选做）** 如果本项目里 `num_layers` 翻倍（4 → 8）但 `max_seq_len` 减半（128 → 64），cache 内存是变大、变小还是不变？用 task 1.5.2 的公式直接算。
4. **（选做）** 真实的 7B 模型 `num_layers=32, hidden_dim=4096, seq=4096`，按本公式算 cache 占多少 GB？算完对比你本机的可用 RAM，思考生产环境里 KV Cache 是"必须"还是"奢侈"。

## 上节 / 下节

- **上节**：[Lab10 - HTTP 服务](/course/practice/labs/lab10-step9/)：把模型塞进 HTTP，验证端到端对话。
- **下节**：[Lab12 - BPE 分词](/course/practice/labs/lab12-step11/)：把 `vocab_size` 从 256 升到 601，每一步 decode 推进更多内容，配合本节的 KV Cache 进一步放大加速比。
