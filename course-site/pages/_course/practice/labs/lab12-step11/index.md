---
title: 'Lab12 - BPE 分词'
layout: 'course_page'
permalink: '/course/practice/labs/lab12-step11/'
lab: 'Lab12'
step: 'step11'
hours: '3h'
deliverable: '完成 BPE pair 统计与 merge 两个 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab12-step11/TASK.md'
---


> **实验编号** Lab12 **预计耗时** 3h **对应 step** [step11](https://github.com/Luyoung0001/miniLLM/tree/main/step11) **本节产出** 补完 BPE pair 统计与 merge 两个接口，并让验证输出相对当前实验基线继续转绿

## 实验目的

完成本 lab 后，你将能够：

1. 用一句话解释 BPE 训练的两步核心循环——**先统计相邻 token 对的频次、再合并最高频的那一对**。
2. 自己手写 BPE 训练循环里"统计 + 合并"这两个最关键的操作（`student_bpe_count_pairs` 和 `student_bpe_apply_merge`）。
3. 对比字符级和 BPE 的 token 数，算出压缩比（一般 3~7 倍）。
4. 在 `bpe_vocab.txt` 里读出"哪一行是被焊了 N 次的高频短语"。

这一章是"使用 BPE"和"自己实现 BPE 关键算子"的折中：你**不**重写整棵 tokenizer 树，但**要**亲自动手写两个核心算子（找最高频 pair、在序列里做替换），再用 framework 的 `bpe_train` / `bpe_encode` 验证自己的实现和它一致。

## 为什么到最后还要从字符级切到 BPE

走到这时，你已经可以用字符级 tokenizer 完成训练、生成、对话、甚至服务化了。那为什么课程还要在后半段专门再做一章 BPE？原因很简单：字符级虽然概念最直接，但它对真实文本来说太低效。一个常见英文单词如果按字符切，可能要占掉很多 token；而 BPE 的目的，就是把经常一起出现的子串焊成一个更大的子词单位。

从项目角度看，这一章不是在推翻前面的 tokenizer，而是在给你展示“同一个模型系统里，词表层怎么从最朴素版本升级到更实用版本”。从课程角度看，它也很合适作为后半程的算法章节，因为 BPE 的核心循环足够清楚：统计相邻 pair，再合并最高频 pair。你可以把它完整地写出来，也能清楚验证每一步在做什么。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 对应 step | `step11/` |
| 学员文件 | `framework/student.c`（唯一需要改的文件） |
| 共享代码 | `../../framework/` 下的 `bpe_tokenizer.{h,c}`、`tensor.{h,c}` 等 |
| 编译 | `make` |
| 运行 | `make test`（自动跑 `verify_run_all`） |
| 验证 | `make clean && make test` |

涉及的 framework API（你**只**会用到这一小撮）：

- `bpe_tokenizer_create()` / `bpe_tokenizer_free(tok)`
- `bpe_train(tok, text, num_merges)` —— 完整训练循环
- `bpe_encode(tok, text, &out_len, 0, 0)` —— 编码成 token id 数组
- `bpe_decode(tok, ids, len)` —— 把 id 数组拼回文本
- `bpe_decode_token(tok, id)` —— 单 token 反查字符串

> 你**不会**直接调 framework 内部 `count_pairs` / `merge_pair`——那两个函数是 `static` 的；学员必须自己写等价版本。

## 实验内容

这一章虽然只有两个 student 函数，但它们刚好切中了 BPE 训练里最不可替代的两个动作。第一题决定“下一轮应该合并谁”，第二题决定“合并动作落到整条序列上后会发生什么”。只要这两个动作成立，完整训练循环其实就已经被你抓住主干了。

### 1.5.1 阅读 `framework/student.c`，理解 TODO 函数（15 分钟）

打开 `framework/student.c`，里面有两个 `// TODO(student)` 块：

- `student_bpe_count_pairs(tokens, len, out_first, out_second)` —— 给一段 int 数组（即 token 序列），找出**频次最高**的相邻 pair，把它的两个 id 写进 `*out_first` / `*out_second`，返回出现次数。**注意**频次 < 2 时返回 0。
- `student_bpe_apply_merge(tokens, len, first, second, new_id, &out_len)` —— 在 `tokens` 里把所有"值等于 (first, second) 的相邻 pair"替换成 `new_id`，返回新数组（长度写进 `*out_len`）。

两个函数都用纯 C、不调 framework 私有 API。

这一步先读 TODO 的意义，在于让你把“统计”和“替换”当成两种不同职责来看，而不是一上来就把 BPE 当成一整个大黑箱。课程越到后面，越强调这种拆职责的习惯，因为综合章真正需要的就是把多个窄职责重新串起来。

### 1.5.2 实现 `student_bpe_count_pairs`（40 分钟）

在 `framework/student.c` 的 `student_bpe_count_pairs` 里写循环：

- 双重循环扫一遍相邻对 `(tokens[i], tokens[i+1])`。
- 维护一个小数组（或简单的"找最大 + 计数"两遍扫描法），记录出现次数最多的那一对。
- 边界：`len < 2` 直接返回 0；最高频 < 2 也返回 0（与 `bpe_train` 内部"找不到就停"一致）。

> **提示**：参考 `framework/bpe_tokenizer.c` 里 `count_pairs` + `find_most_frequent_pair` 的逻辑，但**不能直接调**。建议 O(n^2) 暴力即可——本 lab 输入不超过几千个 token。

跑：

```bash
cd course/practice/labs/lab12-step11
make clean && make test
```

预期至少 `[TEST 2] ... [PASS]`。没看到？先单独 `printf` 一下 `*out_first` / `*out_second`，看是否填对。

这一题最值得你盯住的是“频次最高”这四个字。BPE 的训练并不神秘，本质上就是在问：当前语料里，哪两个相邻 token 最值得被视为一个更大的整体。只要这层直觉建立了，后面你再看 `bpe_vocab.txt` 里那些高频子词，就不会觉得它们是凭空冒出来的。

### 1.5.3 实现 `student_bpe_apply_merge`（40 分钟）

继续在同一文件里实现"在序列里替换 pair"：

- 新开一个 `result` 数组（长度 = `len`，最坏情况不增长）。
- 扫到 `(tokens[i], tokens[i+1]) == (first, second)` 时，写入 `new_id` 并 `i++` 跳过下一个；否则照抄 `tokens[i]`。
- 写回 `*out_len`。
- 返回新数组的指针（caller 负责 `free`）。

> **细节**：第一遍建议保留 1 个 sample "ab" 合并成 "ab"，手动走一遍 —— 一定要处理 i 越界 (`i < len-1`)。

跑 `make clean && make test`，预期 `[TEST 3] ... [PASS]`。

这一题是“把统计结果真正落到序列上”的那一步。也正因为如此，它非常适合帮助你建立另一个工程直觉：算法里“找到规则”和“把规则应用到数据上”往往是两件完全不同的事。前者决定选谁，后者决定怎么改。课程故意把它们拆开，就是为了让你在代码里也保留这种结构。

### 1.5.4 跑通整个 4-测试 verify（20 分钟）

`framework/verify.c` 一共 4 个 TEST，覆盖：

1. `student_bpe_count_pairs` 找最高频 pair 正确；
2. `student_bpe_count_pairs` 边界（频次 < 2 返回 0）；
3. `student_bpe_apply_merge` 替换 pair 后长度正确、内容正确；
4. 用你的两个函数手工跑 50 轮 BPE 训练，得到的 token 序列和 `bpe_train` 一致。

跑 `make clean && make test` 直到 4 个 `[PASS]` 全亮。

### 1.5.5 （选做）对比字符级和 BPE 的 token 数（30 分钟）

参考 chapter 12.3 的 `compare_tok.c` 写法，在你自己的 `student.c` 里再加一个 `int main` 之外的辅助函数（或临时塞进 `verify_run_all` 末尾），加载 `../../step11/bpe_vocab.txt`（如果它存在），打印一句 `"hello world how are you"` 的 BPE token 数和字符数。

> 提示：bpe_vocab.txt 在 step11 里默认**不存在**——你得先 `cd ../../step11 && make && ./train_bpe train_english.txt 50` 跑出来。本子任务只算附加分。

## 现象

`Lab12` 当前实验第一次运行时不会全红。因为 `count_pairs` 的一个边界测试不依赖完整实现，所以你会先看到 1 项通过，其余 3 项失败。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 1] count_pairs: most frequent pair in "ababab"             [FAIL]
[TEST 2] count_pairs: returns 0 when no pair freq>=2             [PASS]
[TEST 3] apply_merge: replaces (a,b) -> ab in sequence           [FAIL]
[TEST 4] mini-BPE loop: 50 merges shortens the sequence          [FAIL]
1/4 tests passed
```

### 完成本章 TODO 后的目标输出

```
$ cd course/practice/labs/lab12-step11
$ make clean && make test
mkdir -p build
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/student.c -o build/student.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c framework/verify.c -o build/verify.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/bpe_tokenizer.c -o build/bpe_tokenizer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tensor.c -o build/tensor.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/math_ops.c -o build/math_ops.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/tokenizer.c -o build/tokenizer.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -c ../../framework/embedding.c -o build/embedding.o
gcc -Wall -Wextra -O2 -std=c99 -I../../framework -o student build/student.o build/verify.o \
    build/bpe_tokenizer.o build/tensor.o build/math_ops.o build/tokenizer.o build/embedding.o -lm
[TEST 1] count_pairs: most frequent pair in "ababab"    ... [PASS]
[TEST 2] count_pairs: returns 0 when no pair freq>=2    ... [PASS]
[TEST 3] apply_merge: replaces (a,b) -> ab in sequence  ... [PASS]
[TEST 4] mini-BPE loop: 50 merges matches bpe_train     ... [PASS]
ALL TESTS PASSED (4/4)
```

这一章比较直接：看到 `ALL TESTS PASSED (4/4)`，说明 pair 统计、merge、以及 50 轮 mini-BPE 闭环都已经成立。

本章结束后，你最好已经能把 BPE 看成“在词表层做压缩与抽象”的系统能力，而不是“又一个 tokenizer 技巧”。只要这点想清楚，下一章把 BPE、KV Cache、生成和对话重新串起来时，你就会知道每一层究竟在解决哪一种不同的问题。

> 4 个测试里任何一个 `[FAIL]`，按失败信息回到对应 TODO 函数检查；常见错误是 `(a,b)` 顺序写反、`i` 越界忘了 `i < len-1`、返回值漏写 `*out_len`。

## 思考题

1. **（必做）** 为什么 BPE 的"找最高频 pair"那一步要做"频次 ≥ 2 才合并"？如果把阈值改成 1，会发生什么？提示：训练集里每个字符至少出现 1 次。
2. **（必做）** `student_bpe_apply_merge` 是 O(n) 的——扫一遍数组。如果训练一次 BPE 要做 K=500 次合并、每次在 N=10000 个 token 上跑，你的总时间复杂度是多少？和 `bpe_train` 实际跑 500 步比，能差几倍？提示：每次合并后 N 会变小。
3. **（选做）** BPE 是"无监督"的吗？想想它训练时需要什么数据——是带标签的句子还是无标签的纯文本？
4. **（选做）** 如果 `num_merges=0`，会发生什么？想一下 `bpe_train` 那个 `for (k=0; k<num_merges; k++)` 循环——0 次等于不做任何合并，结果就是字符级。
5. **（选做）** 为什么 `chat_bpe` 比 step8 的 `chat` 加载慢？提示：词汇表从 256 涨到 600 多，embedding 表变大。

## 上节

[Lab11 - KV Cache](/course/practice/labs/lab11-step10/)：上一章把推理的 O(n) 重复计算砍到 O(1) 复用。

## 下节

[Lab13 - BPE 对话整合](/course/practice/labs/lab13-end-to-end/)：把 Lab08、Lab09、Lab11、Lab12 串起来，形成一个真正能跑的 BPE 对话回路。
