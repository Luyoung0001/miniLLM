---
title: 'Lab 与实践说明'
layout: 'course_page'
permalink: '/course/practice/'
prev_page:
  title: 'Chapter 0 — 出发前的准备'
  url: '/course/00-orientation/'
next_page:
  title: 'Chapter 1 — step0 张量与数学运算'
  url: '/course/chapters/ch01-step0-tensor/'
source_path: 'course/practice/README.md'
---

从 `Chapter 1` 开始，这门课就不再只是阅读材料，而是要求你真正动手写代码。后面的每一章都会配一个对应的 lab。chapter 负责把知识讲清楚，lab 负责把这一章真正要做的代码落下来。

如果你把这门课想成“先看讲义，再做实验”，这里就是实验区的总入口。

## 先把 lab 仓库拉到本地

读课程网页时，你只需要拉取这个 lab 仓库：

```bash
git clone https://github.com/Luyoung0001/minillm_lab.git
cd minillm_lab
bash scripts/bootstrap-practice.sh
```

后面所有需要你修改的代码，都在这个仓库里。课程网页会用 `labs/...` 描述实验位置；这些路径都以 `minillm_lab` 仓库根目录为起点。

lab 仓库地址是：

```text
https://github.com/Luyoung0001/minillm_lab.git
```

## 你应该怎样使用这里

最推荐的节奏非常固定：

1. 先读对应 chapter，明白这章为什么需要这个新概念。
2. 再进入对应 lab，读 `TASK.md`。
3. 打开 `framework/student.c`，只做这一章留给你的部分。
4. 运行 `make clean && make test`，检查结果。

不要反过来一上来就直接改 `student.c`。因为这门课不是纯题单，每一章前面的解释会告诉你为什么现在轮到这个问题、为什么代码恰好改在这个位置。

## 目录长什么样

在 lab 仓库下面，最重要的是 `labs/`：

```text
labs/
├── lab01-step0/
├── lab02-step1/
├── ...
└── lab13-end-to-end/
```

每个 lab 都是一章正式课程对应的动手部分。

以 `lab01-step0/` 为例，它通常包含：

```text
lab01-step0/
├── Makefile
├── TASK.md
└── framework/
    ├── student.c
    ├── student.h
    ├── verify.c
    └── verify.h
```

这里几份文件的作用很简单：

- `TASK.md`：告诉你这一章到底要完成什么。
- `framework/student.c`：你主要修改的地方。
- `framework/verify.c`：自动验证程序。
- `Makefile`：本章编译和测试入口。

## 第一次进入 lab 时会看到什么

如果你刚完成 `Chapter 0`，那么第一站就是：

```bash
cd labs/lab01-step0
make clean && make test
```

第一次运行时，看到部分测试失败是正常的。因为 `student.c` 里本来就留着待完成的函数，课程就是要你把这些空缺一步步补起来。

对 `Lab01` 而言，当前正常的起始现象是：

```text
[TEST 1] ... [FAIL]
[TEST 2] ... [FAIL]
[TEST 3] ... [FAIL]
[TEST 4] ... [PASS]
3 test(s) FAILED.
```

这不是环境坏了，而是说明第一章对应的练习边界已经准备好了。

## 推荐工作流

从 `Lab01` 开始，后面每一章都尽量保持同样的做法：

1. 读 chapter。
2. 读 `TASK.md`。
3. 改 `framework/student.c`。
4. 跑 `make clean && make test`。
5. 对照输出、常见错误和思考题，把这一章真正做完。

这个顺序看起来朴素，但非常有效。它能避免两种最常见的问题：

- 只看讲义，不真正写代码；
- 只盯着代码改，却不知道为什么要这样改。

## 现在从哪里开始

如果你刚跑完 `Chapter 0`，下一步就是：

1. 读 [Chapter 1](/course/chapters/ch01-step0-tensor/)
2. 打开 [Lab01 的任务说明](/course/practice/labs/lab01-step0/)
3. 开始修改 `labs/lab01-step0/framework/student.c`
