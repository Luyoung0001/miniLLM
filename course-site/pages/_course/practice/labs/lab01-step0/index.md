---
title: 'Lab01 - 张量与数学运算'
layout: 'course_page'
permalink: '/course/practice/labs/lab01-step0/'
lab: 'Lab01'
step: 'step0'
hours: '3h'
deliverable: '完成 framework/student.c 中本章 TODO，并用验证命令检查结果'
source_path: 'course/practice/labs/lab01-step0/TASK.md'
---


> **实验编号** Lab01 &nbsp;&nbsp; **预计耗时** 3h &nbsp;&nbsp; **对应 step** [step0](https://github.com/Luyoung0001/miniLLM/tree/main/step0) &nbsp;&nbsp; **本节产出** 补完本章 TODO，并让验证输出相对当前实验基线继续转绿

## 实验目的

完成本 lab 后，你应当能够：

1. 解释 `Tensor` 结构体里 `ndim / shape / strides / data` 各自的含义。
2. 用 stride 公式 `offset = i * strides[0] + j * strides[1]` 手算 2D 张量的一维偏移。
3. 解释 `softmax` 为什么必须先减 `max(x)`，并写出 3 步数值稳定版本。
4. 区分 `tensor_set`（写）和 `tensor_get`（读）的对称性——它们用同一组 stride。

## 实验环境

- **涉及文件**：
  - `framework/student.c` —— 你**唯一**需要修改的文件（含 `TODO(student)` 注释）
  - `framework/student.h` —— 3 个待实现函数声明
  - `framework/verify.c` —— 自动验证程序（**不要改**）
  - `../../framework/tensor.c` 与 `math_ops.c` —— 共享实现（通过 Makefile 链接）
- **命令**：
  ```bash
  make clean && make test
  ```

## 实验内容

### 1.5.1 跑通 baseline

```bash
make clean && make test
```

应当看到 4 个 `[TEST N] ... [FAIL]`（因为你的 `student_*` 还没实现，全是占位）。这是预期。

### 1.5.2 实现 `student_tensor_get`

打开 `framework/student.c`，找到 `student_tensor_get` 函数。

阅读 `TODO(student)` 提示，写 3~5 行实现：

```
offset = i * t->strides[0] + j * t->strides[1]
return t->data[offset]
```

边界：t 为 NULL 或越界时返回 0。

### 1.5.3 实现 `student_tensor_set`

与 1.5.2 对称——只是反过来写 `data[offset] = value`。

### 1.5.4 实现 `student_softmax_stable`

3 步：

1. 遍历 `in[]` 找最大值 `max`
2. `out[i] = expf(in[i] - max)`
3. 归一化：`out[i] /= sum(out)`

### 1.5.5 跑通 4 个测试

```bash
make clean && make test
```

## 现象

这一节第一次运行时，请先确认你看到的是**当前实验基线**，而不是已经完成实验后的理想输出。`Lab01` 的作用就是让你在最小闭环里看到：环境通了、verify 通了、但 3 个核心函数还没写。

### 当前实验基线（2026-06-17 复核）

```text
[TEST 1]  ... [FAIL]
[TEST 2]  ... [FAIL]
[TEST 3]  ... [FAIL]
[TEST 4]  ... [PASS]
3 test(s) FAILED.
```

这说明 `student_tensor_get`、`student_tensor_set`、`student_softmax_stable` 仍然是空实现，但 Makefile 和验证器已经在正常工作。

### 完成本章 TODO 后的目标输出

```
==========================================
       Lab01 - 张量与数学运算 验证
==========================================
[TEST 1]  ... [PASS]
[TEST 2]  ... [PASS]
[TEST 3]  ... [PASS]
[TEST 4]  ... [PASS]
==========================================
All tests passed!
```

## 常见失败

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `[TEST 1] ... [FAIL] get(1,2)=0.0` | 没实现 `student_tensor_get` 或 strides 取错 | 打印 offset 检查 |
| `[TEST 3] ... [FAIL] sum(out) != 1.0` | softmax 没归一化 | 第 3 步漏了 `out[i] /= sum` |
| `[TEST 4] ... [FAIL] NaN/Inf` | 没减 max | 第 1 步加 max 减法 |
| `undefined reference to student_tensor_get` | 漏写 `extern` 或函数名打错 | 对照 student.h |

## 思考题

1. **(必做)** 如果把 `student_tensor_get` 里的 `i * t->strides[0]` 写成 `i * t->shape[1]`，2D 张量能跑通吗？换成 3D 张量会怎样？
2. **(必做)** `softmax` 的 3 步里，第 2 步的 `expf` 在输入是 `-1e9` 时会返回什么？`-1e9 - max` 呢？为什么要先减 max？
3. **(选做)** 把 `student_tensor_get` 改成支持 3D 张量（用 `for k` 循环遍历 `ndim`），跑通 `tensor_create(3, ...)` 的张量。
4. **(选做)** 不用 `expf`，自己用泰勒级数实现 `exp(x)`，精度 1e-3 即可。

## 上节 / 下节

- **上节**：[`Chapter 0` — 出发前的准备](/course/00-orientation/)
- **下节**：[`Lab02 - 字符级 Tokenizer`](/course/practice/labs/lab02-step1/)
