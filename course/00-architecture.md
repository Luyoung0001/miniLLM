---
title: miniLLM 课程架构
---

# miniLLM 课程架构

这份文档不是给学员拿来直接照做的章节，而是给课程作者、助教、以及后续继续重构这套课程的人看的“骨架说明”。它回答一个更上游的问题：为什么这门课现在要被组织成“Chapter 0 先完成讲义交接，再进入 practice scaffold，一章一 lab 逐步完成项目”的样子。

如果只看项目本身，miniLLM 已经有 `step0/` 到 `step11/` 这样一套很清楚的渐进实现。但如果要把它变成真正适合零基础本科生的课程，仅有 step 目录还不够。课程不仅要给出“项目如何长成”，还要给出“学员如何进入、如何书写、如何验证、如何逐步完成”。这正是本架构文档要固定下来的东西。

## Learner And Goal

### 学习者画像

这门课面向的是“能看懂基础 C 语法，但还没有系统做过一个完整工程项目”的本科生。这里的“基础 C 语法”包括：

- 理解函数、结构体、数组、指针；
- 接受 `malloc/free`、头文件、`.c/.h` 分离编译这些基本工程形式；
- 能在终端里执行 `cd`、`ls`、`make` 这类基础命令。

但它默认学员**没有**以下背景：

- 没有系统学过深度学习；
- 没写过 PyTorch；
- 没推过 Transformer 公式；
- 没做过大型代码阅读；
- 没用过成熟的调试和验证习惯。

因此，这门课的目标不只是把“LLM 的知识点”列出来，而是要用一条低摩擦的路径，把学员从“能读基本 C”带到“能一步步把一个小型 GPT 做出来并讲明白”。

### 最终目标

完整走完课程后，学员应当能完成两件事。

第一件事是项目能力：在 CPU 上训练、加载、生成并对话一个纯 C 写成的小型 GPT，理解从 tokenizer、embedding、attention、transformer block、完整模型、训练循环，到生成、对话、HTTP、KV cache、BPE 的完整链路。

第二件事是课程能力：能说清楚每个阶段为什么会在这个时候引入，自己改过哪些代码，哪些验证结果证明了本章已经完成。

后者很重要。因为真正的“学会”，不只是把程序跑出来，而是知道自己为什么这样改，以及这些输出究竟证明了什么。

## Project Evidence Map

课程架构必须建立在项目证据之上，而不是建立在抽象愿望之上。当前仓库提供的关键事实如下：

| 维度 | 事实 | 仓库证据 |
| --- | --- | --- |
| 参考实现主线 | `step0/` 到 `step11/` 逐步增加能力 | 根目录 `step*/` |
| 最终能力 | 训练、生成、对话、HTTP、KV cache、BPE | `step7/` 到 `step11/` |
| 语言与依赖 | 纯 C，主要依赖 libc / libm | 各 `Makefile` |
| 当前课程讲义 | `course/chapters/` 提供逐章说明 | `course/chapters/*.md` |
| 当前实践骨架 | `course/practice/labs/` 提供 lab 目录、`student.c`、`verify.c` | `course/practice/labs/` |
| 共享框架 | practice scaffold 通过 `course/practice/framework/` 复用基础实现 | `course/practice/framework/` |

最关键的一点是：这个仓库现在已经同时拥有“讲义”和“实践框架”。这意味着课程可以不再停留在“读 step README + 自己猜该怎么练”的阶段，而可以明确转向“讲解 + 骨架填空 + 自动验证”的教学模式。

## Delivery Topology

当前课程采用的是：

- **course repo docs**：`course/`
- **practice scaffold**：`course/practice/`
- **reference solution**：`step0/` 到 `step11/`

从 Git 拓扑上看，practice scaffold 还没有拆成独立 submodule；但从教学角色上看，它已经完全按“独立 practice repo”的方式工作。

这意味着课程现在应当明确使用如下交付模型：

1. `Chapter 0` 只做准备和角色交接。
2. `Chapter 0` 结束时，学员必须已经进入 practice scaffold。
3. `Chapter 1` 开始，讲义和 lab 配对前进。
4. 学员修改的是 practice scaffold 的 learner-owned 文件，而不是直接改参考实现。
5. 参考实现只作为讲义编写依据和事后对照来源。

这个模型比“直接在 `stepN/` 下动刀”更适合教学，原因有两个：

- 它把每章的编码边界压得很小；
- 它让每章都能拥有高信噪比的验证结果。

## 为什么 Chapter 0 不能只做环境检查

很多技术课程的第 0 章只有两件事：安装依赖、跑一个 hello world。对这个项目来说，这样做不够。

原因很简单：当前仓库并不是只有一套“应该修改的代码”。如果不在第 0 章就说明课程层、实践层、参考实现层的分工，学员很可能从一开始就在错误位置工作。

因此，Chapter 0 的职责现在被重新定义为：

- 交代项目故事；
- 交代三层目录角色；
- 确认工具链；
- 把学员送进 `course/practice/labs/lab01-step0/`；
- 让学员亲眼看到 practice scaffold 的初始 smoke test。

为了降低 Chapter 0 的歧义，现在更推荐把这个交接动作固化成脚本入口：`course/practice/scripts/bootstrap-practice.sh`。这样课程结构将来即便改成独立 practice repo 或 submodule，Chapter 0 也仍然只有一个明确出口。

只有做到这一步，正式课程才能算真正开始。

## Prerequisite Ladder

这门课仍然坚持“按项目需要引入知识”，而不是在第一周先上完整理论。

| 阶段 | 进入时要求 | 暂时不要求 |
| --- | --- | --- |
| Chapter 0 | 终端基础、能装 `gcc`/`make` | LLM 背景、数学推导 |
| Chapter 1 | 基本 C 语法 | 线性代数证明 |
| Chapter 2 | 明白字符和整数映射 | NLP 背景 |
| Chapter 3 | 向量和数组的基本直觉 | 三角函数公式推导 |
| Chapter 4 | 矩阵乘法和 softmax 直觉 | 反向传播 |
| Chapter 5-7 | 更强的数值与模型直觉 | 完整理论证明 |
| Chapter 8-13 | 已完成前面实践骨架 | 复杂系统设计背景 |

这个 prerequisite ladder 之所以成立，一个重要前提就是：实践骨架把每章任务限定在“当前已经介绍过的那一点点能力”上。

## Practice Scaffold Map

课程的核心路径必须显式映射到 practice scaffold，而不能只写“对应 stepN”。

| Chapter | Practice Path | Learner Files | Verify Command | 累积结果 |
| --- | --- | --- | --- | --- |
| 0 | `course/practice/labs/lab01-step0/` | 无修改，先 smoke test | `bash scripts/bootstrap-practice.sh` 或 `make clean && make test` | 进入正式实践环境 |
| 1 | `course/practice/labs/lab01-step0/` | `framework/student.c` | `make clean && make test` | 会处理 tensor get/set 与稳定 softmax |
| 2 | `course/practice/labs/lab02-step1/` | `framework/student.c` | `make clean && make test` | 字符级 tokenizer |
| 3 | `course/practice/labs/lab03-step2/` | `framework/student.c` | `make clean && make test` | embedding 与位置编码 |
| 4 | `course/practice/labs/lab04-step3/` | `framework/student.c` | `make clean && make test` | attention |
| 5 | `course/practice/labs/lab05-step4/` | `framework/student.c` | `make clean && make test` | transformer block |
| 6 | `course/practice/labs/lab06-step5/` | `framework/student.c` | `make clean && make test` | 完整 GPT 模型与保存加载 |
| 7 | `course/practice/labs/lab07-step6/` | `framework/student.c` | `make clean && make test` | 训练数学三件套 |
| 8 | `course/practice/labs/lab08-step7/` | `framework/student.c` | `make clean && make test` | 采样策略 |
| 9 | `course/practice/labs/lab09-step8/` | `framework/student.c` | `make clean && make test` | 提示词模板与回复提取 |
| 10 | `course/practice/labs/lab10-step9/` | `framework/student.c` | `make clean && make test` | HTTP 协议骨架 |
| 11 | `course/practice/labs/lab11-step10/` | `framework/student.c` | `make clean && make test` | KV cache 包装接口 |
| 12 | `course/practice/labs/lab12-step11/` | `framework/student.c` | `make clean && make test` | BPE 核心算子 |
| 13 | `course/practice/labs/lab13-end-to-end/` | `framework/student.c` | `make clean && make test` | BPE 对话整合 |

这里故意把“Chapter 对应 Lab”写成课程骨架的一部分。因为从现在开始，章节设计的最小单位不再只是“知识点”，而是“知识点 + 对应实践入口 + 对应验证方式”。

## Chapter Roadmap

课程主线仍然与项目模块增长保持一致，但正式表述应该从“step 演化”切换为“chapter handout + practice lab”双线并行。

| Chapter | 学员在讲义里得到什么 | 学员在实践里完成什么 | 验证信号 |
| --- | --- | --- | --- |
| 0 | 角色分工、环境准备、进入 practice scaffold | 跑通 Lab01 smoke test | `make test` 输出 3 FAIL + 1 PASS |
| 1 | 理解 tensor、stride、softmax 稳定性 | 完成 `student_tensor_get/set` 和稳定 softmax | Lab01 全 PASS |
| 2 | 理解文本到 token 的离散映射 | 完成 tokenizer 核心逻辑 | Lab02 通过 |
| 3 | 理解 embedding 与位置信息 | 完成 embedding 相关任务 | Lab03 通过 |
| 4 | 理解 self-attention 为什么必要 | 完成 attention 相关任务 | Lab04 通过 |
| 5 | 理解 block 组装逻辑 | 完成 transformer block 相关任务 | Lab05 通过 |
| 6 | 理解完整模型结构和保存加载 | 完成 GPT 模型任务 | Lab06 通过 |
| 7 | 理解训练四步循环 | 完成训练相关任务 | Lab07 通过 |
| 8 | 理解自回归生成与采样 | 完成采样相关任务 | Lab08 通过 |
| 9 | 理解对话模板与上下文管理 | 完成 prompt 与回复处理任务 | Lab09 通过 |
| 10 | 理解服务化接口 | 完成 HTTP 协议骨架任务 | Lab10 通过 |
| 11 | 理解重复计算与缓存 | 完成 KV cache 包装任务 | Lab11 通过 |
| 12 | 理解 BPE 的动机与训练流程 | 完成 BPE 核心算子任务 | Lab12 通过 |
| 13 | 把后半段组件重新串起来 | 完成 BPE 对话整合实验 | Lab13 通过 |

## 当前问题与重构方向

基于当前仓库状态，这套课程仍有几个明显问题：

1. 一部分 chapter 还残留旧模板痕迹，文风更接近 lab 说明，而不是完整讲义。
2. `course/practice/` 还没有真正拆成独立 practice repo 或 submodule，虽然角色已经确定。
3. 部分章节虽然写到了 lab 风格，但没有完全把“讲义先行、practice 紧跟”的结构统一到底。

因此，当前重构的优先级应该是：

1. 先把入口链路改正确：课程首页、Chapter 0、practice scaffold README。
2. 再把核心章节改成“讲义式正文 + 明确实践位置 + 对应 lab”。
3. 最后再考虑是否真的把 `course/practice/` 抽成 submodule。

这个顺序是务实的。因为如果讲义和课程骨架还没稳定，提前拆 submodule 只会放大维护成本。

## Migration Plan

课程后续迁移应按以下顺序执行：

1. 保留 `step0/` 到 `step11/` 作为参考实现，不把它们改造成学员主工作区。
2. 把 `course/README.md` 和 `Chapter 0` 固定为“讲义仓库 -> practice scaffold”交接模型。
3. 重写 `course/chapters/`，去掉提纲化、重复化和旧模板残留，强化讲义感。
4. 保持 `course/practice/labs/` 与 chapter 的一一映射。
5. 待课程路径稳定后，再考虑把 `course/practice/` 抽成独立 practice repo 或 submodule。

如果后续真的进入第 5 步，具体拆分命令、维护者工作流和学员拉取方式，已经整理在 [附录 A4：practice scaffold submodule 迁移说明](appendices/A4-practice-submodule-migration.md)。

## Open Questions

- `course/practice/` 何时正式抽成 Git submodule？
- 其余 chapters 是否全部需要重写为更强的讲义风格？
- 是否要为每章补一张“本章 practice path / learner-owned files / verify command”的固定小表？
