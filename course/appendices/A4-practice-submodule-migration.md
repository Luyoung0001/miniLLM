---
title: A4-practice-submodule-migration
type: appendix
---

> **附录**：这不是给学员主线阅读的章节，而是给课程维护者和助教看的迁移说明。

# 附录 A4：practice scaffold submodule 迁移说明

> **适用场景**：当你决定把当前仓库内嵌的 `course/practice/` 抽成一个真正独立的 practice repo，并以 Git submodule 的方式重新挂回课程仓库时，按这份说明做。
>
> **不适用场景**：如果你还在频繁改 chapter 到 lab 的映射关系，或者 `framework/student.c` / `verify.c` / `TASK.md` 结构还没有稳定，不要急着拆。先把课程形态稳定下来，再做仓库级拆分。

## A4.1 迁移目标先说清楚

这次迁移的目标，不是单纯“把一个目录挪出去”，而是把课程结构里的三层角色彻底固定下来：

- `course/`：讲义、章节、附录、维护文档
- `course/practice/`：practice scaffold，学员真正修改的代码骨架
- `step0/` 到 `step11/`：参考实现

迁移完成后，课程对学员呈现出来的路径应该仍然保持不变：

1. 克隆课程仓库。
2. 拉取或初始化 submodule。
3. 进入 `course/practice/`。
4. 运行 `bash scripts/bootstrap-practice.sh`。
5. 从 `Lab01` 开始逐章推进。

也就是说，这次迁移最重要的设计原则是：尽量保持路径不变，尽量保持 Chapter 0 的交接动作不变，只改变仓库归属关系。

## A4.2 为什么推荐 submodule，而不是直接把文档和实践彻底拆散

对这套课程来说，最需要保留的是“章节路径和实践路径的稳定对应关系”。如果你把 practice repo 完全独立出去，再让讲义里到处写另一份仓库路径，学员和助教都很容易在两个仓库之间来回迷路。

submodule 的优点恰好在这里：

1. 对学员来说，`course/practice/` 仍然存在，chapter 里的相对路径几乎不用改。
2. 对维护者来说，practice scaffold 可以独立发版本、独立提交、独立验收。
3. 对课程演进来说，讲义仓库和实践仓库的版本关系可以通过 submodule commit 明确锁定。

换句话说，submodule 不是为了炫 Git 技巧，而是为了保住课程阅读体验，同时把维护边界做干净。

## A4.3 迁移前检查清单

在真正拆之前，先确认下面 6 件事都成立：

1. `Chapter 0 -> bootstrap -> Lab01` 的入口链路已经稳定。
2. 每个核心 chapter 都已经对应到一个固定 lab 目录。
3. `framework/student.c`、`verify.c`、`Makefile` 的组织方式短期内不会大改。
4. `course/practice/scripts/check.sh --dry-run` 能稳定列出全部 lab。
5. 文档里主要引用的实践路径都是 `course/practice/...` 这种相对稳定的写法。
6. 你已经决定 practice repo 的远端地址和维护权限。

如果上面任意一项还不稳，先不要拆。因为一旦拆出去，之后任何大规模目录调整都会同时影响两个仓库。

## A4.4 推荐的目标拓扑

推荐采用下面这种拓扑：

```text
miniLLM/                       # 课程主仓库
├── course/
│   ├── README.md
│   ├── 00-orientation.md
│   ├── chapters/
│   ├── appendices/
│   └── practice/                 # Git submodule 挂载点
├── step0/
├── step1/
├── ...
└── step11/
```

而独立出去的 practice repo 本体，目录结构应当尽量保持和今天的 `course/practice/` 一致：

```text
miniLLM-practice/
├── README.md
├── framework/
├── labs/
└── scripts/
```

请注意，这里最值得保留的是 practice repo 自身的内部结构，而不是一定要保留原仓库名。真正影响课程路径稳定性的，是 submodule 挂载点 `course/practice/`，不是远端仓库叫不叫 `miniLLM-practice`。

## A4.5 一次性拆分步骤

下面给的是维护者视角的推荐操作顺序。假设你最终的 practice repo 远端地址叫：

```text
https://github.com/<owner>/miniLLM-practice.git
```

请把它替换成你真实准备使用的地址。

### 第一步：在新仓库里安置 practice scaffold

最简单的做法，是先把当前 `course/practice/` 的内容复制到一个新目录，初始化成独立仓库：

```bash
mkdir -p /tmp/miniLLM-practice
rsync -a course/practice/ /tmp/miniLLM-practice/
cd /tmp/miniLLM-practice
git init
git add .
git commit -m "Initialize miniLLM practice scaffold"
git remote add origin https://github.com/<owner>/miniLLM-practice.git
git branch -M main
git push -u origin main
```

如果你想保留更细的历史，可以用 `git subtree split` 或专门的过滤工具；但对于课程仓库刚起步、`course/` 还在快速演化的阶段，先把实践骨架独立出去并稳定下来，通常比保留复杂历史更重要。

### 第二步：在课程仓库里替换成 submodule

确认新仓库已经 push 成功后，回到课程仓库：

```bash
cd /path/to/miniLLM
rm -rf course/practice
git submodule add -b main https://github.com/<owner>/miniLLM-practice.git course/practice
git add .gitmodules course/practice
git commit -m "Add practice scaffold as submodule"
```

这里故意让挂载路径仍然叫 `course/practice`，就是为了让现有 chapter、`TASK.md`、脚本、附录里的绝大多数路径引用都继续成立。

### 第三步：验证 Chapter 0 入口不变

submodule 加回去之后，第一件事不是继续改文档，而是直接跑：

```bash
git submodule update --init --recursive
bash course/practice/scripts/bootstrap-practice.sh
bash course/practice/scripts/check.sh --dry-run
```

只要这两条命令还能跑通，你就已经保住了这门课最重要的入口链路。

## A4.6 学员侧工作流应该怎么改

迁移完成后，给学员的最小工作流应明确写成下面这样：

```bash
git clone --recurse-submodules https://github.com/<owner>/miniLLM.git
cd miniLLM
git submodule update --init --recursive
cd course/practice
bash scripts/bootstrap-practice.sh
```

如果学员已经先 clone 了主仓库、但忘了加 `--recurse-submodules`，补救方式就是：

```bash
git submodule update --init --recursive
```

不要在 Chapter 0 里要求学员自己猜 submodule 命令。Chapter 0 的职责仍然应该是：告诉他只要把 practice scaffold 拉下来，然后运行 bootstrap 脚本即可。

## A4.7 维护者侧工作流应该怎么改

submodule 迁移完成后，维护工作会自然分成两层。

### 讲义仓库维护者

主要负责：

- chapter 内容
- 附录
- 课程导航
- 章节到 lab 的映射
- submodule 指针升级

典型动作：

```bash
cd /path/to/miniLLM
cd course/practice
git checkout main
git pull
cd ../..
git add course/practice
git commit -m "Bump practice scaffold"
```

### practice repo 维护者

主要负责：

- `labs/*/framework/student.c`
- `verify.c`
- `Makefile`
- `scripts/check.sh`
- 共享 scaffold 结构

典型动作：

```bash
cd /path/to/miniLLM/course/practice
git checkout -b fix/lab08-topk-wording
# 修改 practice repo 内容
git add .
git commit -m "Refine Lab08 practice wording"
git push origin fix/lab08-topk-wording
```

如果你直接在 submodule 目录里改完文件却忘了提交 practice repo，本体课程仓库只会看到“submodule dirty”，不会记录具体内容。这是 submodule 最常见、也最容易让维护者困惑的点。

## A4.8 文档层面的最小改动原则

只要你保持挂载路径仍然是 `course/practice/`，迁移完成后文档只需要做三种小改动：

1. 在 `Chapter 0` 里把“进入 practice scaffold”改成“先初始化 submodule，再进入 practice scaffold”。
2. 在课程首页和 `practice/README.md` 里明确指出：`course/practice/` 现在是 Git submodule。
3. 在常见陷阱附录里补一条“忘拉 submodule”的排错项。

不要在迁移时顺手改写整套 chapter 的路径。路径不变，本来就是这次迁移最大的收益之一。

## A4.9 迁移后最值得补的一条排错项

迁移成 submodule 后，Chapter 0 最常见的新报错会变成：

- `course/practice/` 目录为空
- `scripts/bootstrap-practice.sh` 不存在
- `labs/lab01-step0` 找不到

这几乎都不是课程坏了，而是学员没有初始化 submodule。建议在 [A2：常见陷阱与调试](A2-pitfalls.md) 里后续补一条类似下面的说明：

```text
症状：course/practice/ 目录存在，但里面是空的，或只有一个 gitlink。
原因：clone 主仓库后没有执行 git submodule update --init --recursive。
解决：回到仓库根目录执行 git submodule update --init --recursive。
```

## A4.10 一条底线原则

submodule 迁移成功的标准，不是 Git 图更漂亮，而是学员读 Chapter 0 到进入 Lab01 的路径几乎不变。

如果你拆完之后，学员需要额外理解两套仓库、三套路径、四条初始化命令，说明这次迁移虽然在 Git 上做对了，但在教学上做错了。

对这门课来说，最好的迁移结果始终是：

- 课程维护边界更清楚；
- 实践骨架可以独立版本化；
- 学员仍然只需要跟着章节、脚本和 lab 顺序往前走。
