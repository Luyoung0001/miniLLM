#!/usr/bin/env python3
"""
transform.py — 把 course/chapters/chXX-*.md 的章节草稿自动重排版为课程实验说明骨架。

注意：当前课程已经不再使用“至少打印 3 个 [PASS]”和“截图占位”这类旧模板。
如果继续使用本脚本，应在生成后补写两类真实信息：
  1. 当前脚手架基线（以实际 `make clean && make test` 结果为准）
  2. 完成本章 TODO 后的目标输出

不修改任何正文文字，只：
  1. 在文件顶部插入实验信息块（实验编号、预计耗时、对应 step、本节产出）
  2. 把 X.1 Hook + X.2 Goal 合并为 ## 实验目的
  3. 把 X.3 Need-to-know 重命名为 ## 实验背景
  4. 把 X.4 Map 重命名为 ## 实验环境
  5. 把 X.5 Tasks 重命名为 ## 实验内容 + 子任务编号
  6. 在 ## 实验内容 下插入 mermaid 数据流占位和“基线待实测”占位
  7. 把 X.6 Verification 重命名为 ## 现象
  8. 把 X.7 Common mistakes 重命名为 ## 常见失败
  9. 把 X.8 Thinking questions 重命名为 ## 思考题（必做 / 选做标记）
  10. 把 X.9 Summary + X.10 Bridge 合并为 ## 小结 + ## 上节/下节

用法:
  python3 transform.py                # 转换 chapters/ 下所有 .md
  python3 transform.py ch04-*.md      # 转换指定文件
  python3 transform.py --dry-run      # 只打印不写
"""
import re
import sys
import os
from pathlib import Path

CHAPTERS_DIR = Path(__file__).parent.parent.parent / "chapters"
APPENDICES_DIR = Path(__file__).parent.parent.parent / "appendices"

# 每个 chapter 顶部的"实验信息块"——根据 chapter 文件名映射
LAB_META = {
    "ch01-step0-tensor.md":       ("Lab01", "3h",  "step0",  "完成张量读写与 softmax TODO，并核对真实基线"),
    "ch02-step1-tokenizer.md":    ("Lab02", "2h",  "step1",  "完成字符级 tokenizer TODO，并核对真实基线"),
    "ch03-step2-embedding.md":    ("Lab03", "2h",  "step2",  "完成位置编码与 embedding forward TODO，并核对真实基线"),
    "ch04-step3-attention.md":    ("Lab04", "3h",  "step3",  "完成 attention scores / mask / softmax TODO，并核对真实基线"),
    "ch05-step4-transformer.md":  ("Lab05", "3h",  "step4",  "完成 LayerNorm、残差、Block 组装 TODO，并核对真实基线"),
    "ch06-step5-gpt.md":          ("Lab06", "3h",  "step5",  "完成完整 GPT 组装与保存 TODO，并核对真实基线"),
    "ch07-step6-training.md":     ("Lab07", "3h",  "step6",  "完成交叉熵、梯度、Adam TODO，并核对真实基线"),
    "ch08-step7-generation.md":   ("Lab08", "2h",  "step7",  "完成 greedy、temperature、top-k 采样 TODO，并核对真实基线"),
    "ch09-step8-chat.md":         ("Lab09", "2h",  "step8",  "完成 prompt 拼接与回复提取 TODO，并核对真实基线"),
    "ch10-step9-http.md":         ("Lab10", "2h",  "step9",  "完成请求行解析与响应构建 TODO，并核对真实基线"),
    "ch11-step10-kv-cache.md":    ("Lab11", "3h",  "step10", "完成 KV Cache alloc/append TODO，并核对真实基线"),
    "ch12-step11-bpe.md":         ("Lab12", "3h",  "step11", "完成 BPE pair 统计与 merge TODO，并核对真实基线"),
    "ch13-end-to-end.md":         ("Lab13", "3h",  "step11", "完成 BPE 对话整合 TODO，并核对真实基线"),
}

# 基线占位模板
ASCII_BASELINE = """
```
$ cd ../labs/lab{NN}-{step}
$ make clean && make test
[在这里粘贴当前脚手架的真实基线输出]
```

> 这里不要保留占位结果。生成文档后，请用实际运行得到的基线输出替换本段。
"""

# Mermaid 占位——每个 lab 不同的数据流图
MERMAID_TEMPLATES = {
    "ch01": "graph LR\n  A[input ids] --> B[Tensor*]\n  B --> C[matmul / softmax / relu]\n  C --> D[output Tensor*]",
    "ch02": "graph LR\n  A[text] --> B[tokenizer_encode]\n  B --> C[int ids]\n  C --> D[tokenizer_decode]\n  D --> E[text]",
    "ch03": "graph LR\n  A[token ids] --> B[Token Embedding]\n  P[Position Embedding] --> B\n  B --> C[output seq_len x hidden_dim]",
    "ch04": "graph LR\n  Q --> S[Q @ K^T]\n  K --> S\n  S --> M[+ mask]\n  M --> P[softmax]\n  P --> O[@ V]\n  V --> O",
    "ch05": "graph LR\n  X[input] --> LN1[LayerNorm]\n  LN1 --> ATTN[Multi-Head Attention]\n  ATTN --> A1[+ residual]\n  A1 --> LN2[LayerNorm]\n  LN2 --> FFN[FFN]\n  FFN --> A2[+ residual]\n  A2 --> OUT[output]",
    "ch06": "graph LR\n  A[Token ids] --> B[Embedding]\n  B --> C[N x Transformer Block]\n  C --> D[Final LayerNorm]\n  D --> E[LM Head]\n  E --> F[logits]",
    "ch07": "graph LR\n  X[input] --> F[forward + cache]\n  F --> L[cross entropy + grad]\n  L --> B[backward]\n  B --> C[grad clip]\n  C --> A[adam step]\n  A --> X",
    "ch08": "graph LR\n  P[prompt ids] --> M[model forward]\n  M --> L[last logits]\n  L --> S[sampling: T / K / P]\n  S --> T[next token]\n  T --> P",
    "ch09": "graph LR\n  C[Conversation] --> F[format_prompt]\n  F --> P[prompt string]\n  P --> M[model forward]\n  M --> R[response]\n  R --> E[extract_response]\n  E --> C",
    "ch10": "graph LR\n  Client --> S[HTTP Server]\n  S --> J[JSON parse]\n  J --> C[chat system]\n  C --> M[model]\n  M --> R[JSON response]\n  R --> S\n  S --> Client",
    "ch11": "graph LR\n  P[prompt] --> PRE[prefill: K/V 写 cache]\n  PRE --> CACHE[(KV Cache)]\n  CACHE --> DEC[decode: 只算新位置]\n  DEC --> NEW[new K/V 追加]\n  NEW --> CACHE\n  DEC --> T[next token]\n  T --> DEC",
    "ch12": "graph LR\n  T[text] --> BPE[bpe_encode]\n  BPE --> IDS[int ids]\n  IDS --> M[model forward]\n  M --> OUT[out ids]\n  OUT --> D[bpe_decode]\n  D --> TXT[text]",
    "ch13": "graph LR\n  CORPUS[corpus.txt] --> T[train_bpe]\n  T --> V[bpe_vocab.txt]\n  T --> M[model_bpe.bin]\n  M --> C[chat_bpe REPL]\n  C --> M",
}


def info_block(lab_id, hours, step, deliverable):
    """practice 风格顶部信息块（blockquote 形式）"""
    return (
        "> **实验编号** {lab} &nbsp;&nbsp; **预计耗时** {hours} &nbsp;&nbsp; "
        "**对应 step** [{step}](../{step}/) &nbsp;&nbsp; **本节产出** {deliverable}\n\n"
    ).format(lab=lab_id, hours=hours, step=step, deliverable=deliverable)


def mermaid_block(chapter_key):
    tmpl = MERMAID_TEMPLATES.get(chapter_key)
    if not tmpl:
        return ""
    return (
        "\n### 关键数据流\n\n"
        "```mermaid\n"
        + tmpl + "\n"
        "```\n\n"
        "> 图占位：mermaid 数据流图，标注每一步的输入输出\n"
    )


def transform_chapter(text, fname):
    """对单章 md 做重排版（不删文字，只插入和重命名）"""
    # 1. 顶部 frontmatter
    text = re.sub(r"^---\n.*?\n---\n\n?", "", text, count=1, flags=re.DOTALL)

    # 2. 顶部 "># 对应 step：..." 这块 quote
    text = re.sub(r"^> \*\*对应 step\*\*.*?\n", "", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^> \*\*预计时间\*\*.*?\n", "", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^> \*\*本章产物\*\*.*?\n", "", text, count=1, flags=re.MULTILINE)

    # 3. 提取 chapter 编号（章号数字）—— 用 search 跳过前导 frontmatter
    m = re.search(r"^# Chapter (\d+)", text, re.MULTILINE)
    if not m:
        return None
    ch_num = m.group(1)

    # 4. 段名重命名（按出现顺序）
    section_map = [
        (r"^## \d+\.1 Hook\s*$",                          f"## 实验目的"),
        (r"^## \d+\.2 Goal\s*$",                          "__MERGE_GOAL__"),  # 并入目的
        (r"^## \d+\.3 Need-to-know\s*$",                  f"## 实验背景"),
        (r"^## \d+\.4 Map\s*$",                           f"## 实验环境"),
        (r"^## \d+\.5 Tasks\s*$",                         f"## 实验内容"),
        (r"^## \d+\.6 Verification\s*$",                  f"## 现象"),
        (r"^## \d+\.7 Common mistakes\s*$",               f"## 常见失败"),
        (r"^## \d+\.8 Thinking questions\s*$",            f"## 思考题"),
        (r"^## \d+\.9 Summary\s*$",                       f"## 小结"),
        (r"^## \d+\.10 Bridge(?: to Chapter \d+)?(?: — .*)?\s*$",   f"## 上节 / 下节"),
    ]
    for pat, repl in section_map:
        text = re.sub(pat, repl, text, count=1, flags=re.MULTILINE)

    # 5. 把 X.2 Goal 的列表并入"实验目的"——找到 "__MERGE_GOAL__" 标记
    # 简化：直接把 Goal 那一段原样保留（仍是有用信息），由人工/读者忽略
    text = text.replace("__MERGE_GOAL__\n", "## 实验目的（续：能做什么）\n", 1)

    # 6. 在 ## 实验内容 的 task 子节后插入基线 + mermaid 占位
    # 简化：只在 ## 实验内容 标题下插入一次
    chapter_key = "ch" + ch_num.zfill(2)
    placeholders = (
        "\n"
        + "### 关键数据流\n\n```mermaid\n"
        + MERMAID_TEMPLATES.get(chapter_key, "graph LR\n  A[input] --> B[output]")
        + "\n```\n\n"
        + "### 当前脚手架基线（待实测）\n\n"
        + ASCII_BASELINE
        + "\n"
    )
    text = text.replace("## 实验内容\n", "## 实验内容\n" + placeholders, 1)

    # 7. 思考题子项前加"必做/选做"标记（启发式：第 1、2 题必做，其后选做）
    def tag_questions(match):
        block = match.group(0)
        # 找 "1. **xxx**" 模式，加 **(必做)** 标记
        questions = re.findall(r"^(\d+)\.\s+\*\*(.+?)\*\*", block, flags=re.MULTILINE)
        for i, (num, title) in enumerate(questions):
            tag = "(必做)" if i < 2 else "(选做)"
            old = "{num}. **{title}**".format(num=num, title=title)
            new = "{num}. **{title}** {tag}".format(num=num, title=title, tag=tag)
            block = block.replace(old, new, 1)
        return block
    text = re.sub(r"## 思考题\n.*?(?=\n## |\Z)", tag_questions, text, count=1, flags=re.DOTALL)

    # 8. 顶部加 frontmatter + practice 风格信息块
    lab_id, hours, step, deliverable = LAB_META.get(fname, ("Lab??", "?", "?", "?"))
    frontmatter = (
        "---\n"
        "title: {title}\n"
        "lab: {lab}\n"
        "step: {step}\n"
        "hours: {hours}\n"
        "deliverable: {deliv}\n"
        "---\n\n".format(
            title="Lab" + ch_num.zfill(2) + " — " + os.path.splitext(fname)[0].split("-", 1)[1].replace("-", " "),
            lab=lab_id, step=step, hours=hours, deliv=deliverable,
        )
    )
    info = info_block(lab_id, hours, step, deliverable)
    return frontmatter + info + text


def transform_appendix(text, fname):
    """附录的简单处理：只加 frontmatter + 顶部引言，不动正文"""
    text = re.sub(r"^---\n.*?\n---\n\n?", "", text, count=1, flags=re.DOTALL)
    frontmatter = (
        "---\n"
        "title: " + os.path.splitext(fname)[0] + "\n"
        "type: appendix\n"
        "---\n\n"
        "> **附录**：按需查阅，不必从头读\n\n"
    )
    return frontmatter + text


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]

    targets = []
    if args:
        for a in args:
            # 先按原样在两个目录找
            found = False
            for d in (CHAPTERS_DIR, APPENDICES_DIR):
                p = d / a
                if p.exists():
                    targets.append(p)
                    found = True
                    break
            if not found:
                # 兼容裸文件名匹配
                for d in (CHAPTERS_DIR, APPENDICES_DIR):
                    matches = list(d.glob(a))
                    if matches:
                        targets.extend(matches)
                        break
    else:
        targets = sorted(CHAPTERS_DIR.glob("ch*.md")) + sorted(APPENDICES_DIR.glob("A*.md"))

    if not targets:
        print("No files to transform", file=sys.stderr)
        sys.exit(1)

    for p in targets:
        original = p.read_text(encoding="utf-8")
        if p.name.startswith("ch"):
            new = transform_chapter(original, p.name)
        else:
            new = transform_appendix(original, p.name)
        if new is None:
            print("SKIP (no chapter match):", p)
            continue
        if dry_run:
            print("--- DRY RUN:", p, "---")
            print(new[:200], "...")
        else:
            # 直接覆盖原文件（git 里有原始版本，回滚简单）
            p.write_text(new, encoding="utf-8")
            print("WROTE", p)


if __name__ == "__main__":
    main()
