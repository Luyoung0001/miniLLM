# Step 1: Tokenizer - 文本分词器

本步骤实现字符级 Tokenizer，将文本转换为模型可处理的数字序列。

## 目标

1. 实现字符级分词器
2. 支持特殊 token (PAD, UNK, BOS, EOS)
3. 支持文本编码和解码
4. 支持词汇表的保存和加载

## 知识铺垫

### 什么是 Tokenizer？

Tokenizer 是 NLP 中将文本转换为数字的工具。LLM 无法直接处理文本，需要先将文本转换为数字序列。

**分词方式对比：**

| 方式 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | "hello" → [h, e, l, l, o] | 词汇表小，无 OOV | 序列长 |
| 词级 | "hello world" → [hello, world] | 语义完整 | OOV 问题 |
| 子词级 (BPE) | "playing" → [play, ing] | 平衡 | 复杂 |

本项目使用**字符级**分词，简单易懂，适合教学。

### 特殊 Token

| Token | 作用 |
|-------|------|
| `<PAD>` | 填充，用于对齐不同长度的序列 |
| `<UNK>` | 未知字符，处理词汇表外的字符 |
| `<BOS>` | 序列开始标记 |
| `<EOS>` | 序列结束标记 |

### 编码过程

```
文本: "Hi"
  ↓
字符: ['H', 'i']
  ↓
ASCII: [72, 105]
  ↓
Token ID: [76, 109]  (加上特殊 token 偏移 4)
```

### 词汇表结构

```
ID 0:   <PAD>
ID 1:   <UNK>
ID 2:   <BOS>
ID 3:   <EOS>
ID 4:   (ASCII 0, 通常不使用)
...
ID 36:  (ASCII 32, 空格)
...
ID 69:  (ASCII 65, 'A')
...
ID 101: (ASCII 97, 'a')
...
ID 259: (ASCII 255)
```

总词汇表大小: 4 (特殊) + 256 (ASCII) = **260**

## 文件结构

```
step1/
├── Makefile
├── README.md
├── src/
│   ├── tensor.h        # (来自 step0)
│   ├── tensor.c
│   ├── math_ops.h
│   ├── math_ops.c
│   ├── tokenizer.h     # 分词器声明
│   ├── tokenizer.c     # 分词器实现
│   └── main.c          # 测试程序
└── data/
    └── vocab.txt       # 词汇表文件 (可选)
```

## 核心 API

### 创建和销毁

```c
// 创建字符级 tokenizer
Tokenizer* tokenizer_create(void);

// 从文件加载
Tokenizer* tokenizer_load(const char* vocab_path);

// 释放内存
void tokenizer_free(Tokenizer* tok);

// 保存词汇表
int tokenizer_save(Tokenizer* tok, const char* path);
```

### 编码和解码

```c
// 编码文本为 token ID 序列
int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len,
                      int add_bos, int add_eos);

// 解码 token ID 序列为文本
char* tokenizer_decode(Tokenizer* tok, int* ids, int len);

// 单字符编码
int tokenizer_encode_char(Tokenizer* tok, char c);

// 单 ID 解码
const char* tokenizer_decode_id(Tokenizer* tok, int id);
```

## 使用示例

```c
#include "tokenizer.h"

int main() {
    // 创建 tokenizer
    Tokenizer* tok = tokenizer_create();

    // 编码
    int len;
    int* ids = tokenizer_encode(tok, "Hello", &len, 1, 1);
    // ids = [2, 76, 105, 112, 112, 115, 3]
    //       BOS  H    e    l    l    o   EOS

    // 解码
    char* text = tokenizer_decode(tok, ids, len);
    // text = "Hello"

    // 打印 token 信息
    tokenizer_print_tokens(tok, ids, len);

    free(ids);
    free(text);
    tokenizer_free(tok);
    return 0;
}
```

## 编译和运行

```bash
# 编译
make

# 运行测试
make test

# 清理
make clean
```

## 预期输出

```
========================================
      miniLLM Step1 测试套件
         Tokenizer 测试
========================================

=== 测试: Tokenizer 创建 ===
[PASS] 创建 tokenizer 成功
[PASS] 词汇表大小为 260
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 创建 tokenizer 成功
- [ ] 特殊 token 正确设置
- [ ] 字符编码正确 (ASCII + 偏移)
- [ ] 编解码往返一致: decode(encode(text)) == text
- [ ] 词汇表保存和加载正确
- [ ] 无内存泄漏

## 下一步

完成 Step 1 后，继续 Step 2 实现 Embedding 层。
