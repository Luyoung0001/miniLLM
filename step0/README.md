# Step 0: 基础设施 - 张量库与数学运算

本步骤实现 miniLLM 的基础设施：张量数据结构和数学运算库。

## 目标

1. 实现多维张量数据结构
2. 实现基础数学运算（矩阵乘法、逐元素运算等）
3. 实现激活函数（ReLU、GELU、Softmax 等）
4. 实现归约运算（sum、mean、max 等）

## 知识铺垫

### 什么是张量？

张量是多维数组的泛化：
- **0 维张量**：标量（scalar），如 `3.14`
- **1 维张量**：向量（vector），如 `[1, 2, 3]`
- **2 维张量**：矩阵（matrix），如 `[[1, 2], [3, 4]]`
- **N 维张量**：更高维的数组

在深度学习中，张量是数据和参数的基本表示形式。

### 内存布局

我们使用**行优先（row-major）**顺序存储数据，这是 C 语言的默认方式。

对于 2x3 矩阵：
```
[[a, b, c],
 [d, e, f]]
```
在内存中存储为：`[a, b, c, d, e, f]`

### 步长（Strides）

步长用于计算多维索引对应的一维偏移量。

对于形状 `[2, 3]` 的张量：
- `strides = [3, 1]`
- 元素 `[i, j]` 的偏移量 = `i * 3 + j * 1`

### 矩阵乘法

矩阵乘法是神经网络的核心运算：

```
C[i,j] = Σ_k A[i,k] * B[k,j]
```

对于 A[M,K] @ B[K,N] = C[M,N]：
- 需要 M×K×N 次乘法
- 时间复杂度 O(n³)

### 激活函数

激活函数为神经网络引入非线性：

**ReLU（Rectified Linear Unit）**
```
ReLU(x) = max(0, x)
```

**GELU（Gaussian Error Linear Unit）**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```
GELU 是 Transformer 中常用的激活函数。

**Softmax**
```
Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```
将任意实数向量转换为概率分布。

### 数值稳定性

计算 Softmax 时，直接计算 exp(x) 可能导致数值溢出。解决方法：
```
Softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

## 文件结构

```
step0/
├── Makefile          # 构建脚本
├── README.md         # 本文档
├── src/
│   ├── tensor.h      # 张量数据结构声明
│   ├── tensor.c      # 张量操作实现
│   ├── math_ops.h    # 数学运算声明
│   ├── math_ops.c    # 数学运算实现
│   └── main.c        # 测试程序
└── build/            # 编译输出目录
```

## 核心 API

### 张量创建

```c
// 创建指定形状的张量
Tensor* tensor_create(int ndim, int* shape);

// 创建全零/全一张量
Tensor* tensor_zeros(int ndim, int* shape);
Tensor* tensor_ones(int ndim, int* shape);

// 创建随机张量（正态分布）
Tensor* tensor_randn(int ndim, int* shape, float mean, float std);

// 便捷函数
Tensor* tensor_create_2d(int rows, int cols);
```

### 张量操作

```c
// 索引访问
float tensor_get(Tensor* t, int* indices);
void tensor_set(Tensor* t, int* indices, float value);

// 拷贝和释放
Tensor* tensor_copy(Tensor* src);
void tensor_free(Tensor* t);
```

### 矩阵运算

```c
// 矩阵乘法: C = A @ B
Tensor* matmul(Tensor* a, Tensor* b);

// 矩阵-向量乘法
void matvec(Tensor* out, Tensor* mat, Tensor* vec);
```

### 激活函数

```c
void relu(Tensor* out, Tensor* in);
void gelu(Tensor* out, Tensor* in);
void softmax(Tensor* out, Tensor* in);
void sigmoid(Tensor* out, Tensor* in);
```

### 归约运算

```c
float tensor_sum(Tensor* t);
float tensor_mean(Tensor* t);
float tensor_max(Tensor* t);
int tensor_argmax(Tensor* t);
```

## 编译和运行

```bash
# 编译
make

# 运行测试
make test

# 清理
make clean

# 内存检查（需要 valgrind）
make memcheck
```

## 预期输出

运行 `make test` 后，应看到类似输出：

```
========================================
      miniLLM Step0 测试套件
========================================

=== 测试: 张量创建 ===
[PASS] 创建 1D 张量
[PASS] 1D 张量维度正确
...

========================================
测试结果: XX 通过, 0 失败
========================================
```

## 验收标准

- [ ] 所有测试通过
- [ ] 矩阵乘法结果与手算一致
- [ ] Softmax 输出和为 1
- [ ] 无内存泄漏（valgrind 检测通过）

## 下一步

完成 Step 0 后，继续 Step 1 实现 Tokenizer。
