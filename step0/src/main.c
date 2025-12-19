#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "math_ops.h"

#define EPSILON 1e-5
#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

void assert_true(int condition, const char* test_name) {
    if (condition) {
        printf("[%s] %s\n", TEST_PASS, test_name);
        tests_passed++;
    } else {
        printf("[%s] %s\n", TEST_FAIL, test_name);
        tests_failed++;
    }
}

void assert_float_eq(float a, float b, const char* test_name) {
    if (fabsf(a - b) < EPSILON) {
        printf("[%s] %s (%.6f == %.6f)\n", TEST_PASS, test_name, a, b);
        tests_passed++;
    } else {
        printf("[%s] %s (%.6f != %.6f)\n", TEST_FAIL, test_name, a, b);
        tests_failed++;
    }
}

// ============ 张量创建测试 ============

void test_tensor_create() {
    printf("\n=== 测试: 张量创建 ===\n");

    // 测试 1D 张量
    Tensor* t1 = tensor_create_1d(5);
    assert_true(t1 != NULL, "创建 1D 张量");
    assert_true(t1->ndim == 1, "1D 张量维度正确");
    assert_true(t1->shape[0] == 5, "1D 张量形状正确");
    assert_true(t1->size == 5, "1D 张量大小正确");
    tensor_free(t1);

    // 测试 2D 张量
    Tensor* t2 = tensor_create_2d(3, 4);
    assert_true(t2 != NULL, "创建 2D 张量");
    assert_true(t2->ndim == 2, "2D 张量维度正确");
    assert_true(t2->shape[0] == 3 && t2->shape[1] == 4, "2D 张量形状正确");
    assert_true(t2->size == 12, "2D 张量大小正确");
    tensor_free(t2);

    // 测试 3D 张量
    Tensor* t3 = tensor_create_3d(2, 3, 4);
    assert_true(t3 != NULL, "创建 3D 张量");
    assert_true(t3->ndim == 3, "3D 张量维度正确");
    assert_true(t3->size == 24, "3D 张量大小正确");
    tensor_free(t3);
}

void test_tensor_zeros_ones() {
    printf("\n=== 测试: zeros/ones ===\n");

    int shape[] = {2, 3};
    Tensor* zeros = tensor_zeros(2, shape);
    assert_true(zeros != NULL, "创建 zeros 张量");
    int all_zero = 1;
    for (int i = 0; i < zeros->size; i++) {
        if (zeros->data[i] != 0.0f) all_zero = 0;
    }
    assert_true(all_zero, "zeros 张量全为 0");
    tensor_free(zeros);

    Tensor* ones = tensor_ones(2, shape);
    assert_true(ones != NULL, "创建 ones 张量");
    int all_one = 1;
    for (int i = 0; i < ones->size; i++) {
        if (ones->data[i] != 1.0f) all_one = 0;
    }
    assert_true(all_one, "ones 张量全为 1");
    tensor_free(ones);
}

void test_tensor_indexing() {
    printf("\n=== 测试: 张量索引 ===\n");

    Tensor* t = tensor_create_2d(3, 4);
    // 填充数据: 0, 1, 2, ..., 11
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)i;
    }

    // 测试 tensor_get
    int idx00[] = {0, 0};
    int idx12[] = {1, 2};
    int idx23[] = {2, 3};
    assert_float_eq(tensor_get(t, idx00), 0.0f, "索引 [0,0]");
    assert_float_eq(tensor_get(t, idx12), 6.0f, "索引 [1,2]");
    assert_float_eq(tensor_get(t, idx23), 11.0f, "索引 [2,3]");

    // 测试 tensor_set
    tensor_set(t, idx12, 99.0f);
    assert_float_eq(tensor_get(t, idx12), 99.0f, "设置后索引 [1,2]");

    tensor_free(t);
}

void test_tensor_copy() {
    printf("\n=== 测试: 张量拷贝 ===\n");

    int shape[] = {2, 3};
    Tensor* a = tensor_randn(2, shape, 0.0f, 1.0f);
    Tensor* b = tensor_copy(a);

    assert_true(b != NULL, "拷贝张量成功");
    assert_true(a->data != b->data, "拷贝张量数据地址不同");
    assert_true(tensor_shape_equal(a, b), "拷贝张量形状相同");

    int data_equal = 1;
    for (int i = 0; i < a->size; i++) {
        if (a->data[i] != b->data[i]) data_equal = 0;
    }
    assert_true(data_equal, "拷贝张量数据相同");

    tensor_free(a);
    tensor_free(b);
}

// ============ 矩阵运算测试 ============

void test_matmul() {
    printf("\n=== 测试: 矩阵乘法 ===\n");

    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // A @ B = [[19, 22], [43, 50]]
    Tensor* a = tensor_create_2d(2, 2);
    Tensor* b = tensor_create_2d(2, 2);
    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {5, 6, 7, 8};
    tensor_fill_data(a, data_a);
    tensor_fill_data(b, data_b);

    Tensor* c = matmul(a, b);
    assert_true(c != NULL, "矩阵乘法返回非空");
    assert_true(c->shape[0] == 2 && c->shape[1] == 2, "矩阵乘法形状正确");
    assert_float_eq(c->data[0], 19.0f, "矩阵乘法 [0,0]=19");
    assert_float_eq(c->data[1], 22.0f, "矩阵乘法 [0,1]=22");
    assert_float_eq(c->data[2], 43.0f, "矩阵乘法 [1,0]=43");
    assert_float_eq(c->data[3], 50.0f, "矩阵乘法 [1,1]=50");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);

    // 测试非方阵: [2,3] @ [3,4] = [2,4]
    Tensor* m1 = tensor_ones(2, (int[]){2, 3});
    Tensor* m2 = tensor_ones(2, (int[]){3, 4});
    Tensor* m3 = matmul(m1, m2);
    assert_true(m3->shape[0] == 2 && m3->shape[1] == 4, "非方阵乘法形状正确");
    assert_float_eq(m3->data[0], 3.0f, "非方阵乘法结果正确");

    tensor_free(m1);
    tensor_free(m2);
    tensor_free(m3);
}

void test_matvec() {
    printf("\n=== 测试: 矩阵-向量乘法 ===\n");

    // mat = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // vec = [1, 1, 1]
    // result = [6, 15]
    Tensor* mat = tensor_create_2d(2, 3);
    float data_mat[] = {1, 2, 3, 4, 5, 6};
    tensor_fill_data(mat, data_mat);

    Tensor* vec = tensor_ones(1, (int[]){3});
    Tensor* out = tensor_zeros(1, (int[]){2});

    matvec(out, mat, vec);
    assert_float_eq(out->data[0], 6.0f, "矩阵-向量乘法 [0]=6");
    assert_float_eq(out->data[1], 15.0f, "矩阵-向量乘法 [1]=15");

    tensor_free(mat);
    tensor_free(vec);
    tensor_free(out);
}

// ============ 激活函数测试 ============

void test_relu() {
    printf("\n=== 测试: ReLU ===\n");

    Tensor* in = tensor_create_1d(5);
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    tensor_fill_data(in, data);

    Tensor* out = tensor_create_1d(5);
    relu(out, in);

    assert_float_eq(out->data[0], 0.0f, "ReLU(-2) = 0");
    assert_float_eq(out->data[1], 0.0f, "ReLU(-1) = 0");
    assert_float_eq(out->data[2], 0.0f, "ReLU(0) = 0");
    assert_float_eq(out->data[3], 1.0f, "ReLU(1) = 1");
    assert_float_eq(out->data[4], 2.0f, "ReLU(2) = 2");

    tensor_free(in);
    tensor_free(out);
}

void test_softmax() {
    printf("\n=== 测试: Softmax ===\n");

    // 测试 1D softmax
    Tensor* in1d = tensor_create_1d(3);
    float data1d[] = {1.0f, 2.0f, 3.0f};
    tensor_fill_data(in1d, data1d);

    Tensor* out1d = tensor_create_1d(3);
    softmax(out1d, in1d);

    float sum1d = tensor_sum(out1d);
    assert_float_eq(sum1d, 1.0f, "1D Softmax 和为 1");
    assert_true(out1d->data[2] > out1d->data[1] && out1d->data[1] > out1d->data[0],
                "1D Softmax 保持单调性");

    tensor_free(in1d);
    tensor_free(out1d);

    // 测试 2D softmax (每行)
    Tensor* in2d = tensor_create_2d(2, 3);
    float data2d[] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    tensor_fill_data(in2d, data2d);

    Tensor* out2d = tensor_create_2d(2, 3);
    softmax(out2d, in2d);

    float row0_sum = out2d->data[0] + out2d->data[1] + out2d->data[2];
    float row1_sum = out2d->data[3] + out2d->data[4] + out2d->data[5];
    assert_float_eq(row0_sum, 1.0f, "2D Softmax 第一行和为 1");
    assert_float_eq(row1_sum, 1.0f, "2D Softmax 第二行和为 1");

    // 第二行相等输入，输出应该接近 1/3
    assert_float_eq(out2d->data[3], 1.0f/3.0f, "均匀输入 softmax 接近 1/3");

    tensor_free(in2d);
    tensor_free(out2d);
}

void test_gelu() {
    printf("\n=== 测试: GELU ===\n");

    Tensor* in = tensor_create_1d(3);
    float data[] = {-1.0f, 0.0f, 1.0f};
    tensor_fill_data(in, data);

    Tensor* out = tensor_create_1d(3);
    gelu(out, in);

    // GELU(-1) ≈ -0.1588, GELU(0) = 0, GELU(1) ≈ 0.8413
    assert_true(fabsf(out->data[0] - (-0.1588f)) < 0.01f, "GELU(-1) ≈ -0.159");
    assert_float_eq(out->data[1], 0.0f, "GELU(0) = 0");
    assert_true(fabsf(out->data[1] - 0.8413f) < 0.01f || out->data[2] > 0.8f, "GELU(1) ≈ 0.841");

    tensor_free(in);
    tensor_free(out);
}

// ============ 归约运算测试 ============

void test_reductions() {
    printf("\n=== 测试: 归约运算 ===\n");

    Tensor* t = tensor_create_1d(5);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    tensor_fill_data(t, data);

    assert_float_eq(tensor_sum(t), 15.0f, "sum = 15");
    assert_float_eq(tensor_mean(t), 3.0f, "mean = 3");
    assert_float_eq(tensor_max(t), 5.0f, "max = 5");
    assert_float_eq(tensor_min(t), 1.0f, "min = 1");
    assert_true(tensor_argmax(t) == 4, "argmax = 4");
    assert_true(tensor_argmin(t) == 0, "argmin = 0");

    // 方差和标准差
    // var = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = (4+1+0+1+4)/5 = 2
    assert_float_eq(tensor_var(t), 2.0f, "var = 2");
    assert_float_eq(tensor_std(t), sqrtf(2.0f), "std = sqrt(2)");

    tensor_free(t);
}

void test_axis_reductions() {
    printf("\n=== 测试: 按轴归约 ===\n");

    // 2x3 矩阵
    // [[1, 2, 3],
    //  [4, 5, 6]]
    Tensor* t = tensor_create_2d(2, 3);
    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_fill_data(t, data);

    // 按 axis=0 求和: [5, 7, 9]
    Tensor* sum0 = tensor_sum_axis(t, 0);
    assert_true(sum0->ndim == 1 && sum0->shape[0] == 3, "axis=0 求和形状正确");
    assert_float_eq(sum0->data[0], 5.0f, "axis=0 sum[0]=5");
    assert_float_eq(sum0->data[1], 7.0f, "axis=0 sum[1]=7");
    assert_float_eq(sum0->data[2], 9.0f, "axis=0 sum[2]=9");
    tensor_free(sum0);

    // 按 axis=1 求和: [6, 15]
    Tensor* sum1 = tensor_sum_axis(t, 1);
    assert_true(sum1->ndim == 1 && sum1->shape[0] == 2, "axis=1 求和形状正确");
    assert_float_eq(sum1->data[0], 6.0f, "axis=1 sum[0]=6");
    assert_float_eq(sum1->data[1], 15.0f, "axis=1 sum[1]=15");
    tensor_free(sum1);

    tensor_free(t);
}

// ============ 逐元素运算测试 ============

void test_elementwise() {
    printf("\n=== 测试: 逐元素运算 ===\n");

    Tensor* a = tensor_create_1d(3);
    Tensor* b = tensor_create_1d(3);
    float data_a[] = {1.0f, 2.0f, 3.0f};
    float data_b[] = {4.0f, 5.0f, 6.0f};
    tensor_fill_data(a, data_a);
    tensor_fill_data(b, data_b);

    Tensor* out = tensor_create_1d(3);

    // 加法
    tensor_add(out, a, b);
    assert_float_eq(out->data[0], 5.0f, "a + b [0] = 5");
    assert_float_eq(out->data[1], 7.0f, "a + b [1] = 7");

    // 减法
    tensor_sub(out, a, b);
    assert_float_eq(out->data[0], -3.0f, "a - b [0] = -3");

    // 乘法
    tensor_mul(out, a, b);
    assert_float_eq(out->data[0], 4.0f, "a * b [0] = 4");
    assert_float_eq(out->data[2], 18.0f, "a * b [2] = 18");

    // 标量乘法
    tensor_mul_scalar(out, a, 2.0f);
    assert_float_eq(out->data[0], 2.0f, "a * 2 [0] = 2");
    assert_float_eq(out->data[2], 6.0f, "a * 2 [2] = 6");

    tensor_free(a);
    tensor_free(b);
    tensor_free(out);
}

// ============ 数学函数测试 ============

void test_math_functions() {
    printf("\n=== 测试: 数学函数 ===\n");

    Tensor* in = tensor_create_1d(4);
    float data[] = {1.0f, 2.0f, 4.0f, 9.0f};
    tensor_fill_data(in, data);

    Tensor* out = tensor_create_1d(4);

    // exp
    tensor_exp(out, in);
    assert_float_eq(out->data[0], expf(1.0f), "exp(1)");
    assert_float_eq(out->data[1], expf(2.0f), "exp(2)");

    // log
    tensor_log(out, in);
    assert_float_eq(out->data[0], logf(1.0f), "log(1) = 0");
    assert_float_eq(out->data[1], logf(2.0f), "log(2)");

    // sqrt
    tensor_sqrt(out, in);
    assert_float_eq(out->data[2], 2.0f, "sqrt(4) = 2");
    assert_float_eq(out->data[3], 3.0f, "sqrt(9) = 3");

    // square
    tensor_square(out, in);
    assert_float_eq(out->data[1], 4.0f, "2^2 = 4");
    assert_float_eq(out->data[2], 16.0f, "4^2 = 16");

    tensor_free(in);
    tensor_free(out);
}

void test_batched_matmul() {
    printf("\n=== 测试: 批量矩阵乘法 ===\n");

    // 形状 [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
    int shape_a[] = {2, 2, 3};
    int shape_b[] = {2, 3, 2};
    Tensor* a = tensor_ones(3, shape_a);
    Tensor* b = tensor_ones(3, shape_b);

    Tensor* c = batched_matmul(a, b);
    assert_true(c != NULL, "批量矩阵乘法返回非空");
    assert_true(c->shape[0] == 2 && c->shape[1] == 2 && c->shape[2] == 2,
                "批量矩阵乘法形状正确 [2,2,2]");
    assert_float_eq(c->data[0], 3.0f, "批量矩阵乘法结果正确");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

// ============ 主函数 ============

int main() {
    printf("========================================\n");
    printf("      miniLLM Step0 测试套件\n");
    printf("========================================\n");

    // 张量测试
    test_tensor_create();
    test_tensor_zeros_ones();
    test_tensor_indexing();
    test_tensor_copy();

    // 矩阵运算测试
    test_matmul();
    test_matvec();
    test_batched_matmul();

    // 激活函数测试
    test_relu();
    test_softmax();
    test_gelu();

    // 归约运算测试
    test_reductions();
    test_axis_reductions();

    // 逐元素运算测试
    test_elementwise();

    // 数学函数测试
    test_math_functions();

    // 打印总结
    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
