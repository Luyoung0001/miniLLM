# miniLLM Step4 Makefile
# Transformer Block 实现

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# 目录
SRC_DIR = src
BUILD_DIR = build

# 源文件
SRCS = $(SRC_DIR)/tensor.c $(SRC_DIR)/math_ops.c $(SRC_DIR)/tokenizer.c \
       $(SRC_DIR)/embedding.c $(SRC_DIR)/attention.c \
       $(SRC_DIR)/layernorm.c $(SRC_DIR)/ffn.c $(SRC_DIR)/transformer.c \
       $(SRC_DIR)/main.c

OBJS = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/math_ops.o $(BUILD_DIR)/tokenizer.o \
       $(BUILD_DIR)/embedding.o $(BUILD_DIR)/attention.o \
       $(BUILD_DIR)/layernorm.o $(BUILD_DIR)/ffn.o $(BUILD_DIR)/transformer.o \
       $(BUILD_DIR)/main.o

# 目标
TARGET = step4

# 默认目标
all: $(BUILD_DIR) $(TARGET)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 链接
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 编译规则
$(BUILD_DIR)/tensor.o: $(SRC_DIR)/tensor.c $(SRC_DIR)/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/math_ops.o: $(SRC_DIR)/math_ops.c $(SRC_DIR)/math_ops.h $(SRC_DIR)/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.c $(SRC_DIR)/tokenizer.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/embedding.o: $(SRC_DIR)/embedding.c $(SRC_DIR)/embedding.h $(SRC_DIR)/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/attention.o: $(SRC_DIR)/attention.c $(SRC_DIR)/attention.h $(SRC_DIR)/tensor.h $(SRC_DIR)/math_ops.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/layernorm.o: $(SRC_DIR)/layernorm.c $(SRC_DIR)/layernorm.h $(SRC_DIR)/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/ffn.o: $(SRC_DIR)/ffn.c $(SRC_DIR)/ffn.h $(SRC_DIR)/tensor.h $(SRC_DIR)/math_ops.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/transformer.o: $(SRC_DIR)/transformer.c $(SRC_DIR)/transformer.h \
                            $(SRC_DIR)/layernorm.h $(SRC_DIR)/attention.h $(SRC_DIR)/ffn.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(SRC_DIR)/tensor.h $(SRC_DIR)/math_ops.h \
                     $(SRC_DIR)/layernorm.h $(SRC_DIR)/ffn.h $(SRC_DIR)/transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

# 运行测试
test: $(TARGET)
	./$(TARGET)

# 调试版本
debug: CFLAGS += -g -DDEBUG
debug: clean all

# 清理
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# 内存检查
memcheck: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

.PHONY: all test debug clean memcheck
