# miniLLM Step11 - BPE Tokenizer
# Makefile for BPE-based training and inference

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

# Source files
SRC_DIR = src
SRCS = $(SRC_DIR)/tensor.c $(SRC_DIR)/math_ops.c $(SRC_DIR)/embedding.c \
       $(SRC_DIR)/attention.c $(SRC_DIR)/layernorm.c $(SRC_DIR)/ffn.c \
       $(SRC_DIR)/transformer.c $(SRC_DIR)/model.c $(SRC_DIR)/kv_cache.c \
       $(SRC_DIR)/bpe_tokenizer.c

# Training sources (additional)
TRAIN_SRCS = $(SRC_DIR)/loss.c $(SRC_DIR)/backward.c $(SRC_DIR)/optimizer.c

# Targets
.PHONY: all clean train chat demo test

all: train_bpe chat_bpe demo_bpe continue_train test_bpe

# BPE Training program
train_bpe: train_bpe.c $(SRCS) $(TRAIN_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Continue training program
continue_train: continue_train.c $(SRCS) $(TRAIN_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Interactive chat program
chat_bpe: chat_bpe.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Demo program
demo_bpe: demo_bpe.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# BPE tokenizer test
test_bpe: test_bpe.c $(SRC_DIR)/bpe_tokenizer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Quick commands
train: train_bpe
	./train_bpe train_english.txt 200

chat: chat_bpe
	./chat_bpe

demo: demo_bpe
	./demo_bpe

test: test_bpe
	./test_bpe

clean:
	rm -f train_bpe chat_bpe demo_bpe continue_train test_bpe
	rm -f *.o
