#include "train.h"
#include "loss.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

TrainConfig default_train_config(void) {
    TrainConfig config;
    config.num_epochs = 10;
    config.learning_rate = 1e-3f;
    config.min_lr = 1e-5f;
    config.warmup_steps = 100;
    config.batch_size = 4;
    config.seq_len = 32;
    config.log_interval = 10;
    config.eval_interval = 100;
    config.save_interval = 500;
    config.grad_clip = 1.0f;
    config.data_path = NULL;
    config.eval_path = NULL;
    config.save_path = NULL;
    return config;
}

float perplexity(float loss) {
    return expf(loss);
}

void print_train_progress(TrainState* state) {
    printf("Step %d/%d | Epoch %d | Loss: %.4f | Avg Loss: %.4f | PPL: %.2f | LR: %.6f | Grad Norm: %.4f\n",
           state->current_step,
           state->total_steps,
           state->current_epoch,
           state->current_loss,
           state->avg_loss,
           perplexity(state->avg_loss),
           state->learning_rate,
           state->grad_norm);
}

float train_step(
    GPTModel* model,
    int* input_ids,
    int* targets,
    int batch_size,
    int seq_len,
    AdamOptimizer* optimizer,
    Gradients* grads,
    BackwardCache* cache,
    float grad_clip
) {
    if (model == NULL || input_ids == NULL || targets == NULL) return 0.0f;

    int vocab_size = model->config.vocab_size;
    float total_loss = 0.0f;

    // 清零梯度
    gradients_zero(grads);

    // 创建 logits 张量
    int logits_shape[] = {seq_len, vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);
    Tensor* grad_logits = tensor_zeros(2, logits_shape);

    // 对每个样本进行前向和反向传播
    for (int b = 0; b < batch_size; b++) {
        int* sample_input = &input_ids[b * seq_len];
        int* sample_target = &targets[b * seq_len];

        // 前向传播 (带缓存)
        model_forward_with_cache(model, sample_input, seq_len, cache, logits);

        // 计算损失和梯度
        float loss = cross_entropy_loss_with_grad(
            logits, sample_target, seq_len, vocab_size, grad_logits
        );
        total_loss += loss;

        // 反向传播
        model_backward(model, sample_input, seq_len, grad_logits, cache, grads);
    }

    // 平均损失
    total_loss /= batch_size;

    // 梯度裁剪
    float grad_norm = 0.0f;
    if (grad_clip > 0.0f) {
        grad_norm = gradients_clip(grads, grad_clip);
    } else {
        grad_norm = gradients_norm(grads);
    }

    // 参数更新
    adam_step(optimizer, model, grads);

    tensor_free(logits);
    tensor_free(grad_logits);

    return total_loss;
}

float evaluate(GPTModel* model, DataLoader* dataloader) {
    if (model == NULL || dataloader == NULL) return 0.0f;

    int vocab_size = model->config.vocab_size;
    int seq_len = dataloader->seq_len;
    int batch_size = dataloader->batch_size;

    dataloader_reset(dataloader);

    // 分配内存
    int* input_ids = (int*)malloc(batch_size * seq_len * sizeof(int));
    int* targets = (int*)malloc(batch_size * seq_len * sizeof(int));
    int logits_shape[] = {seq_len, vocab_size};
    Tensor* logits = tensor_zeros(2, logits_shape);

    float total_loss = 0.0f;
    int num_batches = 0;

    int actual_batch;
    while ((actual_batch = dataloader_next_batch(dataloader, input_ids, targets)) > 0) {
        for (int b = 0; b < actual_batch; b++) {
            model_forward(model, &input_ids[b * seq_len], seq_len, NULL, logits);
            float loss = cross_entropy_loss(logits, &targets[b * seq_len], seq_len, vocab_size);
            total_loss += loss;
        }
        num_batches += actual_batch;
    }

    free(input_ids);
    free(targets);
    tensor_free(logits);

    return (num_batches > 0) ? total_loss / num_batches : 0.0f;
}

float train(GPTModel* model, Tokenizer* tokenizer, TrainConfig* config) {
    if (model == NULL || tokenizer == NULL || config == NULL) return -1.0f;

    printf("========================================\n");
    printf("         Training Started\n");
    printf("========================================\n");

    // 创建数据加载器
    DataLoader* train_dl = NULL;
    DataLoader* eval_dl = NULL;

    if (config->data_path != NULL) {
        train_dl = dataloader_create(config->data_path, tokenizer,
                                     config->seq_len, config->batch_size);
        if (train_dl == NULL) {
            fprintf(stderr, "Error: cannot create training dataloader\n");
            return -1.0f;
        }
        dataloader_print_info(train_dl);
    } else {
        fprintf(stderr, "Error: no training data path specified\n");
        return -1.0f;
    }

    if (config->eval_path != NULL) {
        eval_dl = dataloader_create(config->eval_path, tokenizer,
                                    config->seq_len, config->batch_size);
    }

    // 创建优化器和梯度
    AdamOptimizer* optimizer = adam_create(model, config->learning_rate);
    Gradients* grads = gradients_create(model);
    BackwardCache* cache = backward_cache_create(model, config->seq_len);

    // 分配批次内存
    int* input_ids = (int*)malloc(config->batch_size * config->seq_len * sizeof(int));
    int* targets = (int*)malloc(config->batch_size * config->seq_len * sizeof(int));

    // 训练状态
    TrainState state;
    state.current_epoch = 0;
    state.current_step = 0;
    state.total_steps = config->num_epochs * dataloader_num_batches(train_dl);
    state.avg_loss = 0.0f;

    float loss_sum = 0.0f;
    int loss_count = 0;

    clock_t start_time = clock();

    // 训练循环
    for (int epoch = 0; epoch < config->num_epochs; epoch++) {
        state.current_epoch = epoch + 1;
        dataloader_reset(train_dl);
        dataloader_shuffle(train_dl);

        int actual_batch;
        while ((actual_batch = dataloader_next_batch(train_dl, input_ids, targets)) > 0) {
            state.current_step++;

            // 学习率调度
            state.learning_rate = warmup_cosine_lr(
                config->learning_rate,
                config->min_lr,
                state.current_step,
                config->warmup_steps,
                state.total_steps
            );
            optimizer->lr = state.learning_rate;

            // 训练一步
            float loss = train_step(
                model, input_ids, targets,
                actual_batch, config->seq_len,
                optimizer, grads, cache,
                config->grad_clip
            );

            state.current_loss = loss;
            state.grad_norm = gradients_norm(grads);

            loss_sum += loss;
            loss_count++;
            state.avg_loss = loss_sum / loss_count;

            // 打印日志
            if (state.current_step % config->log_interval == 0) {
                print_train_progress(&state);
            }

            // 评估
            if (eval_dl != NULL && state.current_step % config->eval_interval == 0) {
                float eval_loss = evaluate(model, eval_dl);
                printf("  [Eval] Loss: %.4f | PPL: %.2f\n", eval_loss, perplexity(eval_loss));
            }

            // 保存检查点
            if (config->save_path != NULL && state.current_step % config->save_interval == 0) {
                char checkpoint_path[256];
                snprintf(checkpoint_path, sizeof(checkpoint_path),
                         "%s.step%d", config->save_path, state.current_step);
                if (model_save(model, checkpoint_path) == 0) {
                    printf("  [Save] Checkpoint saved to %s\n", checkpoint_path);
                }
            }
        }

        printf("Epoch %d completed. Avg Loss: %.4f\n", epoch + 1, state.avg_loss);
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("========================================\n");
    printf("         Training Completed\n");
    printf("========================================\n");
    printf("Total time: %.2f seconds\n", elapsed);
    printf("Final loss: %.4f\n", state.avg_loss);
    printf("Final PPL: %.2f\n", perplexity(state.avg_loss));

    // 保存最终模型
    if (config->save_path != NULL) {
        if (model_save(model, config->save_path) == 0) {
            printf("Model saved to %s\n", config->save_path);
        }
    }

    // 清理
    free(input_ids);
    free(targets);
    backward_cache_free(cache);
    gradients_free(grads);
    adam_free(optimizer);
    if (eval_dl) dataloader_free(eval_dl);
    dataloader_free(train_dl);

    return state.avg_loss;
}
