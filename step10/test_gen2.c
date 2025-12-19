#include <stdio.h>
#include <stdlib.h>
#include "src/model.h"
#include "src/tokenizer.h"
#include "src/generate.h"

int main() {
    GPTModel* model = model_load("model.bin");
    Tokenizer* tok = tokenizer_create();
    
    GenerateConfig cfg = default_generate_config();
    cfg.max_new_tokens = 40;
    cfg.temperature = 0.8f;
    cfg.do_sample = 1;  // 采样
    cfg.top_k = 10;
    cfg.seed = 42;
    
    const char* prompts[] = {
        "hello ",
        "the ",
        "one ",
        "good ",
        "how "
    };
    
    printf("=== 生成测试 (温度采样 T=0.8) ===\n\n");
    
    for (int i = 0; i < 5; i++) {
        cfg.seed = 42 + i;
        char* result = generate_with_kv_cache(model, tok, prompts[i], &cfg);
        printf("输入: '%s'\n", prompts[i]);
        printf("输出: '%s'\n\n", result ? result : "(null)");
        if (result) free(result);
    }
    
    tokenizer_free(tok);
    model_free(model);
    return 0;
}
