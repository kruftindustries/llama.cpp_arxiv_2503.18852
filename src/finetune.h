#ifndef FINETUNE_H
#define FINETUNE_H

#include <stddef.h>
#include <stdbool.h>

// Forward declarations
struct llama_model;
struct llama_context;
struct llama_graph_reasoning_s;

#ifdef __cplusplus
extern "C" {
#endif

// Finetune parameters
struct llama_finetune_params {
    float learning_rate;
    float weight_decay;
    int batch_size;
    int epochs;
    bool use_graph_reasoning;
};

// Default parameters
struct llama_finetune_params llama_finetune_default_params();

// Initialize model for finetuning
bool llama_model_finetune_init(struct llama_model* model);

// Finetune on token sequence
bool llama_finetune(struct llama_context* ctx, 
                   const int* tokens, int n_tokens,
                   const struct llama_finetune_params* params);

// Save finetuned model
bool llama_model_finetune_save(struct llama_model* model, const char* filename);

#ifdef __cplusplus
}
#endif

#endif // FINETUNE_H
