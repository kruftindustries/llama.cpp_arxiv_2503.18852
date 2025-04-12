// src/finetune.h
#ifndef FINETUNE_H
#define FINETUNE_H

#include "llama.h"
#include <vector>

struct llama_finetune_params {
    float learning_rate;
    float weight_decay;
    int   batch_size;
    int   epochs;
    bool  use_graph_reasoning;
};

// Initialize model for finetuning
LLAMA_API bool llama_model_finetune_init(struct llama_model * model);

// Finetune on text data
LLAMA_API bool llama_finetune(
    struct llama_context * ctx,
    const llama_token * tokens,
    int n_tokens,
    const llama_finetune_params * params);

// Save finetuned model
LLAMA_API bool llama_model_finetune_save(
    struct llama_model * model,
    const char * filename);

#endif // FINETUNE_H
