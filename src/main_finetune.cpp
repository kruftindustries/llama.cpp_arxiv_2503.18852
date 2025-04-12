#include "finetune.h"
#include "graph_reasoning.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

// Forward declarations for llama.cpp functions
extern "C" {
    struct llama_model* llama_load_model_from_file(const char* path, int n_ctx);
    struct llama_context* llama_new_context(struct llama_model* model);
    void llama_free_model(struct llama_model* model);
    void llama_free_context(struct llama_context* ctx);
    int llama_tokenize(struct llama_context* ctx, const char* text, int* tokens, int n_max_tokens);
}

// Finetune command parameters
struct finetune_cmd_params {
    std::string model_in;
    std::string model_out;
    std::string dataset;
    int epochs = 1;
    float learning_rate = 1e-5f;
    bool use_graph_reasoning = false;
};

// Parse command line arguments
bool parse_finetune_params(int argc, char** argv, finetune_cmd_params& params) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-in") == 0 && i+1 < argc) {
            params.model_in = argv[++i];
        } else if (strcmp(argv[i], "--model-out") == 0 && i+1 < argc) {
            params.model_out = argv[++i];
        } else if (strcmp(argv[i], "--dataset") == 0 && i+1 < argc) {
            params.dataset = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) {
            params.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i+1 < argc) {
            params.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--use-graph-reasoning") == 0) {
            params.use_graph_reasoning = true;
        }
    }
    
    return !params.model_in.empty() && !params.model_out.empty() && !params.dataset.empty();
}

// Load text file
std::string load_text_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return "";
    }
    
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    std::string result(size, '\0');
    fread(&result[0], 1, size, f);
    fclose(f);
    
    return result;
}

// Finetune command implementation
int finetune_main(int argc, char** argv) {
    finetune_cmd_params params;
    if (!parse_finetune_params(argc, argv, params)) {
        fprintf(stderr, "Usage: %s finetune --model-in MODEL --model-out OUT --dataset FILE [--epochs N] [--learning-rate R] [--use-graph-reasoning]\n", argv[0]);
        return 1;
    }
    
    // Load model
    struct llama_model* model = llama_load_model_from_file(params.model_in.c_str(), 2048);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", params.model_in.c_str());
        return 1;
    }
    
    // Initialize for finetuning
    if (!llama_model_finetune_init(model)) {
        fprintf(stderr, "Failed to initialize model for finetuning\n");
        llama_free_model(model);
        return 1;
    }
    
    // Create context
    struct llama_context* ctx = llama_new_context(model);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }
    
    // Initialize graph reasoning if requested
    llama_graph_reasoning* gr = nullptr;
    if (params.use_graph_reasoning) {
        gr = llama_graph_reasoning_init();
        // In a real implementation, we'd associate with context:
        // ctx->graph_reasoning = gr;
    }
    
    // Load dataset
    std::string dataset_text = load_text_file(params.dataset);
    if (dataset_text.empty()) {
        fprintf(stderr, "Failed to load dataset '%s'\n", params.dataset.c_str());
        if (gr) llama_graph_reasoning_free(gr);
        llama_free_context(ctx);
        llama_free_model(model);
        return 1;
    }
    
    // Tokenize dataset
    std::vector<int> tokens(dataset_text.size()); // Rough upper bound
    int n_tokens = llama_tokenize(ctx, dataset_text.c_str(), tokens.data(), tokens.size());
    if (n_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize dataset\n");
        if (gr) llama_graph_reasoning_free(gr);
        llama_free_context(ctx);
        llama_free_model(model);
        return 1;
    }
    tokens.resize(n_tokens);
    
    // Set up finetune parameters
    struct llama_finetune_params ft_params = llama_finetune_default_params();
    ft_params.learning_rate = params.learning_rate;
    ft_params.epochs = params.epochs;
    ft_params.use_graph_reasoning = params.use_graph_reasoning;
    
    // Run finetuning
    printf("Finetuning model with %d tokens for %d epochs...\n", n_tokens, params.epochs);
    if (!llama_finetune(ctx, tokens.data(), n_tokens, &ft_params)) {
        fprintf(stderr, "Finetuning failed\n");
        if (gr) llama_graph_reasoning_free(gr);
        llama_free_context(ctx);
        llama_free_model(model);
        return 1;
    }
    
    // Save model
    printf("Saving model to '%s'...\n", params.model_out.c_str());
    if (!llama_model_finetune_save(model, params.model_out.c_str())) {
        fprintf(stderr, "Failed to save model\n");
        if (gr) llama_graph_reasoning_free(gr);
        llama_free_context(ctx);
        llama_free_model(model);
        return 1;
    }
    
    // Cleanup
    if (gr) llama_graph_reasoning_free(gr);
    llama_free_context(ctx);
    llama_free_model(model);
    
    printf("Finetuning complete!\n");
    return 0;
}

// Add finetune command handler to main.cpp

// Add this function to main.cpp (or create a separate entry point):
/*
int main(int argc, char** argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "finetune") == 0) {
            return finetune_main(argc-1, argv+1);
        }
        // Other commands...
    }
    
    // Regular main code...
}
*/
