// src/graph_reasoning.h
#ifndef GRAPH_REASONING_H
#define GRAPH_REASONING_H

#include "llama.h"
#include <vector>
#include <string>
#include <unordered_map>

// Node in reasoning graph
struct llama_graph_node {
    int id;
    std::vector<float> embedding;
    std::string content;
};

// Edge in reasoning graph
struct llama_graph_edge {
    int source;
    int target;
    float weight;
    bool is_surprising;
};

// Graph reasoning structure
struct llama_graph_reasoning {
    // Graph components
    std::vector<llama_graph_node> nodes;
    std::vector<llama_graph_edge> edges;
    
    // Parameters
    float d_target;        // Critical discovery parameter target
    float alpha_target;    // Surprising edge fraction target
    float lambda_d;        // Weight for discovery parameter
    float lambda_se;       // Weight for semantic entropy
    float lambda_alpha;    // Weight for surprising edge fraction
    
    // State tracking
    float structural_entropy;
    float semantic_entropy;
    float critical_discovery_param;
    float surprising_edge_fraction;
};

// Graph reasoning functions
LLAMA_API struct llama_graph_reasoning * llama_graph_reasoning_init();
LLAMA_API void llama_graph_reasoning_free(struct llama_graph_reasoning * gr);

// Extract graph from reasoning text
LLAMA_API bool llama_graph_extract(
    struct llama_graph_reasoning * gr,
    struct llama_context * ctx,
    const char * text,
    size_t text_len);

// Calculate graph metrics
LLAMA_API void llama_graph_update_metrics(struct llama_graph_reasoning * gr);

// Calculate reward for current graph state
LLAMA_API float llama_graph_compute_reward(struct llama_graph_reasoning * gr);

#endif // GRAPH_REASONING_H
