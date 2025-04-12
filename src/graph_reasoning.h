#ifndef GRAPH_REASONING_H
#define GRAPH_REASONING_H

#include <vector>
#include <string>
#include <unordered_map>

// Forward declarations
struct ggml_tensor;
struct ggml_context;
struct llama_model;
struct llama_context;

namespace llama {

// Graph reasoning components
struct GraphNode {
    int id;
    std::vector<float> embedding;
    std::string content;
};

struct GraphEdge {
    int source;
    int target;
    float weight;
    bool is_surprising;
};

class GraphReasoning {
public:
    GraphReasoning();
    ~GraphReasoning();
    
    // Add nodes and edges
    int add_node(const std::string& content, const std::vector<float>& embedding);
    void add_edge(int source, int target, float weight = 1.0f);
    
    // Extract graph from text
    bool extract_from_text(llama_context* ctx, const char* text, size_t text_len);
    
    // Calculate entropy metrics
    float compute_structural_entropy();
    float compute_semantic_entropy();
    float compute_critical_discovery_parameter();
    float compute_surprising_edge_fraction();
    
    // Compute RL reward
    float compute_reward();
    
    // Parameters
    void set_parameters(float d_target, float alpha_target, 
                        float lambda_d, float lambda_se, float lambda_alpha);
    
private:
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;
    
    // Internal data
    std::vector<std::vector<float>> adjacency_matrix;
    std::vector<std::vector<float>> semantic_adjacency_matrix;
    
    // Parameters
    float d_target;       // Target critical discovery parameter
    float alpha_target;   // Target surprising edge fraction
    float lambda_d;       // Weight for discovery parameter
    float lambda_se;      // Weight for semantic entropy
    float lambda_alpha;   // Weight for surprising edge fraction
    
    // Helper methods
    void update_adjacency_matrices();
    std::vector<float> compute_eigenvalues(const std::vector<std::vector<float>>& matrix);
    float calculate_entropy(const std::vector<float>& eigenvalues);
};

} // namespace llama

// C API for integration with llama.cpp
extern "C" {
    // Graph reasoning struct (opaque pointer)
    typedef struct llama_graph_reasoning_s llama_graph_reasoning;
    
    // Create/destroy
    llama_graph_reasoning* llama_graph_reasoning_init();
    void llama_graph_reasoning_free(llama_graph_reasoning* gr);
    
    // Extract graph from text
    bool llama_graph_extract(llama_graph_reasoning* gr, struct llama_context* ctx, 
                             const char* text, size_t text_len);
    
    // Calculate reward
    float llama_graph_compute_reward(llama_graph_reasoning* gr);
    
    // Set parameters
    void llama_graph_set_parameters(llama_graph_reasoning* gr, 
                                   float d_target, float alpha_target,
                                   float lambda_d, float lambda_se, float lambda_alpha);
}

#endif // GRAPH_REASONING_H
