// src/graph_reasoning.h
#ifndef GRAPH_REASONING_H
#define GRAPH_REASONING_H

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "llama.h"

namespace llama {

struct GraphNode {
    int id;
    std::string content;
    Eigen::VectorXf embedding;
};

struct GraphEdge {
    int source_id;
    int target_id;
    float weight;
    bool is_surprising; // Semantically distant but structurally connected
};

class ReasoningGraph {
public:
    ReasoningGraph();
    
    // Graph construction methods
    int add_node(const std::string& content, const Eigen::VectorXf& embedding);
    void add_edge(int source_id, int target_id, float weight = 1.0f);
    
    // Entropy calculations
    float compute_structural_entropy() const; // Von Neumann graph entropy
    float compute_semantic_entropy() const;   // Based on embedding similarity
    float compute_critical_discovery_parameter() const; // D = (Sstruct - Ssem) / (Sstruct + Ssem)
    float compute_surprising_edge_fraction() const;    // α
    
    // Utilities
    void update_node_embeddings(llama_context* ctx);
    std::vector<int> detect_communities() const; // Louvain method
    
private:
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;
    Eigen::MatrixXf adjacency_matrix;
    Eigen::MatrixXf semantic_adjacency_matrix;
    
    // Helper methods
    void update_adjacency_matrices();
    Eigen::MatrixXf compute_normalized_laplacian(const Eigen::MatrixXf& adj_matrix) const;
};

// Reinforcement learning for graph reasoning
class GraphReasoningRL {
public:
    GraphReasoningRL(ReasoningGraph* graph, llama_context* ctx);
    
    // RL parameters
    void set_target_parameters(float d_target, float alpha_target);
    void set_lambda_weights(float lambda_d, float lambda_se, float lambda_alpha);
    
    // Compute reward based on graph state
    float compute_reward();
    
    // Update model weights based on reward and actions
    void update_model(llama_context* ctx, const std::vector<llama_token>& tokens, float reward);
    
private:
    ReasoningGraph* graph;
    llama_context* ctx;
    
    // Target parameters
    float d_target;
    float alpha_target;
    
    // Lambda weights for different reward components
    float lambda_d;
    float lambda_se;
    float lambda_alpha;
};

} // namespace llama

#endif // GRAPH_REASONING_H
