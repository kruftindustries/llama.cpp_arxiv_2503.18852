// src/graph_reasoning.cpp
#include "graph_reasoning.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <Eigen/Eigenvalues>

namespace llama {

ReasoningGraph::ReasoningGraph() {}

int ReasoningGraph::add_node(const std::string& content, const Eigen::VectorXf& embedding) {
    int new_id = nodes.size();
    nodes.push_back({new_id, content, embedding});
    update_adjacency_matrices();
    return new_id;
}

void ReasoningGraph::add_edge(int source_id, int target_id, float weight) {
    // Check if nodes exist
    if (source_id >= nodes.size() || target_id >= nodes.size()) {
        return;
    }
    
    // Calculate semantic similarity
    float sim = nodes[source_id].embedding.normalized().dot(nodes[target_id].embedding.normalized());
    bool is_surprising = sim < 0.1f; // Threshold for surprising edges
    
    edges.push_back({source_id, target_id, weight, is_surprising});
    update_adjacency_matrices();
}

float ReasoningGraph::compute_structural_entropy() const {
    Eigen::MatrixXf L = compute_normalized_laplacian(adjacency_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(L);
    Eigen::VectorXf eigenvalues = es.eigenvalues();
    
    // Normalize eigenvalues to sum to 1
    float sum = eigenvalues.sum();
    if (sum > 0) {
        eigenvalues /= sum;
    }
    
    // Calculate Von Neumann entropy: -∑λᵢlog(λᵢ)
    float entropy = 0.0f;
    for (int i = 0; i < eigenvalues.size(); i++) {
        float λ = eigenvalues[i];
        if (λ > 1e-10f) { // Avoid log(0)
            entropy -= λ * std::log(λ);
        }
    }
    
    return entropy;
}

float ReasoningGraph::compute_semantic_entropy() const {
    Eigen::MatrixXf L = compute_normalized_laplacian(semantic_adjacency_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(L);
    Eigen::VectorXf eigenvalues = es.eigenvalues();
    
    // Normalize eigenvalues to sum to 1
    float sum = eigenvalues.sum();
    if (sum > 0) {
        eigenvalues /= sum;
    }
    
    // Calculate entropy: -∑μᵢlog(μᵢ)
    float entropy = 0.0f;
    for (int i = 0; i < eigenvalues.size(); i++) {
        float μ = eigenvalues[i];
        if (μ > 1e-10f) { // Avoid log(0)
            entropy -= μ * std::log(μ);
        }
    }
    
    return entropy;
}

float ReasoningGraph::compute_critical_discovery_parameter() const {
    float s_struct = compute_structural_entropy();
    float s_sem = compute_semantic_entropy();
    
    if (s_struct + s_sem < 1e-10f) {
        return 0.0f;
    }
    
    return (s_struct - s_sem) / (s_struct + s_sem);
}

float ReasoningGraph::compute_surprising_edge_fraction() const {
    if (edges.empty()) {
        return 0.0f;
    }
    
    int surprising_count = 0;
    for (const auto& edge : edges) {
        if (edge.is_surprising) {
            surprising_count++;
        }
    }
    
    return static_cast<float>(surprising_count) / edges.size();
}

void ReasoningGraph::update_adjacency_matrices() {
    if (nodes.empty()) {
        return;
    }
    
    int n = nodes.size();
    adjacency_matrix = Eigen::MatrixXf::Zero(n, n);
    semantic_adjacency_matrix = Eigen::MatrixXf::Zero(n, n);
    
    // Fill structural adjacency matrix
    for (const auto& edge : edges) {
        adjacency_matrix(edge.source_id, edge.target_id) = edge.weight;
        adjacency_matrix(edge.target_id, edge.source_id) = edge.weight; // Assuming undirected graph
    }
    
    // Fill semantic adjacency matrix based on embedding similarity
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sim = nodes[i].embedding.normalized().dot(nodes[j].embedding.normalized());
            sim = (sim + 1.0f) / 2.0f; // Scale from [-1,1] to [0,1]
            semantic_adjacency_matrix(i, j) = sim;
            semantic_adjacency_matrix(j, i) = sim;
        }
    }
}

Eigen::MatrixXf ReasoningGraph::compute_normalized_laplacian(const Eigen::MatrixXf& adj_matrix) const {
    int n = adj_matrix.rows();
    Eigen::VectorXf degree = adj_matrix.rowwise().sum();
    Eigen::MatrixXf D_inv_sqrt = Eigen::MatrixXf::Zero(n, n);
    
    // Compute D^(-1/2)
    for (int i = 0; i < n; i++) {
        D_inv_sqrt(i, i) = degree[i] > 0 ? 1.0f / std::sqrt(degree[i]) : 0;
    }
    
    // L = I - D^(-1/2) * A * D^(-1/2)
    Eigen::MatrixXf L = Eigen::MatrixXf::Identity(n, n) - D_inv_sqrt * adj_matrix * D_inv_sqrt;
    return L;
}

// GraphReasoningRL implementation
GraphReasoningRL::GraphReasoningRL(ReasoningGraph* graph, llama_context* ctx) 
    : graph(graph), ctx(ctx), d_target(-0.03f), alpha_target(0.12f),
      lambda_d(1.0f), lambda_se(0.5f), lambda_alpha(0.5f) {}

void GraphReasoningRL::set_target_parameters(float d_target, float alpha_target) {
    this->d_target = d_target;
    this->alpha_target = alpha_target;
}

void GraphReasoningRL::set_lambda_weights(float lambda_d, float lambda_se, float lambda_alpha) {
    this->lambda_d = lambda_d;
    this->lambda_se = lambda_se;
    this->lambda_alpha = lambda_alpha;
}

float GraphReasoningRL::compute_reward() {
    float d = graph->compute_critical_discovery_parameter();
    float s_sem = graph->compute_semantic_entropy();
    float alpha = graph->compute_surprising_edge_fraction();
    
    // Reward components
    float d_reward = -lambda_d * std::pow(d - d_target, 2);
    float sem_reward = lambda_se * s_sem;
    float alpha_reward = lambda_alpha * (1.0f - std::abs(alpha - alpha_target));
    
    return d_reward + sem_reward + alpha_reward;
}

void GraphReasoningRL::update_model(llama_context* ctx, const std::vector<llama_token>& tokens, float reward) {
    // This would be implemented based on the RL gradient update equation
    // ∇θJ(θ) = E[Rt·∇θlogπθ(a|G)]
    
    // This is a placeholder - actual implementation would require modifying llama's weight update mechanism
    // to incorporate the policy gradient update
}

} // namespace llama
