#include "graph_reasoning.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace llama {

GraphReasoning::GraphReasoning() {
    // Default parameters from paper
    d_target = -0.03f;      // Target critical discovery parameter
    alpha_target = 0.12f;   // Target surprising edge fraction
    lambda_d = 1.0f;        // Weight for discovery parameter
    lambda_se = 0.5f;       // Weight for semantic entropy 
    lambda_alpha = 0.5f;    // Weight for surprising edge fraction
}

GraphReasoning::~GraphReasoning() {
    // Clean up resources
}

int GraphReasoning::add_node(const std::string& content, const std::vector<float>& embedding) {
    int id = nodes.size();
    nodes.push_back({id, embedding, content});
    update_adjacency_matrices();
    return id;
}

void GraphReasoning::add_edge(int source, int target, float weight) {
    if (source >= nodes.size() || target >= nodes.size()) {
        return;
    }
    
    // Calculate semantic similarity
    float sim = 0.0f;
    const auto& src_emb = nodes[source].embedding;
    const auto& tgt_emb = nodes[target].embedding;
    
    if (!src_emb.empty() && !tgt_emb.empty()) {
        // Cosine similarity
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        
        for (size_t i = 0; i < src_emb.size() && i < tgt_emb.size(); i++) {
            dot += src_emb[i] * tgt_emb[i];
            norm1 += src_emb[i] * src_emb[i];
            norm2 += tgt_emb[i] * tgt_emb[i];
        }
        
        if (norm1 > 0 && norm2 > 0) {
            sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
        }
    }
    
    // Define surprising edge (semantically distant but structurally connected)
    bool is_surprising = sim < 0.1f;
    
    edges.push_back({source, target, weight, is_surprising});
    update_adjacency_matrices();
}

bool GraphReasoning::extract_from_text(llama_context* ctx, const char* text, size_t text_len) {
    // This is a placeholder for actual NLP extraction
    // Would need to parse text to identify concepts and relationships
    
    // Example logic:
    // 1. Extract sentences
    // 2. Identify entities/concepts 
    // 3. Determine relationships
    // 4. Create nodes for concepts with embeddings from ctx
    // 5. Create edges for relationships
    
    // For demonstration, just add some dummy nodes and edges
    std::vector<float> dummy_embedding(768, 0.0f);
    int node1 = add_node("Concept1", dummy_embedding);
    int node2 = add_node("Concept2", dummy_embedding);
    add_edge(node1, node2, 1.0f);
    
    return true;
}

void GraphReasoning::update_adjacency_matrices() {
    if (nodes.empty()) {
        return;
    }
    
    size_t n = nodes.size();
    
    // Resize and clear matrices
    adjacency_matrix.resize(n);
    semantic_adjacency_matrix.resize(n);
    
    for (size_t i = 0; i < n; i++) {
        adjacency_matrix[i].resize(n, 0.0f);
        semantic_adjacency_matrix[i].resize(n, 0.0f);
    }
    
    // Fill structural adjacency matrix
    for (const auto& edge : edges) {
        adjacency_matrix[edge.source][edge.target] = edge.weight;
        adjacency_matrix[edge.target][edge.source] = edge.weight; // Assuming undirected
    }
    
    // Fill semantic adjacency matrix based on embedding similarity
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            const auto& emb1 = nodes[i].embedding;
            const auto& emb2 = nodes[j].embedding;
            
            float sim = 0.0f;
            
            if (!emb1.empty() && !emb2.empty()) {
                // Cosine similarity
                float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
                
                for (size_t k = 0; k < emb1.size() && k < emb2.size(); k++) {
                    dot += emb1[k] * emb2[k];
                    norm1 += emb1[k] * emb1[k];
                    norm2 += emb2[k] * emb2[k];
                }
                
                if (norm1 > 0 && norm2 > 0) {
                    sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
                }
            }
            
            // Convert from [-1,1] to [0,1]
            sim = (sim + 1.0f) / 2.0f;
            
            semantic_adjacency_matrix[i][j] = sim;
            semantic_adjacency_matrix[j][i] = sim;
        }
    }
}

std::vector<float> GraphReasoning::compute_eigenvalues(const std::vector<std::vector<float>>& matrix) {
    // This is a simplified placeholder
    // In a real implementation, use Eigen or other linear algebra library
    
    // For demonstration, return dummy values
    // In practice, compute eigenvalues of normalized Laplacian
    std::vector<float> eigenvalues(matrix.size(), 0.5f);
    return eigenvalues;
}

float GraphReasoning::calculate_entropy(const std::vector<float>& eigenvalues) {
    float entropy = 0.0f;
    float sum = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0f);
    
    if (sum <= 0) {
        return 0.0f;
    }
    
    for (float val : eigenvalues) {
        if (val > 1e-10f) {
            float norm_val = val / sum;
            entropy -= norm_val * std::log(norm_val);
        }
    }
    
    return entropy;
}

float GraphReasoning::compute_structural_entropy() {
    std::vector<float> eigenvalues = compute_eigenvalues(adjacency_matrix);
    return calculate_entropy(eigenvalues);
}

float GraphReasoning::compute_semantic_entropy() {
    std::vector<float> eigenvalues = compute_eigenvalues(semantic_adjacency_matrix);
    return calculate_entropy(eigenvalues);
}

float GraphReasoning::compute_critical_discovery_parameter() {
    float s_struct = compute_structural_entropy();
    float s_sem = compute_semantic_entropy();
    
    if (s_struct + s_sem < 1e-10f) {
        return 0.0f;
    }
    
    return (s_struct - s_sem) / (s_struct + s_sem);
}

float GraphReasoning::compute_surprising_edge_fraction() {
    if (edges.empty()) {
        return 0.0f;
    }
    
    int count = 0;
    for (const auto& edge : edges) {
        if (edge.is_surprising) {
            count++;
        }
    }
    
    return static_cast<float>(count) / edges.size();
}

float GraphReasoning::compute_reward() {
    float d = compute_critical_discovery_parameter();
    float s_sem = compute_semantic_entropy();
    float alpha = compute_surprising_edge_fraction();
    
    float d_reward = -lambda_d * std::pow(d - d_target, 2);
    float sem_reward = lambda_se * s_sem;
    float alpha_reward = lambda_alpha * (1.0f - std::abs(alpha - alpha_target));
    
    return d_reward + sem_reward + alpha_reward;
}

void GraphReasoning::set_parameters(float d_target, float alpha_target, 
                                   float lambda_d, float lambda_se, float lambda_alpha) {
    this->d_target = d_target;
    this->alpha_target = alpha_target;
    this->lambda_d = lambda_d;
    this->lambda_se = lambda_se;
    this->lambda_alpha = lambda_alpha;
}

} // namespace llama

// C API implementation
extern "C" {

struct llama_graph_reasoning_s {
    llama::GraphReasoning impl;
};

llama_graph_reasoning* llama_graph_reasoning_init() {
    return new llama_graph_reasoning_s();
}

void llama_graph_reasoning_free(llama_graph_reasoning* gr) {
    delete gr;
}

bool llama_graph_extract(llama_graph_reasoning* gr, struct llama_context* ctx, 
                         const char* text, size_t text_len) {
    return gr->impl.extract_from_text(ctx, text, text_len);
}

float llama_graph_compute_reward(llama_graph_reasoning* gr) {
    return gr->impl.compute_reward();
}

void llama_graph_set_parameters(llama_graph_reasoning* gr, 
                               float d_target, float alpha_target,
                               float lambda_d, float lambda_se, float lambda_alpha) {
    gr->impl.set_parameters(d_target, alpha_target, lambda_d, lambda_se, lambda_alpha);
}

}
