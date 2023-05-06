#pragma once 
#include <vector>

using vertex_t = int;
using edge_weight = double;

// Standard definition of a directed graph with non-negative real edge weights 
// Store edges as adjacency lists
struct Edge {
    vertex_t to;
    edge_weight weight;
};

class Graph {
private:
    std::vector<std::vector<Edge>> adjacency_lists;
public: 
    Graph(int num_vertices) : adjacency_lists(num_vertices) {}

    void add_edge(vertex_t from, vertex_t to, edge_weight weight) {
        adjacency_lists[from].push_back({to, weight});
    }

    int num_vertices() const {
        return adjacency_lists.size();
    }

    const std::vector<Edge>& edges_from(vertex_t v) const {
        return adjacency_lists[v];
    }
};