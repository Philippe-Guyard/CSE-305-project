#pragma once 
#include <vector>
#include <random>

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
    // # of vertices, # of edges 
    int n, m;
public: 
    Graph(int num_vertices = 0) : n(num_vertices), adjacency_lists(num_vertices), m(0) {}

    void add_edge(vertex_t from, vertex_t to, edge_weight weight) {
        adjacency_lists[from].push_back({to, weight});
        ++m;
    }

    int num_vertices() const {
        return adjacency_lists.size();
    }

    const std::vector<Edge>& edges_from(vertex_t v) const {
        return adjacency_lists[v];
    }

    friend std::ostream& operator<<(std::ostream& os, const Graph& g) {
        os << g.n << " " << g.m << std::endl;
        for(vertex_t v = 0; v < g.n; ++v) {
            for(auto& e : g.edges_from(v)) {
                os << v << " " << e.to << " " << e.weight << std::endl;
            }
        }
        return os;
    }

    friend std::istream& operator>>(std::istream& in, Graph& g) {
        int new_n, new_m;
        in >> new_n >> new_m;

        g.n = new_n;
        g.m = 0;
        g.adjacency_lists.clear();
        g.adjacency_lists.resize(g.n);

        vertex_t src, dest;
        edge_weight w;
        for(int i = 0; i < new_m; ++i) {
            in >> src >> dest >> w;
            std::cout << g.n << std::endl;
            std::cout << src << " " << dest << " " << w << std::endl;
            g.add_edge(src, dest, w);
        }

        if (g.m != new_m) 
            throw std::runtime_error("Cannot read properly");

        return in;
    }
};

static std::default_random_engine engine(31); 
static std::uniform_real_distribution<double> uniform(0, 1);

Graph make_random_graph(int n = -1) {
    // Generates a completely random graph with exactly 
    // n vertices. Every edge has exactly probability 
    // 1/2 of appearing in the graph. For all edges in the graph,
    // weights are generated randomly as number between 0 and 1 
    const int MAX_N = 1e5;
    if (n < 0) 
        n = (int)(uniform(engine) * MAX_N);
    
    if (n > MAX_N) 
        throw std::runtime_error("graph too big");
    

    Graph g(n);
    for(vertex_t v = 0; v < n; ++v) {
        for(vertex_t u = 0; u < n; ++u) {
            if (u == v)
                continue;
            
            if (uniform(engine) > 0.5) 
                g.add_edge(u, v, uniform(engine));
        }
    }

    return g;
}