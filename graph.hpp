#pragma once
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stack>

using vertex_t = int;
using edge_weight = double;

struct Edge {
    vertex_t to;
    edge_weight weight;
};

struct UEdge {
    vertex_t from, to;

    bool operator==(const UEdge &other) const {
        return from == other.from && to == other.to;
    }
};

struct DEdge {
    vertex_t from, to;
    edge_weight weight;

    bool operator==(const DEdge &other) const {
        return from == other.from && to == other.to && weight == other.weight;
    }

    bool operator<(const DEdge &other) const {
        return from <= other.from || to <= other.to;
    }
};

struct TEdge {
    vertex_t from, to;
    edge_weight weight;
    int b;
    
    bool operator==(const TEdge &other) const {
        return from == other.from && to == other.to && weight == other.weight && b == other.b;
    }
};

namespace std {
    template <>
    struct hash<UEdge> {
        std::size_t operator()(const UEdge &edge) const {
            std::size_t h1 = std::hash<vertex_t>()(edge.from);
            std::size_t h2 = std::hash<vertex_t>()(edge.to);
            return h1 ^ h2;
        }
    };
    template <>
    struct hash<DEdge> {
        std::size_t operator()(const DEdge &edge) const {
            std::size_t h1 = std::hash<vertex_t>()(edge.from);
            std::size_t h2 = std::hash<vertex_t>()(edge.to);
            std::size_t h3 = std::hash<edge_weight>()(edge.weight);
            return h1 ^ h2 ^ h3;
        }
    };
    template <>
    struct hash<TEdge> {
        std::size_t operator()(const TEdge &edge) const {
            std::size_t h1 = std::hash<vertex_t>()(edge.from);
            std::size_t h2 = std::hash<vertex_t>()(edge.to);
            std::size_t h3 = std::hash<edge_weight>()(edge.weight);
            std::size_t h4 = std::hash<int>()(edge.b);
            return h1 ^ h2 ^ h3 ^ h4;
        }
    };
}

using edge_block = std::vector<DEdge>;

// Standard definition of a directed graph with non-negative real edge weights
// Store edges as adjacency lists

template <typename Key, typename Value>
class HashArray {
private:
    std::unordered_map<Key, Value> dict;
    Value defaultValue;

public:
    HashArray(const Value &defaultValue)
        : defaultValue(defaultValue), dict({}) {}

    void insert(const Key &key, const Value &value) {
        dict[key] = value;
    }

    Value get(const Key &key) const {
        auto it = dict.find(key);
        if (it != dict.end()) {
            return it->second;
        }
        return defaultValue;
    }
};

class Graph {
private:
    std::vector<std::vector<Edge>> adjacency_lists;
    std::vector<std::unordered_set<vertex_t>> vertex_adjancency;

    // TODO: Normally, I think we should work directly with blocked_adjacency_lists
    // but since Philippe wrote this code and I dont want to break anything, I will
    // define a separate attribute and explicitely convert one to the other.
    // Philippe can then integrate it directly if he wants to.

    // # of vertices, # of edges
    size_t n, m;

    void check_edge(vertex_t from, vertex_t to, edge_weight w = 0) const {
        if (from < 0 || from >= n || to < 0 || to >= n || w < 0) {
            throw std::runtime_error("Invalid edge: " + std::to_string(from) + " " + std::to_string(to) + " " + std::to_string(w));
        }
    }
public:
    Graph(size_t num_vertices = 0) : n(num_vertices), adjacency_lists(num_vertices), vertex_adjancency(num_vertices), m(0) {}

    bool has_edge(vertex_t from, vertex_t to) const {
        check_edge(from, to);
        return vertex_adjancency[from].find(to) != vertex_adjancency[from].end();
    }

    void add_edge(vertex_t from, vertex_t to, edge_weight weight) {
        check_edge(from, to, weight);
        if (has_edge(from, to)) {
            throw std::runtime_error("Edge already exists");
        }
        adjacency_lists[from].push_back({to, weight});
        vertex_adjancency[from].insert(to);
        ++m;
    }

    size_t num_vertices() const {
        return n;
    }

    size_t num_edges() const {
        return m;
    }

    const std::vector<Edge> &edges_from(vertex_t v) const {
        return adjacency_lists[v];
    }

    friend std::ostream &operator<<(std::ostream &os, const Graph &g) {
        os << g.n << " " << g.m << std::endl;
        for (vertex_t v = 0; v < g.n; ++v) {
            for (auto &e : g.edges_from(v)) {
                os << v << " " << e.to << " " << e.weight << std::endl;
            }
        }
        return os;
    }

    friend std::istream &operator>>(std::istream &in, Graph &g) {
        int new_n, new_m;
        in >> new_n >> new_m;

        g.n = new_n;
        g.m = 0;
        g.adjacency_lists.clear();
        g.adjacency_lists.resize(g.n);

        vertex_t src, dest;
        edge_weight w;
        for (int i = 0; i < new_m; ++i) {
            in >> src >> dest >> w;
            std::cout << g.n << std::endl;
            std::cout << src << " " << dest << " " << w << std::endl;
            g.add_edge(src, dest, w);
        }

        if (g.m != new_m)
            throw std::runtime_error("Cannot read properly");

        return in;
    }

    edge_weight get_delta_0() const {
        edge_weight delta_0 = std::numeric_limits<edge_weight>::infinity();

        // can be made parallel
        for (vertex_t v = 0; v < n; ++v) {
            for (auto &e : edges_from(v)) {
                if (e.weight < delta_0) {
                    delta_0 = e.weight;
                }
            }
        }
        return delta_0;
    }

    std::vector<std::vector<edge_block>> gen_blocked_adjacency_lists() const {
        edge_weight delta_0 = get_delta_0();

        // now for each vertex v, we want to sort the edges coming out of v
        // into blocks. A block j contains all edges e such that
        // delta_0 * 2^j <= e.weight < delta_0 * 2^(j+1)

        std::vector<std::vector<edge_block>> blocked_adjacency_lists(n);
        for (vertex_t v = 0; v < n; ++v) {
            for (auto &e : edges_from(v)) {
                edge_weight lower_bound = delta_0;
                int block_index = 0;

                while (e.weight >= lower_bound * 2) {
                    lower_bound *= 2;
                    block_index++;
                }

                // If there is not enough space, we should extend the vector and potentially add some empty blocks
                while (blocked_adjacency_lists[v].size() <= block_index) {
                    edge_block new_block({});
                    blocked_adjacency_lists[v].push_back(new_block);
                }
                blocked_adjacency_lists[v][block_index].push_back(DEdge({v, e.to, e.weight}));
            }
        }

        return blocked_adjacency_lists;
    };
};

class GraphGenerator {
private:
    static std::default_random_engine engine;
    static std::uniform_real_distribution<double> uniform;

    static bool is_connected(const Graph &g) {
        std::vector<bool> visited(g.num_vertices(), false);
        std::stack<vertex_t> stack;
        stack.push(0);

        while (!stack.empty()) {
            vertex_t v = stack.top();
            visited[v] = true;
            stack.pop();

            for (auto &e : g.edges_from(v)) {
                if (!visited[e.to]) {
                    visited[e.to] = true;
                    stack.push(e.to);
                }
            }
        }

        for (bool v : visited) {
            if (!v) {
                return false;
            }
        }
        return true;
    }
public:
    static Graph make_random_graph(int n = -1)
    {
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
        for (vertex_t v = 0; v < n; ++v)
        {
            for (vertex_t u = 0; u < n; ++u)
            {
                if (u == v)
                    continue;

                if (uniform(engine) > 0.5)
                    g.add_edge(u, v, uniform(engine));
            }
        }

        return g;
    }

    static Graph make_random_connected_graph(int n = -1) {
        Graph g = make_random_graph(n);
        const size_t EDGES_PER_ITER = 10;

        while (!is_connected(g)) {
            // Add 10 random edges to g until it is connected
            size_t edges_added = 0;
            while (edges_added < EDGES_PER_ITER) {
                vertex_t v = (vertex_t)(uniform(engine) * n);
                vertex_t u = (vertex_t)(uniform(engine) * n);
                if (v == u || g.has_edge(v, u)) {
                    continue;
                }

                g.add_edge(v, u, uniform(engine));
            }
        }

        return g;
    }
};

std::default_random_engine GraphGenerator::engine;
std::uniform_real_distribution<double> GraphGenerator::uniform(0, 1);