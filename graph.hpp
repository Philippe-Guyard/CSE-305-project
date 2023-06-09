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

struct Edge
{
    vertex_t to;
    edge_weight weight;
};

struct UEdge
{
    vertex_t from, to;

    UEdge(vertex_t u, vertex_t v) : from(u), to(v) {}

    bool operator==(const UEdge &other) const
    {
        return from == other.from && to == other.to;
    }
};

struct DEdge
{
    vertex_t from, to;
    edge_weight weight;

    DEdge(vertex_t f, vertex_t t, edge_weight w) : from(f), to(t), weight(w) {}

    bool operator==(const DEdge &other) const
    {
        return from == other.from && to == other.to && weight == other.weight;
    }

    bool operator<(const DEdge &other) const
    {
        return from < other.from || to < other.to || weight < other.weight;
    }
};

struct TEdge
{
    vertex_t from, to;
    edge_weight weight;
    int b;

    bool operator==(const TEdge &other) const
    {
        return from == other.from && to == other.to && weight == other.weight && b == other.b;
    }
};

namespace std
{
    template <>
    struct hash<UEdge>
    {
        std::size_t operator()(const UEdge &edge) const
        {
            std::size_t h1 = std::hash<vertex_t>()(edge.from);
            std::size_t h2 = std::hash<vertex_t>()(edge.to);
            return h1 ^ h2;
        }
    };
    template <>
    struct hash<DEdge>
    {
        std::size_t operator()(const DEdge &edge) const
        {
            std::size_t h1 = std::hash<vertex_t>()(edge.from);
            std::size_t h2 = std::hash<vertex_t>()(edge.to);
            std::size_t h3 = std::hash<edge_weight>()(edge.weight);
            return h1 ^ h2 ^ h3;
        }
    };
    template <>
    struct hash<TEdge>
    {
        std::size_t operator()(const TEdge &edge) const
        {
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
class HashArray
{
private:
    std::unordered_map<Key, Value> dict;
    Value defaultValue;

public:
    HashArray(const Value &defaultValue)
        : defaultValue(defaultValue), dict({}) {}

    void insert(const Key &key, const Value &value)
    {
        dict[key] = value;
    }

    Value get(const Key &key) const
    {
        auto it = dict.find(key);
        if (it != dict.end())
        {
            return it->second;
        }
        return defaultValue;
    }

    void clear()
    {
        dict.clear();
    }

    auto begin() const
    {
        return dict.begin();
    }

    auto end() const
    {
        return dict.end();
    }
};

class Graph
{
private:
    std::vector<std::vector<Edge>> adjacency_lists;
    // std::vector<std::unordered_set<vertex_t>> vertex_adjancency;

    // TODO: Normally, I think we should work directly with blocked_adjacency_lists
    // but since Philippe wrote this code and I dont want to break anything, I will
    // define a separate attribute and explicitely convert one to the other.
    // Philippe can then integrate it directly if he wants to.

    // # of vertices, # of edges
    // size_t n, m;

    void check_edge(vertex_t from, vertex_t to, edge_weight w = 0) const
    {
        if (from < 0 || from >= n || to < 0 || to >= n || w < 0)
        {
            throw std::runtime_error("Invalid edge: " + std::to_string(from) + " " + std::to_string(to) + " " + std::to_string(w));
        }
    }

public:
    std::vector<std::unordered_set<vertex_t>> vertex_adjancency;
    size_t n, m;
    Graph(size_t num_vertices = 0) : n(num_vertices), adjacency_lists(num_vertices), vertex_adjancency(num_vertices), m(0) {}

    bool has_edge(vertex_t from, vertex_t to) const
    {
        check_edge(from, to);
        return vertex_adjancency[from].find(to) != vertex_adjancency[from].end();
    }

    void add_edge(vertex_t from, vertex_t to, edge_weight weight)
    {
        check_edge(from, to, weight);
        if (has_edge(from, to))
        {
            throw std::runtime_error("Edge already exists");
        }
        adjacency_lists[from].push_back({to, weight});
        vertex_adjancency[from].insert(to);
        ++m;
    }

    size_t num_vertices() const
    {
        return n;
    }

    size_t num_edges() const
    {
        return m;
    }

    const std::vector<Edge> &edges_from(vertex_t v) const
    {
        return adjacency_lists[v];
    }

    friend std::ostream &operator<<(std::ostream &os, const Graph &g)
    {
        os << g.n << " " << g.m << std::endl;
        for (vertex_t v = 0; v < g.n; ++v)
        {
            for (auto &e : g.edges_from(v))
            {
                os << v << " " << e.to << " " << e.weight << std::endl;
            }
        }
        return os;
    }

    friend std::istream &operator>>(std::istream &in, Graph &g)
    {
        int new_n, new_m;
        in >> new_n >> new_m;

        g.n = new_n;
        g.m = 0;
        g.adjacency_lists.clear();
        g.adjacency_lists.resize(g.n);

        vertex_t src, dest;
        edge_weight w;
        for (int i = 0; i < new_m; ++i)
        {
            in >> src >> dest >> w;
            std::cout << g.n << std::endl;
            std::cout << src << " " << dest << " " << w << std::endl;
            g.add_edge(src, dest, w);
        }

        if (g.m != new_m)
            throw std::runtime_error("Cannot read properly");

        return in;
    }

    edge_weight get_delta_0() const
    {
        edge_weight delta_0 = std::numeric_limits<edge_weight>::infinity();

        // can be made parallel
        for (vertex_t v = 0; v < n; ++v)
        {
            for (auto &e : edges_from(v))
            {
                if (e.weight < delta_0)
                {
                    delta_0 = e.weight;
                }
            }
        }
        return delta_0;
    }

    std::vector<std::vector<edge_block>> gen_blocked_adjacency_lists() const
    {
        edge_weight delta_0 = get_delta_0();

        // now for each vertex v, we want to sort the edges coming out of v
        // into blocks. A block j contains all edges e such that
        // delta_0 * 2^j <= e.weight < delta_0 * 2^(j+1)

        std::vector<std::vector<edge_block>> blocked_adjacency_lists(n);
        for (vertex_t v = 0; v < n; ++v)
        {
            for (auto &e : edges_from(v))
            {
                edge_weight lower_bound = delta_0;
                int block_index = 0;

                while (e.weight >= lower_bound * 2)
                {
                    lower_bound *= 2;
                    block_index++;
                }

                // If there is not enough space, we should extend the vector and potentially add some empty blocks
                while (blocked_adjacency_lists[v].size() <= block_index)
                {
                    edge_block new_block({});
                    blocked_adjacency_lists[v].push_back(new_block);
                }
                blocked_adjacency_lists[v][block_index].push_back(DEdge({v, e.to, e.weight}));
            }
        }

        return blocked_adjacency_lists;
    };

    std::vector<std::pair<std::vector<Edge>, std::vector<Edge>>> semisort_delta_parallel(double delta) const {
        // For every vertex, a pair of vectors:
        // All edges with weight < delta
        // All edges with weight >= delta
        std::vector<std::pair<std::vector<Edge>, std::vector<Edge>>> semisorted_edges(n);
        #pragma omp parallel for schedule(dynamic)
        for(vertex_t v = 0; v < n; ++v) {
            for(auto &e : edges_from(v)) {
                if(e.weight < delta) 
                    semisorted_edges[v].first.push_back(e);
                else 
                    semisorted_edges[v].second.push_back(e);
            }
        }

        return semisorted_edges;
    }
};
