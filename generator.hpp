#pragma once

#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>
#include "graph.hpp"



    class UnionFind
    {
        std::vector<int> parent;
        std::vector<int> rank;

    public:
        UnionFind(int n) : parent(n), rank(n, 0)
        {
            for (int i = 0; i < n; ++i)
                parent[i] = i;
        }

        int find(int x)
        {
            if (parent[x] != x)
                parent[x] = find(parent[x]);
            return parent[x];
        }

        void unite(int x, int y)
        {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY)
            {
                if (rank[rootX] < rank[rootY])
                {
                    parent[rootX] = rootY;
                }
                else if (rank[rootX] > rank[rootY])
                {
                    parent[rootY] = rootX;
                }
                else
                {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }

        bool same(int x, int y)
        {
            return find(x) == find(y);
        }
    };

    
class GraphGenerator
{
private:
    static std::default_random_engine engine;
    static std::uniform_real_distribution<double> uniform;

    static bool is_connected(const Graph &g)
    {
        std::vector<bool> visited(g.num_vertices(), false);
        std::stack<vertex_t> stack;
        stack.push(0);

        while (!stack.empty())
        {
            vertex_t v = stack.top();
            visited[v] = true;
            stack.pop();

            for (auto &e : g.edges_from(v))
            {
                if (!visited[e.to])
                {
                    visited[e.to] = true;
                    stack.push(e.to);
                }
            }
        }

        for (bool v : visited)
        {
            if (!v)
            {
                return false;
            }
        }
        return true;
    }

public:
    /*
     * Generates a random graph with n vertices.
     * The weights of the edges are generated randomly as numbers between 0 and 1.
     * Every edge has probability p of appearing in the graph.
     */
    static Graph make_random_graph(int n = -1, double p = 0.5)
    {
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

                if (uniform(engine) < p)
                    g.add_edge(u, v, uniform(engine));
            }
        }

        return g;
    }

    /*
     * Same as make_random_graph, but ensures that the graph is connected.
     */
    static Graph make_random_connected_graph(int n = -1, double p = 0.5)
    {
        Graph g = make_random_graph(n, p);
        n = g.num_vertices();
        // One edge every for every 10 000 possible
        size_t edges_per_iter = (n * n) / 10000 + 1;

        // Avoid checking connectedness too often, as it can be expensive
        while (!is_connected(g))
        {
            // Add random edges to g until it is connected
            size_t edges_added = 0;
            while (edges_added < edges_per_iter)
            {
                vertex_t v = (vertex_t)(uniform(engine) * n);
                vertex_t u = (vertex_t)(uniform(engine) * n);
                if (v == u || g.has_edge(v, u))
                {
                    continue;
                }

                g.add_edge(v, u, uniform(engine));
            }
        }

        return g;
    }


    static Graph make_random_sparse_graph(int n = -1, int max_degree = -1, double p = 0.2, double p_sparse = 0.03)
    {
        const int MAX_N = 1e5;
        if (n < 0)
            n = (int)(uniform(engine) * MAX_N);

        if (max_degree < 0 || max_degree > n)
            throw std::runtime_error("invalid degree");

        if (n > MAX_N)
            throw std::runtime_error("graph too big");

        int sparse_count = (int)(n * p_sparse);
        std::unordered_set<int> sparse_nodes;

        while (sparse_nodes.size() < sparse_count)
        {
            int rand_node = (int)(uniform(engine) * n);
            sparse_nodes.insert(rand_node);
        }

        Graph g(n);
        std::vector<int> degrees(n, 0);
        UnionFind uf(n);

        for (vertex_t v = 0; v < n; ++v)
        {
            for (vertex_t u = 0; u < n; ++u)
            {
                if (u == v || degrees[v] >= max_degree || degrees[u] >= max_degree)
                    continue;

                bool is_sparse_node = (sparse_nodes.find(v) != sparse_nodes.end()) || (sparse_nodes.find(u) != sparse_nodes.end());
                int sparse_degree = is_sparse_node ? max_degree / 2 : max_degree;

                if (uniform(engine) < p && degrees[v] < sparse_degree && degrees[u] < sparse_degree)
                {
                    g.add_edge(u, v, uniform(engine));
                    degrees[v]++;
                    degrees[u]++;
                    uf.unite(v, u);
                }
            }
        }

        // Add edges until the graph is connected
        while (uf.find(0) != uf.find(n - 1))
        {
            vertex_t v = (vertex_t)(uniform(engine) * n);
            vertex_t u = (vertex_t)(uniform(engine) * n);
            if (v == u || g.has_edge(v, u) || degrees[v] >= max_degree || degrees[u] >= max_degree)
                continue;

            g.add_edge(v, u, uniform(engine));
            degrees[v]++;
            degrees[u]++;
            uf.unite(v, u);
        }

        return g;
    }

    /*
     * Same as make_random_graph, but ensures that the degree of any node
     * does not exceed the given maximum degree, d.
     */
    // static Graph make_random_sparse_graph(int n = -1, int max_degree = -1, double p = 0.2, double p_sparse = 0.03)
    // {
    //     const int MAX_N = 1e5;
    //     if (n < 0)
    //         n = (int)(uniform(engine) * MAX_N);

    //     if (max_degree < 0 || max_degree > n)
    //         throw std::runtime_error("invalid degree");

    //     if (n > MAX_N)
    //         throw std::runtime_error("graph too big");

    //     int sparse_count = (int)(n * p_sparse);
    //     std::unordered_set<int> sparse_nodes;

    //     while (sparse_nodes.size() < sparse_count)
    //     {
    //         int rand_node = (int)(uniform(engine) * n);
    //         sparse_nodes.insert(rand_node);
    //     }

    //     Graph g(n);
    //     std::vector<int> degrees(n, 0);

    //     for (vertex_t v = 0; v < n; ++v)
    //     {
    //         for (vertex_t u = 0; u < n; ++u)
    //         {
    //             if (u == v || degrees[v] >= max_degree || degrees[u] >= max_degree)
    //                 continue;

    //             bool is_sparse_node = (sparse_nodes.find(v) != sparse_nodes.end()) || (sparse_nodes.find(u) != sparse_nodes.end());
    //             int sparse_degree = is_sparse_node ? max_degree / 2 : max_degree;

    //             if (uniform(engine) < p && degrees[v] < sparse_degree && degrees[u] < sparse_degree)
    //             {
    //                 g.add_edge(u, v, uniform(engine));
    //                 degrees[v]++;
    //                 degrees[u]++;
    //             }
    //         }
    //     }

    //     // Make sure the graph is connected
    //     while (!is_connected(g))
    //     {
    //         // Add random edges to g until it is connected
    //         size_t edges_added = 0;
    //         const size_t edges_per_iter = 10; // Or any number you think is reasonable

    //         while (edges_added < edges_per_iter)
    //         {
    //             vertex_t v = (vertex_t)(uniform(engine) * n);
    //             vertex_t u = (vertex_t)(uniform(engine) * n);
    //             if (v == u || g.has_edge(v, u) || degrees[v] >= max_degree || degrees[u] >= max_degree)
    //             {
    //                 continue;
    //             }

    //             g.add_edge(v, u, uniform(engine));
    //             degrees[v]++;
    //             degrees[u]++;
    //             edges_added++;
    //         }
    //     }

    //     return g;
    // }

    static Graph make_random_dense_graph(int n = -1, int min_degree = -1, double p = 0.5, double p_dense = 0.01)
    {
        Graph g = make_random_graph(n, p);
        n = g.num_vertices();

        // Subset of vertices which should have high degree
        int dense_node_count = (int)(n * p_dense); // 1% of the nodes will be dense
        std::vector<vertex_t> dense_nodes(dense_node_count);

        for (int i = 0; i < dense_node_count; i++)
        {
            dense_nodes[i] = i;
        }

        for (vertex_t v : dense_nodes)
        {
            while (g.edges_from(v).size() < min_degree)
            {
                // Add edges from v to other random vertices until its degree is 'min_degree'
                while (true)
                {
                    vertex_t u = (vertex_t)(uniform(engine) * n);
                    if (v != u && !g.has_edge(v, u))
                    {
                        g.add_edge(v, u, uniform(engine));
                        break;
                    }
                }
            }
        }

        // Now, we have a dense graph but it may not be connected. Let's connect it using a similar method as before.
        // One edge every for every 10 000 possible
        size_t edges_per_iter = (n * n) / 10000 + 1;

        // Avoid checking connectedness too often, as it can be expensive
        while (!is_connected(g))
        {
            // Add random edges to g until it is connected
            size_t edges_added = 0;
            while (edges_added < edges_per_iter)
            {
                vertex_t v = (vertex_t)(uniform(engine) * n);
                vertex_t u = (vertex_t)(uniform(engine) * n);
                if (v == u || g.has_edge(v, u))
                {
                    continue;
                }

                g.add_edge(v, u, uniform(engine));
                edges_added++;
            }
        }

        return g;
    }

    /*
     * Generates a random city graph with n^2 vertices.
     * The city is represented as a grid, where each node represents an intersection, and edges represent streets.
     */
    // static Graph make_random_city_graph(int n = -1)
    // {
    //     Graph g(n * n);

    //     // Connect each node with its neighbors to the right and below
    //     for (int i = 0; i < n; ++i)
    //     {
    //         for (int j = 0; j < n; ++j)
    //         {
    //             vertex_t v = i * n + j;

    //             if (j < n - 1) // Connect with the node to the right
    //             {
    //                 vertex_t u = v + 1;
    //                 g.add_edge(v, u, uniform(engine));
    //                 g.add_edge(u, v, uniform(engine));
    //             }

    //             if (i < n - 1) // Connect with the node below
    //             {
    //                 vertex_t u = v + n;
    //                 g.add_edge(v, u, uniform(engine));
    //                 g.add_edge(u, v, uniform(engine));
    //             }
    //         }
    //     }

    //     return g;
    // }
};

std::default_random_engine GraphGenerator::engine;
std::uniform_real_distribution<double> GraphGenerator::uniform(0, 1);

//  Random city graphs
