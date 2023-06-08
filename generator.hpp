#pragma once

#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>
#include "graph.hpp"
#include "UnionFind.hpp"

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
    static Graph make_random_graph(int n = -1, double p = 0.2)
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
    static Graph make_random_connected_graph(int n = -1, double p = 0.2)
    {
        Graph g = make_random_graph(n, p);
        n = g.num_vertices();
        // One edge every for every 10 000 possible
        size_t edges_per_iter = (n * n) / 10000 + 1;

        // Initialize Union-Find data structure
        UnionFind uf(n);

        // Add edges until the graph is connected
        while (uf.find(0) != uf.find(n - 1))
        {
            vertex_t v = (vertex_t)(uniform(engine) * n);
            vertex_t u = (vertex_t)(uniform(engine) * n);
            if (v == u || g.has_edge(v, u))
            {
                continue;
            }

            g.add_edge(v, u, uniform(engine));
            uf.unite(v, u);
        }

        return g;
    }

    /*
     * some porucentage of nodes is very poorly connected, i.e. has very little edges, up to max_degree (or a bit more for the sake of being connected)
     */
    static Graph make_random_sparse_graph(int n = -1, double p = 0.2, int max_degree = -1, double p_sparse = 0.03)
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
     * some porucentage of nodes is very well connected, i.e. has many edges, at least min_degree
     */
    static Graph make_random_dense_graph(int n = -1, double p = 0.5, int min_degree = -1, double p_dense = 0.01)
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

        UnionFind uf(n);

        // Now, we have a dense graph but it may not be connected. Let's connect it using a similar method as before.
        // One edge every for every 10 000 possible
        size_t edges_per_iter = (n * n) / 10000 + 1;

        while (uf.find(0) != uf.find(n - 1))
        {
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
                uf.unite(v, u);
            }
        }

        return g;
    }
};

std::default_random_engine GraphGenerator::engine;
std::uniform_real_distribution<double> GraphGenerator::uniform(0, 1);
