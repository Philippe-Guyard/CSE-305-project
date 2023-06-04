#pragma once

#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>
#include "graph.hpp"

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

    /*
     * Same as make_random_graph, but ensures that the degree of any node
     * does not exceed the given maximum degree, d.
     */
    static Graph make_random_sparse_graph(int n = -1, double p = 0.5, int d = -1)
    {
        const int MAX_N = 1e5;
        if (n < 0)
            n = (int)(uniform(engine) * MAX_N);

        if (d < 0 || d > n)
            throw std::runtime_error("invalid degree");

        if (n > MAX_N)
            throw std::runtime_error("graph too big");

        Graph g(n);
        std::vector<int> degrees(n, 0);

        for (vertex_t v = 0; v < n; ++v)
        {
            for (vertex_t u = 0; u < n; ++u)
            {
                if (u == v || degrees[v] >= d || degrees[u] >= d)
                    continue;

                if (uniform(engine) < p)
                {
                    g.add_edge(u, v, uniform(engine));
                    degrees[v]++;
                    degrees[u]++;
                }
            }
        }

        return g;
    }

    static Graph make_random_dense_graph(int n = -1, int min_degree = -1, double p = 0.5, double p_dense = 0.1)
    {
        Graph g = make_random_graph(n, p);
        n = g.num_vertices();

        // Subset of vertices which should have high degree
        int dense_node_count = n / 10; // 10% of the nodes will be dense
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
