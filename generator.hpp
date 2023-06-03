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

    // Random uniform sparse graphs, i.e same as above but the maximum degree of a node (= # of outgoing edges) is <= a given d
    static Graph make_random_sparse_graph(int n = -1, int max_degree = -1, double p = 0.5)
    {
        if (n == -1 || max_degree == -1)
            throw std::invalid_argument("Invalid input parameters");

        Graph g(n);

        std::vector<vertex_t> nodes(n);
        std::iota(nodes.begin(), nodes.end(), 0); // Fill the nodes vector with values from 0 to n-1

        for (vertex_t v = 0; v < n; ++v)
        {
            std::shuffle(nodes.begin(), nodes.end(), engine); // Shuffle the nodes
            for (vertex_t u : nodes)
            {
                if (v != u && !g.has_edge(v, u) && g.edges_from(v).size() < max_degree && g.edges_from(u).size() < max_degree)
                {
                    g.add_edge(v, u, uniform(engine));
                }
                if (g.edges_from(v).size() >= max_degree)
                    break;
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
};

std::default_random_engine GraphGenerator::engine;
std::uniform_real_distribution<double> GraphGenerator::uniform(0, 1);

//  Random city graphs
