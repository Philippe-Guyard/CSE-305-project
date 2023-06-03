#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>
#include "graph.hpp"

static std::default_random_engine engine(31);
static std::uniform_real_distribution<double> uniform(0, 1);

Graph make_random_graph(int n = -1)
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

bool is_connected_weak(Graph g)
{
    int num_vertices = g.num_vertices();

    // Perform BFS starting from each vertex
    for (vertex_t v = 0; v < num_vertices; ++v)
    {
        // Create a set to keep track of visited vertices
        std::unordered_set<vertex_t> visited;

        // Create a queue for BFS traversal
        std::queue<vertex_t> queue;

        // Start BFS from current vertex
        queue.push(v);
        visited.insert(v);

        // Perform BFS
        while (!queue.empty())
        {
            vertex_t curr = queue.front();
            queue.pop();

            // Visit all adjacent vertices
            for (const Edge &e : g.edges_from(curr))
            {
                if (visited.find(e.to) == visited.end())
                {
                    visited.insert(e.to);
                    queue.push(e.to);
                }
            }
        }

        // Check if all vertices are rechable from vertex v
        if (visited.size() != num_vertices)
        {
            return false;
        }
    }

    return true;
}

Graph make_random_connected(size_t n)
{
    Graph g = make_random_graph(n);
    while (!is_connected_weak(g))
    {
        vertex_t u = (vertex_t)(uniform(engine) * n);
        vertex_t v = (vertex_t)(uniform(engine) * n);
        if (u == v)
            continue;
        if (u != v)
        {
            g.add_edge(u, v, uniform(engine));
        }
    }
}