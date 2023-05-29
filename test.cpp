#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>

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

Graph make_connected_enough_graph(int n)
{
    // create a graph with n vertices
    Graph g(n);

    // Make sure there is a path from vertex 0 to every other vertex
    for (vertex_t v = 1; v < n; ++v)
    {
        g.add_edge(0, v, uniform(engine));
    }

    // Add additional random edges to the graph
    const int MAX_EDGES = n * (n - 1) / 2; // Maximum possible edges in a complete graph
    std::uniform_int_distribution<int> random_vertex(1, n - 1);

    // Add edges until we have enough
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
    // print the graph

    return g;
}