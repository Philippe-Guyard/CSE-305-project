#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <random>

// Graph make_connected_enough_graph(int n)
// {
//     // create a graph with n vertices
//     Graph g(n);

//     // Make sure there is a path from vertex 0 to every other vertex
//     for (vertex_t v = 1; v < n; ++v)
//     {
//         g.add_edge(0, v, uniform(engine));
//     }

//     // Add additional random edges to the graph
//     const int MAX_EDGES = n * (n - 1) / 2; // Maximum possible edges in a complete graph
//     std::uniform_int_distribution<int> random_vertex(1, n - 1);

//     // Add edges until we have enough
//     for (vertex_t v = 0; v < n; ++v)
//     {
//         for (vertex_t u = 0; u < n; ++u)
//         {
//             if (u == v)
//                 continue;

//             if (uniform(engine) > 0.5)
//                 g.add_edge(u, v, uniform(engine));
//         }
//     }
//     // print the graph

//     return g;
// }