#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include "graph.hpp"

// Function to perform Dijkstra's algorithm
void dijkstra(const Graph &graph, vertex_t source)
{
    int n = graph.num_vertices();

    // Initialize distances to infinity for all vertices except the source
    std::vector<double> distances(n, std::numeric_limits<double>::infinity());
    distances[source] = 0;

    // Priority queue to store vertices and their distances
    std::priority_queue<std::pair<double, vertex_t>, std::vector<std::pair<double, vertex_t>>, std::greater<std::pair<double, vertex_t>>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        vertex_t u = pq.top().second;
        pq.pop();

        // Iterate through all neighboring vertices of u
        for (const Edge &edge : graph.edges_from(u))
        {
            vertex_t v = edge.to;
            double weight = edge.weight;

            // Relax the edge (u, v)
            if (distances[u] + weight < distances[v])
            {
                distances[v] = distances[u] + weight;
                pq.push({distances[v], v});
            }
        }
    }

    return;
}

// this was to check if the graph was being generated correctly and if dijkstra was working
//  change the outpute type in dijkstra to vector<double> and uncomment the main function
//  int main()
//  {
//      // Create a sample graph using the Graph class
//      Graph graph(6);
//      graph.add_edge(0, 1, 2);
//      graph.add_edge(0, 2, 5);
//      graph.add_edge(1, 3, 4);
//      graph.add_edge(1, 4, 7);
//      graph.add_edge(2, 4, 1);
//      graph.add_edge(3, 5, 3);
//      graph.add_edge(4, 3, 2);
//      graph.add_edge(4, 5, 6);

//     vertex_t source = 0;

//     // Run Dijkstra's algorithm
//     vector<double> distances = dijkstra(graph, source);

//     // Print the distances from the source to all other vertices
//     for (vertex_t v = 0; v < graph.num_vertices(); ++v)
//     {
//         cout << "Distance from " << source << " to " << v << ": " << distances[v] << endl;
//     }

//     return 0;
// }
