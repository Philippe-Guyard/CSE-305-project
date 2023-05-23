#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>

#define N 5
#define INF INT_MAX
#define startNode 0

int printSolution(std::vector<int> distances)
{
    printf("Vertex Distance from Source\n");
    for (int i = 0; i < N; i++)
        printf("%d \t\t %d\n", i, distances[i]);
}

void dijkstra(int graph[N][N])
{
    // step 1: Initialization

    //  Set distances of all nodes to infinity except of start node to 0
    std::vector<int> distances(N, INT_MAX);
    distances[startNode] = 0;

    //  Set all nodes to unvisited
    std::vector<bool> visited(N, false);

    // step 2: Find the smallest distances from current node
    for (int i = 0; i < N; i++)
    {

        //  Find the node with the smallest distance
        int currentNode = -1;
        for (int j = 0; j < N; j++)
        {
            if (!visited[j] && (currentNode == -1 || distances[j] < distances[currentNode]))
            {
                currentNode = j;
            }
        }

        //  Mark the current node as visited
        visited[currentNode] = true;

        //  step 3: Update the distances of the neighbors of the current node
        for (int j = 0; j < N; j++)
        {
            if (graph[currentNode][j] != 0 && distances[currentNode] + graph[currentNode][j] < distances[j])
            {
                distances[j] = distances[currentNode] + graph[currentNode][j];
            }
        }
    }
    // printf("Distance from %d to %d is %d\n", startNode, targetNode, distances[targetNode]);
    printSolution(distances);
    return;
}

int main()
{
    int graph[N][N] = {
        {0, 1, 0, 0, 0},
        {1, 0, 2, 0, 0},
        {0, 2, 0, 3, 0},
        {0, 0, 3, 0, 4},
        {0, 0, 0, 4, 0}};

    dijkstra(graph);
    return 0;
}