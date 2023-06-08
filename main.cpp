#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>
#include <memory>
#include <omp.h>

#include "buckets.hpp"
#include "graph.hpp"
#include "benchmarker.hpp"
#include "generator.hpp"
#include "DeltaSteppingSolver.hpp"

void run_tests(Graph g, double p)
{
    // Run the tests on the graph
    std::cout
        << "G has " << g.num_vertices() << " vertices and " << g.num_edges() << " edges";
    std::cout << "(random graph with p = " << p << ") " << std::endl;
    std::cout << std::endl;

    const double delta = 0.1;
    DeltaSteppingSolver solver(g, false);

    Benchmarker::start_one("Total");
    auto res = solver.solve(0, delta);
    Benchmarker::end_one("Total");
    std::cout << "Sequential (delta = " << delta << ") benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    Benchmarker::clear();

    size_t num_threads = 8;
    Benchmarker::start_one("Total");
    auto resPara = solver.solve_parallel_simple(0, delta, num_threads);
    Benchmarker::end_one("Total");
    std::cout << "Parallel (with " << num_threads << " threads) benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    Benchmarker::clear();

    Benchmarker::start_one("Total");
    auto resOmp = solver.solve_parallel_omp(0, delta, num_threads);
    Benchmarker::end_one("Total");
    std::cout << "OMP (with " << num_threads << " threads) benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    Benchmarker::clear();

    Benchmarker::start_one("Total");
    auto resShortcuts = solver.solve_shortcuts(0, delta);
    Benchmarker::end_one("Total");
    std::cout << "Sequential with shortcuts (delta = " << delta << ") benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    Benchmarker::clear();

    Benchmarker::start_one("Total");
    auto resShortcutsPara = solver.solve_shortcuts_parallel(0, delta, num_threads);
    Benchmarker::end_one("Total");
    std::cout << "Parallel with shortcuts (with " << num_threads << " threads) benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    Benchmarker::clear();

    Benchmarker::start_one("Total");
    auto resShortcutsOmp = solver.solve_shortcuts_omp(0, delta);
    Benchmarker::end_one("Total");
    std::cout << "OMP with shortcuts benchmarking summary:" << std::endl;
    Benchmarker::print_summary(std::cout);
    std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto resDijkstra = dijkstra(g, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Dijkstra: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for (size_t i = 0; i < res.size(); i++)
    {
        if (resDijkstra[i] != res[i])
            std::cout << "ERROR in simple solve: " << i << ": " << resDijkstra[i] << " != " << res[i] << std::endl;
        if (resDijkstra[i] != resPara[i])
            std::cout << "ERROR in parallel solve: " << i << ": " << resDijkstra[i] << " != " << resPara[i] << std::endl;
        if (resDijkstra[i] != resShortcuts[i])
            std::cout << "ERROR in shortcuts solve: " << i << ": " << resDijkstra[i] << " != " << resShortcuts[i] << std::endl;
        if (resDijkstra[i] != resShortcutsPara[i])
            std::cout << "ERROR in parallel shortcuts solve: " << i << ": " << resDijkstra[i] << " != " << resShortcutsPara[i] << std::endl;
        if (resDijkstra[i] != resOmp[i])
            std::cout << "ERROR in omp solve: " << i << ": " << resDijkstra[i] << " != " << resOmp[i] << std::endl;
        if (resDijkstra[i] != resShortcutsOmp[i])
            std::cout << "ERROR in omp shortcuts solve: " << i << ": " << resDijkstra[i] << " != " << resShortcutsOmp[i] << std::endl;
    }
}

int main()
{
    /*
           C -> D -> I
          /           \
    A -> B             G -> H
          \           /
           E -> F-----
    */

    // Graph g(9);
    // vertex_t A = 0;
    // vertex_t B = 1;
    // vertex_t C = 2;
    // vertex_t D = 3;
    // vertex_t E = 4;
    // vertex_t F = 5;
    // vertex_t G = 6;
    // vertex_t H = 7;
    // vertex_t I = 8;
    // g.add_edge(A, B, 2);
    // g.add_edge(B, C, 4);
    // g.add_edge(C, D, 5);
    // g.add_edge(D, I, 7);
    // g.add_edge(I, G, 8);
    // g.add_edge(B, E, 20);
    // g.add_edge(E, F, 3);
    // g.add_edge(F, G, 11);
    // g.add_edge(G, H, 15);

    // const size_t N = 30000;
    // const double p = 0.2;
    // const int max_degree = 0.3 * N;
    // const int min_degree = 0.8 * N;
    // const double p_dense = 0.01;
    // const double p_sparse = 0.03;
    // // choose between
    // // make_random_graph(N, p) tested
    // // make_random_connected_graph(N,  p) tested
    // // make_random_sparse_graph(N, p, max_degree,) tested
    // // make_random_dense_graph(N, min_degree ,  p,  p_dense) tested
    // Graph g = GraphGenerator::make_random_sparse_graph(N, max_degree, p, p_sparse);

    // List of graph sizes to test
    std::vector<int> graph_sizes = {25000};

    // Parameters for the graphs
    double p = 0.2;
    double p_dense = 0.01;  // probability that a node is a very connected node (i.e. 1% of nodes are dsuper connected)
    double p_sparse = 0.03; // probability that a node is poorly connected (i.e. 3% of nodes are poorly connected)

    // Run the tests for each graph size and each graph generation function
    for (size_t N : graph_sizes)
    {
        const int max_degree = 0.3 * N; // poorly connected nodes are connected with up to 30% of total nodes
        const int min_degree = 0.8 * N; // well connected nodes are connected with at least 80% of total nodes
        Graph g_random = GraphGenerator::make_random_graph(N, p);
        Graph g_random_connected = GraphGenerator::make_random_connected_graph(N, p);
        Graph g_sparse = GraphGenerator::make_random_sparse_graph(N, p, max_degree, p_sparse);
        Graph g_dense = GraphGenerator::make_random_dense_graph(N, p, min_degree, p_dense);
        run_tests(g_random, p);
        run_tests(g_random_connected, p);
        run_tests(g_sparse, p);
        run_tests(g_dense, p);
    }

    // for (size_t i = 0; i < res.size(); i++)
    // {
    //     if (res[i] != resPara[i])
    //     {
    //         std::cout << "ERROR: " << i << ": " << res[i] << " != " << resPara[i] << std::endl;
    //     }
    // }
    // for (size_t i = 0; i < resPara.size(); i++) {
    //     std::cout << i << ": " << resPara[i] << std::endl;
    // }

    // To serialize a graph
    // std::ofstream out;
    // out.open("graph.txt", std::fstream::out);
    // out << g;
    // out.close();

    // To read a graph
    // std::ifstream in;
    // in.open("graph.txt", std::fstream::in);
    // in >> g;
    // std::cout << g.num_vertices();
    // in.close();

    // Testing the graph generator
    // std::vector<std::vector<int>> graph = generate_connected_graph(5);
    // for (int i = 0; i < 5; i++)
    // {
    //     for (int j = 0; j < 5; j++)
    //         std::cout << graph[i][j] << " ";
    //     std::cout << std::endl;
    // }
    return 0;
}