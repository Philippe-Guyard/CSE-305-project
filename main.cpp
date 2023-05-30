#include <iostream>
#include <vector>
#include <limits>
#include <unordered_set>
#include <fstream>
#include <thread>
#include <optional>

#include "graph.hpp"
#include "test.cpp"
#include "dijkstra.cpp"
#include "buckets.hpp"

using dist_vector = std::vector<double>;

class DeltaSteppingSolver
{
private:
    enum edge_type
    {
        light,
        heavy
    };
    dist_vector distances;
    std::unique_ptr<BucketListBase> buckets;
    const Graph &graph;
    double delta;

    void relax(Edge e)
    {
        double x = e.weight;
        vertex_t w = e.to;

        if (x < distances[w])
        {
            size_t src_bucket_index = (size_t)(distances[w] / delta);
            size_t dest_bucket_index = (size_t)(x / delta);
            // Special case for infinity: somehow size_t(distances[w] / delta) == 0
            if (distances[w] != std::numeric_limits<double>::infinity())
            {
                buckets->erase(src_bucket_index, w);
            }
            buckets->insert(dest_bucket_index, w);
            distances[w] = x;
        }
    }

    bool edge_is_kind(const Edge &e, edge_type kind) const
    {
        if (kind == light)
        {
            return e.weight <= delta;
        }
        else
        {
            return e.weight > delta;
        }
    }

    template<typename Iterator>
    std::vector<Edge> find_requests(Iterator begin, Iterator end, edge_type kind) const
    {
        std::vector<Edge> res = {};
        for(auto it = begin; it != end; it++)
        {
            vertex_t v = *it;
            for (const Edge &e : graph.edges_from(v))
            {
                // Small optimization compared to the paper: we check the distances before pushing them to the vector
                // To avoid iterating over useless distances and calling relax too many times 
                if (edge_is_kind(e, kind) && e.weight + distances[v] < distances[e.to])
                {
                    res.push_back({e.to, e.weight + distances[v]});
                }
            }
        }

        return std::move(res);
    }

    static void find_requests_thread(DeltaSteppingSolver *sol, const std::vector<vertex_t> &R_vec, 
                                     size_t start, size_t end, std::vector<Edge> &result, 
                                     DeltaSteppingSolver::edge_type kind, const std::vector<double> &distances)
    {
        for (size_t j = start; j < end; ++j)
        {
            for (Edge e : sol->graph.edges_from(R_vec[j]))
            {
                if (sol->edge_is_kind(e, kind) && e.weight + distances[R_vec[j]] < sol->distances[e.to])
                {
                    result.push_back({e.to, e.weight + sol->distances[R_vec[j]]});
                }
            }
        }
    }

    // NOTE: For the parallel version we need random index access, so we make R a vector instead of a set
    // This is the improvement from section 4 of the paper
    template <typename Iterator>
    std::vector<Edge> find_requests_parallel(Iterator begin, Iterator end, edge_type kind, size_t num_threads)
    {
        // Another copy... Slow
        std::vector<vertex_t> R_vec(begin, end);
        size_t chunk_size = R_vec.size() / num_threads;
        std::vector<std::vector<Edge>> results(num_threads - 1);
        std::vector<std::thread> threads(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            threads[i] = std::thread(find_requests_thread, this, std::cref(R_vec), i * chunk_size, (i + 1) * chunk_size, std::ref(results[i]), kind, std::cref(distances));
        }
        size_t last_chunk_start = (num_threads - 1) * chunk_size;
        std::vector<Edge> res;
        // Push to res immediately to avoid unnecessary copying 
        find_requests_thread(this, R_vec, last_chunk_start, R_vec.size(), res, kind, distances);

        for (size_t i = 0; i < num_threads - 1; i++)
        {
            threads[i].join();
        }

        // NOTE: This combining might be quite slow. Is there a faster way to parallelize?
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            if (results[i].size() > 0)
                res.insert(res.end(), results[i].begin(), results[i].end());
        }

        return std::move(res);
    }

    void relax_requests(const std::vector<Edge> &requests)
    {
        for (Edge e : requests)
        {
            relax(e);
        }
    }

public:
    // Idk why I do it like this, architecture is weird...
    DeltaSteppingSolver(const Graph &g, bool use_simple=true) : graph(g) {
        if (use_simple)
            buckets = std::make_unique<SimpleBucketList>();
        else
            buckets = std::make_unique<PrioritizedBucketList>();
    }

    dist_vector solve(vertex_t source, double delta)
    {
        // Implement the sequential delta-stepping algorithm as described in section 2 of
        // https://reader.elsevier.com/reader/sd/pii/S0196677403000762?token=DBD927418ED5D4C911A1BF7217666F8DFE7446018FE2D1892A519D20FADCBE4A95A45FB4E44FD74C8BFD946BEE125078&originRegion=eu-west-1&originCreation=20230506110955
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        while (!buckets->empty())
        {
            auto opt_i = buckets->first_non_empty_bucket();
            if (!opt_i.has_value())
            {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (buckets->size_of(i) > 0)
            {
                auto requests = find_requests(buckets->begin_of(i), buckets->end_of(i), light);
                R.insert(buckets->begin_of(i), buckets->end_of(i));
                buckets->clear_at(i);
                relax_requests(requests);
            }
            auto requests = find_requests(R.begin(), R.end(), heavy);
            relax_requests(requests);
        }

        return distances;
    }

    // Same as above, but use the improvement from section 4 for faster requests findings
    dist_vector solve_parallel_simple(vertex_t source, double delta, size_t num_threads)
    {
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        while (true)
        {
            auto opt_i = buckets->first_non_empty_bucket();
            if (!opt_i.has_value())
            {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (buckets->size_of(i) > 0)
            {
                auto requests = find_requests_parallel(buckets->begin_of(i), buckets->end_of(i), light, num_threads);
                R.insert(buckets->begin_of(i), buckets->end_of(i));
                buckets->clear_at(i);
                relax_requests(requests);
            }
            auto requests = find_requests_parallel(R.begin(), R.end(), heavy, num_threads);
            relax_requests(requests);
        }

        return distances;
    }
};

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

    Graph g = make_connected_enough_graph(10000);

    // we don't print any weights here
    // for (size_t i = 0; i < g.num_vertices(); i++)
    // {
    //     std::cout << i << ": "; // prints the vertex number
    //     for (Edge e : g.edges_from(i))
    //     {
    //         std::cout << e.to << " "; // prints the all the destination vertices numbers
    //     }
    //     std::cout << std::endl;
    // }
    // for (size_t i = 0; i < g.num_vertices(); i++) {
    //     std::cout << i << ": ";
    //     for (Edge e : g.edges_from(i)) {
    //         std::cout << e.to << " ";
    //     }
    //     std::cout << std::endl;
    // }

    const double delta = 0.1;
    DeltaSteppingSolver solver(g, false);
    auto start = std::chrono::high_resolution_clock::now();
    auto res = solver.solve(0, delta);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential (delta = " << delta << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    size_t num_threads = 8;
    auto resPara = solver.solve_parallel_simple(0, delta, num_threads);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel (with " << num_threads << " threads): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto resDijkstra = dijkstra(g, 0);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Dijkstra: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    for (size_t i = 0; i < res.size(); i++) {
        if (res[i] != resDijkstra[i] || res[i] != resPara[i]) {
            std::cout << "ERROR: " << i << ": " << res[i] << " != " << resDijkstra[i];
            std::cout << " or " << res[i] << " != " << resPara[i] << std::endl;
        }
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