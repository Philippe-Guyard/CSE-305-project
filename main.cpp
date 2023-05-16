#include <iostream>
#include <vector>
#include <limits>
#include <unordered_set>
#include <fstream>
#include <thread>
#include <optional>

#include "graph.hpp"    

using dist_vector = std::vector<double>;

class DeltaSteppingSolver {
using bucket_t = std::unordered_set<vertex_t>;
private:
    enum edge_type { light, heavy };
    // Start with a constant 100k buckets
    const size_t MAX_BUCKETS = 100000;
    dist_vector distances;
    std::vector<bucket_t> buckets;
    const Graph& graph;
    double delta;

    void relax(Edge e) {
        double x = e.weight;
        vertex_t w = e.to;

        if (x < distances[w]) {
            size_t src_bucket_index = (size_t)(distances[w] / delta);
            size_t dest_bucket_index = (size_t)(x / delta);
            buckets[src_bucket_index].erase(w);
            buckets[dest_bucket_index].insert(w);
            distances[w] = x;
        }
    }

    std::optional<int> first_non_empty_bucket() {
        for (size_t i = 0; i < buckets.size(); i++) {
            if (!buckets[i].empty()) {
                return i;
            }
        }
        return std::nullopt;
    }

    bool edge_is_kind(const Edge& e, edge_type kind) const {
        if (kind == light) {
            return e.weight <= delta;
        }
        else {
            return e.weight > delta;
        }
    }

    std::vector<Edge> find_requests(const bucket_t& R, edge_type kind) const {
        std::vector<Edge> res = {};
        for (vertex_t v : R) {
            for (const Edge& e : graph.edges_from(v)) {
                if (edge_is_kind(e, kind)) {
                    res.push_back({e.to, e.weight + distances[v]});
                }
            }
        }      

        return res;
    }

    static void find_requests_thread(DeltaSteppingSolver *sol, const std::vector<vertex_t>& R_vec, size_t start, size_t end, std::vector<Edge>& result) {
        for (size_t j = start; j < end; ++j) {
            for (Edge e : sol->graph.edges_from(R_vec[j])) {
                if (sol->edge_is_kind(e, sol->light)) {
                    result.push_back({e.to, e.weight + sol->distances[R_vec[j]]});
                }
            }
        }
    }

    // NOTE: For the parallel version we need random index access, so we make R a vector instead of a set
    // This is the improvement from section 4 of the paper
    std::vector<Edge> find_requests_paralle(const bucket_t& R, edge_type kind, size_t num_threads) {
        size_t chunk_size = R.size() / num_threads;
        // Another copy... Slow 
        std::vector<vertex_t> R_vec(R.begin(), R.end());
        std::vector<std::vector<Edge>> results(num_threads - 1);
        std::vector<std::thread> threads(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; i++) {
            threads[i] = std::thread(find_requests_thread, this, std::ref(R_vec), i * chunk_size, (i + 1) * chunk_size, std::ref(results[i]));
        }
        size_t last_chunk_start = (num_threads - 1) * chunk_size;
        std::vector<Edge> res;
        for (size_t j = last_chunk_start; j < R.size(); ++j) {
            for (Edge e : graph.edges_from(R_vec[j])) {
                if (edge_is_kind(e, kind)) {
                    res.push_back({e.to, e.weight + distances[R_vec[j]]});
                }
            }
        }

        for (size_t i = 0; i < num_threads - 1; i++) {
            threads[i].join();
        }

        // NOTE: This combining might be quite slow. Is there a faster way to parallelize?
        for (size_t i = 0; i < num_threads - 1; i++) {
            if (results[i].size() > 0) 
                res.insert(res.end(), results[i].begin(), results[i].end());    
        }

        return res;
    }

    void relax_requests(const std::vector<Edge>& requests) {
        for (Edge e : requests) {
            relax(e);
        }
    }
public:
    // Idk why I do it like this, architecture is weird...
    DeltaSteppingSolver(const Graph &g): graph(g) {}

    dist_vector solve(vertex_t source, double delta) {
        // Implement the sequential delta-stepping algorithm as described in section 2 of 
        // https://reader.elsevier.com/reader/sd/pii/S0196677403000762?token=DBD927418ED5D4C911A1BF7217666F8DFE7446018FE2D1892A519D20FADCBE4A95A45FB4E44FD74C8BFD946BEE125078&originRegion=eu-west-1&originCreation=20230506110955
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets = std::vector<bucket_t>(MAX_BUCKETS);
        relax({source, 0});
        while (true) {
            auto opt_i = first_non_empty_bucket();
            if (!opt_i.has_value()) {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (!buckets[i].empty()) {
                auto requests = find_requests(buckets[i], light);
                R.insert(buckets[i].begin(), buckets[i].end());
                buckets[i].clear();
                relax_requests(requests);
            }
            auto requests = find_requests(R, heavy);
            relax_requests(requests);
        }

        return distances;
    }

    // Same as above, but use the improvement from section 4 for faster requests findings
    dist_vector solve_parallel_simple(vertex_t source, double delta, size_t num_threads) {
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets = std::vector<bucket_t>(MAX_BUCKETS);
        relax({source, 0});
        while (true) {
            auto opt_i = first_non_empty_bucket();
            if (!opt_i.has_value()) {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (!buckets[i].empty()) {
                auto requests = find_requests_paralle(buckets[i], light, num_threads);
                R.insert(buckets[i].begin(), buckets[i].end());
                buckets[i].clear();
                relax_requests(requests);
            }
            auto requests = find_requests_paralle(R, heavy, num_threads);
            relax_requests(requests);
        }

        return distances;
    }
};

int main() {
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
    // g.add_edge(A, B, 1);
    // g.add_edge(B, C, 1);
    // g.add_edge(C, D, 1);
    // g.add_edge(D, I, 1);
    // g.add_edge(I, G, 1);
    // g.add_edge(B, E, 1);
    // g.add_edge(E, F, 1);
    // g.add_edge(F, G, 1);
    // g.add_edge(G, H, 1);
    Graph g = make_random_graph(2000);

    DeltaSteppingSolver solver(g);
    auto start = std::chrono::high_resolution_clock::now();
//     auto res = solver.solve(A, 1);
    auto res = solver.solve(0, 0.25);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    // for (size_t i = 0; i < res.size(); i++) {
    //     std::cout << i << ": " << res[i] << std::endl;
    // }

    start = std::chrono::high_resolution_clock::now();
//     auto resPara = solver.solve_parallel_simple(A, 1, 4);
    auto resPara = solver.solve_parallel_simple(0, 0.25, 4);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    for(size_t i = 0; i < res.size(); i++) {
        if (res[i] != resPara[i]) {
            std::cout << "ERROR: " << i << ": " << res[i] << " != " << resPara[i] << std::endl;
        }
    }
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

    return 0;
}