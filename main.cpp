#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

#include "buckets.hpp"
#include "dijkstra.cpp"
#include "graph.hpp"
#include "test.cpp"
#include "benchmarker.hpp"

using dist_vector = std::vector<double>;

class DeltaSteppingSolver {
private:
    enum edge_type {
        light,
        heavy
    };
    dist_vector distances;
    std::unique_ptr<BucketListBase> buckets;
    const Graph &graph;
    double delta;

    void relax(Edge e) {
        Benchmarker::start_one("relax");
        double x = e.weight;
        vertex_t w = e.to;

        if (x < distances[w]) {
            size_t src_bucket_index = (size_t)(distances[w] / delta);
            size_t dest_bucket_index = (size_t)(x / delta);
            // Special case for infinity: somehow size_t(distances[w] / delta) == 0
            if (distances[w] != std::numeric_limits<double>::infinity()) {
                buckets->erase(src_bucket_index, w);
            }
            buckets->insert(dest_bucket_index, w);
            distances[w] = x;
        }
        Benchmarker::end_one("relax");
    }

    bool edge_is_kind(const Edge &e, edge_type kind) const {
        if (kind == light) {
            return e.weight <= delta;
        } else {
            return e.weight > delta;
        }
    }

    template <typename Iterator>
    std::vector<Edge> find_requests(Iterator begin, Iterator end, edge_type kind) const {
        Benchmarker::start_one("find_requests");
        std::vector<Edge> res = {};
        for (auto it = begin; it != end; it++) {
            vertex_t v = *it;
            for (const Edge &e : graph.edges_from(v)) {
                // Small optimization compared to the paper: we check the distances before pushing them to the vector
                // To avoid iterating over useless distances and calling relax too many times
                if (edge_is_kind(e, kind) && e.weight + distances[v] < distances[e.to]) {
                    res.push_back({e.to, e.weight + distances[v]});
                }
            }
        }

        Benchmarker::end_one("find_requests");
        return std::move(res);
    }

    static void find_requests_thread(DeltaSteppingSolver *sol, const std::vector<vertex_t> &R_vec,
                                     size_t start, size_t end, std::vector<Edge> &result,
                                     DeltaSteppingSolver::edge_type kind) {
        for (size_t j = start; j < end; ++j) {
            for (Edge e : sol->graph.edges_from(R_vec[j])) {
                if (sol->edge_is_kind(e, kind) && e.weight + sol->distances[R_vec[j]] < sol->distances[e.to]) {
                    result.push_back({e.to, e.weight + sol->distances[R_vec[j]]});
                }
            }
        }
    }

    // NOTE: For the parallel version we need random index access, so we make R a vector instead of a set
    // This is the improvement from section 4 of the paper
    template <typename Iterator>
    std::vector<Edge> find_requests_parallel(Iterator begin, Iterator end, edge_type kind, size_t num_threads) {
        Benchmarker::start_one("find_requests_parallel");
        // Another copy... Slow
        std::vector<vertex_t> R_vec(begin, end);
        size_t chunk_size = R_vec.size() / num_threads;
        std::vector<std::vector<Edge>> results(num_threads - 1);
        std::vector<std::thread> threads(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; i++) {
            threads[i] = std::thread(find_requests_thread, this, std::cref(R_vec), i * chunk_size, (i + 1) * chunk_size, std::ref(results[i]), kind);
        }
        size_t last_chunk_start = (num_threads - 1) * chunk_size;
        std::vector<Edge> res;
        // Push to res immediately to avoid unnecessary copying
        find_requests_thread(this, R_vec, last_chunk_start, R_vec.size(), res, kind);

        for (size_t i = 0; i < num_threads - 1; i++) {
            threads[i].join();
        }

        // NOTE: This combining might be quite slow. Is there a faster way to parallelize?
        for (size_t i = 0; i < num_threads - 1; i++) {
            if (results[i].size() > 0)
                res.insert(res.end(), results[i].begin(), results[i].end());
        }

        Benchmarker::end_one("find_requests_parallel");
        return std::move(res);
    }

    void relax_requests(const std::vector<Edge> &requests) {
        for (Edge e : requests) {
            relax(e);
        }
    }

public:
    // Idk why I do it like this, architecture is weird...
    DeltaSteppingSolver(const Graph &g, bool use_simple = true) : graph(g) {
        if (use_simple)
            buckets = std::make_unique<SimpleBucketList>();
        else
            buckets = std::make_unique<PrioritizedBucketList>();
    }

    dist_vector solve(vertex_t source, double delta) {
        // Implement the sequential delta-stepping algorithm as described in section 2 of
        // https://reader.elsevier.com/reader/sd/pii/S0196677403000762?token=DBD927418ED5D4C911A1BF7217666F8DFE7446018FE2D1892A519D20FADCBE4A95A45FB4E44FD74C8BFD946BEE125078&originRegion=eu-west-1&originCreation=20230506110955
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        while (!buckets->empty()) {
            auto opt_i = buckets->first_non_empty_bucket();
            if (!opt_i.has_value()) {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (buckets->size_of(i) > 0) {
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
    dist_vector solve_parallel_simple(vertex_t source, double delta, size_t num_threads) {
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        while (true) {
            auto opt_i = buckets->first_non_empty_bucket();
            if (!opt_i.has_value()) {
                break;
            }
            int i = opt_i.value();
            bucket_t R;
            while (buckets->size_of(i) > 0) {
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

    edge_weight find_delta() {
        auto blocked_adjacency_lists = graph.gen_blocked_adjacency_lists();

        edge_weight inf = std::numeric_limits<double>::infinity();
        HashArray<UEdge, edge_weight> found(inf);
        std::unordered_set<DEdge> Q({}), Q_next({});
        for (int i = 0; i < graph.num_vertices(); ++i) {
            Q.insert(DEdge({i, i, 0}));
        }
        edge_weight delta_0 = graph.get_delta_0();
        edge_weight delta_cur = delta_0;
        std::vector<std::vector<TEdge>> T(ceil(log(graph.get_delta_0() / delta_0)));
        std::unordered_set<DEdge> S({}), S_next({});
        std::multiset<DEdge> Q_dash;

        for (int i = 0; i < T.size(); ++i) {
            S_next.clear();
            Q_next.clear();
            for (auto t_edge : T[i]) {
                auto block = blocked_adjacency_lists[t_edge.to][t_edge.b];
                for (auto edge : block) {
                    //{(u, w, x + c(v, w))};
                    Q_dash.insert(DEdge({t_edge.from, edge.to, t_edge.weight + edge.weight}));
                }
                // log((x + c(first edge of block b + 1 in v’s adjacency list))/∆0)
                // TODO: not sure how t_edge.b + 1 works here
                double c = blocked_adjacency_lists[t_edge.to][t_edge.b + 1][0].weight;
                int j = floor(log((t_edge.weight + c)) / delta_0);
                T[j].push_back(TEdge({t_edge.from, t_edge.to, t_edge.weight, t_edge.b + 1}));
            }

            while (!Q.empty()) {
                for (auto d_edge : Q) {
                    // foreach  edge(v, w) ∈ E having c(v, w) < ∆cur do
                    //         Q' : = Q' ∪ { (u, w, x + c(v, w)) }
                    for (auto edge : graph.edges_from(d_edge.to)) {
                        if (edge.weight < delta_cur) {
                            Q_dash.insert(DEdge({d_edge.from, edge.to, d_edge.weight + edge.weight}));
                        }
                    }
                }
                // semi - sort Q' by common start and destination node -> happens automatically
                // H:= {(u, v, x) : x = min{y : (u, v, y) ∈ Q' } }
                edge_weight min_weight = std::numeric_limits<edge_weight>::max();
                for (const auto &d_edge : Q_dash) {
                    if (d_edge.weight < min_weight) {
                        min_weight = d_edge.weight;
                    }
                }
                std::unordered_set<DEdge> H;
                for (const auto &d_edge : Q_dash) {
                    if (d_edge.weight == min_weight) {
                        H.insert(d_edge);
                    }
                }
                for (auto &d_edge : H) {
                    UEdge conn({d_edge.from, d_edge.to});
                    if (d_edge.weight < delta_cur) {
                        Q.insert(d_edge);

                        if (found.get(conn) == inf) {
                            S.insert(d_edge);
                        }
                    } else {
                        Q_next.insert(d_edge);
                        if (found.get(conn) == inf) {
                            S_next.insert(d_edge);
                        }
                    }
                    found.insert(conn, d_edge.weight);
                }
                Q_dash.clear();
            }
            for (DEdge edge : S) {
                // b = first block in v’s adj.list having edges heavier than ∆cur
                int b = 0;
                bool flag = false;
                while (!flag) {
                    for (auto edge : blocked_adjacency_lists[edge.to][b]) {
                        if (edge.weight > delta_cur) {
                            flag = true;
                            break;
                        }
                    }
                    b++;
                }
                edge_weight x = found.get(UEdge({edge.from, edge.to}));
                double c = blocked_adjacency_lists[edge.to][b][0].weight;
                int j = floor(log((x + c)) / delta_0);
                T[j].push_back(TEdge({edge.from, edge.to, x, b}));
            }
            Q = Q_next;
            S = S_next;
            delta_cur *= 2;
        }
        return graph.get_delta_0();
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

    auto start = std::chrono::high_resolution_clock::now();
    auto resDijkstra = dijkstra(g, 0);
    auto end = std::chrono::high_resolution_clock::now();
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