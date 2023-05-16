#include <iostream>
#include <vector>
#include <limits>
#include <unordered_set>
#include <fstream>

#include "graph.hpp"    

using dist_vector = std::vector<double>;

class SequentialDeltaSteppingSolver {
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

    int first_non_empty_bucket() {
        for (size_t i = 0; i < buckets.size(); i++) {
            if (!buckets[i].empty()) {
                return i;
            }
        }
        return -1;
    }

    bool edge_is_kind(Edge e, edge_type kind) {
        if (kind == light) {
            return e.weight <= delta;
        }
        else {
            return e.weight > delta;
        }
    }

    std::vector<Edge> find_requests(const bucket_t& R, edge_type kind) {
        std::vector<Edge> res = {};
        for (vertex_t v : R) {
            for (Edge e : graph.edges_from(v)) {
                if (edge_is_kind(e, kind)) {
                    res.push_back({e.to, e.weight + distances[v]});
                }
            }
        }      

        return std::move(res);
    }

    void relax_requests(const std::vector<Edge>& requests) {
        for (Edge e : requests) {
            relax(e);
        }
    }
public:
    // Idk why I do it like this, architecture is weird...
    SequentialDeltaSteppingSolver(const Graph &g): graph(g) {}

    dist_vector solve(vertex_t source, double delta) {
        // Implement the sequential delta-stepping algorithm as described in section 2 of 
        // https://reader.elsevier.com/reader/sd/pii/S0196677403000762?token=DBD927418ED5D4C911A1BF7217666F8DFE7446018FE2D1892A519D20FADCBE4A95A45FB4E44FD74C8BFD946BEE125078&originRegion=eu-west-1&originCreation=20230506110955
        this->delta = delta;
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets = std::vector<bucket_t>(MAX_BUCKETS);
        relax({source, 0});
        int i;
        while ((i = first_non_empty_bucket()) != -1) {
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
};

int main() {
    /*    
           C -> D -> I
          /           \
    A -> B             G -> H
          \           /
           E -> F-----
    */

    Graph g(9);
    vertex_t A = 0;
    vertex_t B = 1;
    vertex_t C = 2;
    vertex_t D = 3;
    vertex_t E = 4;
    vertex_t F = 5;
    vertex_t G = 6;
    vertex_t H = 7;
    vertex_t I = 8;
    g.add_edge(A, B, 1);
    g.add_edge(B, C, 1);
    g.add_edge(C, D, 1);
    g.add_edge(D, I, 1);
    g.add_edge(I, G, 1);
    g.add_edge(B, E, 1);
    g.add_edge(E, F, 1);
    g.add_edge(F, G, 1);
    g.add_edge(G, H, 1);

    SequentialDeltaSteppingSolver solver(g);
    auto res = solver.solve(A, 1);
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << i << ": " << res[i] << std::endl;
    }

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