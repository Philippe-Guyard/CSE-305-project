#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>
#include <memory>
#include <omp.h>
#include <algorithm>

#include "buckets.hpp"
#include "graph.hpp"
#include "benchmarker.hpp"
#include "generator.hpp"

using dist_vector = std::vector<double>;

class DeltaSteppingSolver
{
    using shortcut_array = std::unordered_map<vertex_t, std::vector<Edge>>;

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

    void relax(const Edge &e)
    {
        // NOTE: Benchmarking in this function makes it significantly slower
        // Due to the internals of the Benchmarker class
        // Benchmarker::start_one("relax");
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
        // Benchmarker::end_one("relax");
    }

    bool edge_is_kind(const Edge &e, edge_type kind)
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

    template <typename Iterator>
    std::vector<Edge> find_requests(Iterator begin, Iterator end, edge_type kind)
    {
        Benchmarker::start_one("find_requests");
        std::vector<Edge> res = {};
        for (auto it = begin; it != end; it++)
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

        Benchmarker::end_one("find_requests");
        return std::move(res);
    }

    static void find_requests_thread(DeltaSteppingSolver *sol, const std::vector<vertex_t> &R_vec,
                                     size_t start, size_t end, std::vector<Edge> &result,
                                     DeltaSteppingSolver::edge_type kind)
    {
        for (size_t j = start; j < end; ++j)
        {
            for (Edge e : sol->graph.edges_from(R_vec[j]))
            {
                if (sol->edge_is_kind(e, kind) && e.weight + sol->distances[R_vec[j]] < sol->distances[e.to])
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
        Benchmarker::start_one("find_requests_parallel");
        // Another copy... Slow
        Benchmarker::start_one("Copying");
        std::vector<vertex_t> R_vec(begin, end);
        Benchmarker::end_one("Copying");
        size_t chunk_size = R_vec.size() / num_threads;
        std::vector<std::vector<Edge>> results(num_threads - 1);
        std::vector<std::thread> threads(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            threads[i] = std::thread(find_requests_thread, this, std::cref(R_vec), i * chunk_size, (i + 1) * chunk_size, std::ref(results[i]), kind);
        }
        size_t last_chunk_start = (num_threads - 1) * chunk_size;
        std::vector<Edge> res;
        // Push to res immediately to avoid unnecessary copying
        find_requests_thread(this, R_vec, last_chunk_start, R_vec.size(), res, kind);

        for (size_t i = 0; i < num_threads - 1; i++)
        {
            Benchmarker::start_one("Joining");
            threads[i].join();
            Benchmarker::end_one("Joining");
        }

        // NOTE: This combining might be quite slow. Is there a faster way to parallelize?
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            Benchmarker::start_one("Copying");
            if (results[i].size() > 0)
                res.insert(res.end(), results[i].begin(), results[i].end());
            Benchmarker::end_one("Copying");
        }

        Benchmarker::end_one("find_requests_parallel");
        return std::move(res);
    }

    /*
     * Same as find_requests_parallel, but using OpenMP
     */
    template <typename Iterator>
    std::vector<Edge> find_requests_omp(Iterator begin, Iterator end, edge_type kind, size_t num_threads)
    {
        Benchmarker::start_one("find_requests_omp");
        std::vector<vertex_t> R_vec(begin, end);
        std::vector<std::vector<Edge>> results(num_threads);
#pragma omp parallel for shared(R_vec, results) num_threads(num_threads) schedule(dynamic)
        for (size_t j = 0; j < R_vec.size(); ++j)
        {
            vertex_t v = R_vec[j];
            size_t thread_id = omp_get_thread_num();
            for (const Edge &e : graph.edges_from(v))
            {
                // Small optimization compared to the paper: we check the distances before pushing them to the vector
                // To avoid iterating over useless distances and calling relax too many times
                if (edge_is_kind(e, kind) && e.weight + distances[v] < distances[e.to])
                {
                    results[thread_id].push_back({e.to, e.weight + distances[v]});
                }
            }
        }

        std::vector<Edge> res;
        for (size_t i = 0; i < num_threads; ++i)
        {
            res.insert(res.end(), results[i].begin(), results[i].end());
        }

        Benchmarker::end_one("find_requests_omp");
        return std::move(res);
    }

    template <typename ReqFunc>
    void solve_base(vertex_t source, double delta, ReqFunc get_requests)
    {
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
                auto requests = get_requests(buckets->begin_of(i), buckets->end_of(i), edge_type::light);
                R.insert(buckets->begin_of(i), buckets->end_of(i));
                buckets->clear_at(i);
                relax_requests(requests);
            }
            auto requests = get_requests(R.begin(), R.end(), edge_type::heavy);
            relax_requests(requests);
        }
    }

    void relax_requests(const std::vector<Edge> &requests)
    {
        Benchmarker::start_one("relax_requests");
        for (const Edge &e : requests)
        {
            relax(e);
        }
        Benchmarker::end_one("relax_requests");
    }

    shortcut_array to_shortcut_array(const HashArray<UEdge, double> &found)
    {
        shortcut_array shortcuts_from;
        for (auto it = found.begin(); it != found.end(); it++)
        {
            vertex_t u = it->first.from;
            vertex_t v = it->first.to;
            double x = it->second;
            shortcuts_from[u].push_back(Edge({v, x}));
        }
        return shortcuts_from;
    }

    shortcut_array find_shortcuts_simple(vertex_t source)
    {
        Benchmarker::start_one("find_shortcuts_simple");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::unordered_set<DEdge> Q;
        // To have them in sorted order
        HashArray<UEdge, double> Q_dash(std::numeric_limits<double>::infinity());
        Q.insert(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Q_dash.clear();
            Benchmarker::start_one("Computing Q_dash");
            for (auto d_edge : Q)
            {
                size_t u = d_edge.from;
                size_t v = d_edge.to;
                double x = d_edge.weight;
                for (auto edge : graph.edges_from(v))
                {
                    size_t w = edge.to;
                    double c = edge.weight;
                    // For single thread, immediately insert the minimum to avoid
                    // storing the whole multiset
                    if (edge_is_kind(edge, light) && x + c < Q_dash.get(UEdge(u, w)))
                    {
                        Q_dash.insert(UEdge(u, w), x + c);
                    }
                }
            }
            Benchmarker::end_one("Computing Q_dash");

            Q.clear();
            for (auto it = Q_dash.begin(); it != Q_dash.end(); it++)
            {
                vertex_t u = it->first.from;
                vertex_t v = it->first.to;
                double x = it->second;
                // filter by delta <= x < found(u, v)
                if (x >= delta || x > found.get(UEdge(u, v)))
                {
                    continue;
                }

                Q.insert(DEdge(u, v, x));
                found.insert(UEdge(u, v), x);
            }
        }

        auto shortcuts_from = to_shortcut_array(found);
        Benchmarker::end_one("find_shortcuts_simple");
        return shortcuts_from;
    }

    static void find_shortcuts_thread(DeltaSteppingSolver *sol, const std::vector<DEdge> &Q, size_t idx_start, size_t idx_end,
                                      HashArray<UEdge, double> &Q_dash, const HashArray<UEdge, double> &found)
    {
        for (size_t i = idx_start; i < idx_end; i++)
        {
            size_t u = Q[i].from;
            size_t v = Q[i].to;
            double x = Q[i].weight;
            for (auto edge : sol->graph.edges_from(v))
            {
                size_t w = edge.to;
                double c = edge.weight;
                auto edge_key = UEdge(u, w);
                // NOTE: We are immediately doing the filtering of x <= delta and x < found
                // as well as immediately computing minimums on insertion to Q_dash.
                // It is ok to filter now, since the edges that do not satisfy these conditions
                // would not be inserted into Q_dash anyway
                // double cur_best = std::fmin(found.get(edge_key), Q_dash.get(edge_key));
                // if (x + c <= sol->delta && x + c < cur_best) {
                //     Q_dash.insert(edge_key, x + c);
                // }
                if (x + c <= sol->delta && x + c < std::fmin(Q_dash.get(edge_key), found.get(edge_key)))
                {
                    Q_dash.insert(edge_key, x + c);
                }
            }
        }
    }

    std::unordered_map<vertex_t, std::vector<Edge>> find_shortcuts_parallel(vertex_t source, size_t num_threads)
    {
        Benchmarker::start_one("find_shortcuts_parallel");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::vector<DEdge> Q;
        std::vector<HashArray<UEdge, double>> Q_dash(num_threads, HashArray<UEdge, double>(std::numeric_limits<double>::infinity()));
        Q.push_back(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Benchmarker::start_one("Computing Q_dash");
            size_t chunk_size = Q.size() / num_threads;
            // Make sure the extra nodes are spread evenly across threads
            // I am not sure this helps anyhow, but trying desperately to fix a bug...
            size_t extra_nodes = Q.size() - chunk_size * num_threads;
            size_t cur_begin = 0;
            size_t cur_end = chunk_size;
            std::vector<std::thread> threads(num_threads - 1);
            for (size_t i = 0; i < num_threads - 1; i++)
            {
                if (i < extra_nodes)
                    cur_end++;

                Q_dash[i].clear();
                threads[i] = std::thread(find_shortcuts_thread, this, std::cref(Q), cur_begin,
                                         cur_end, std::ref(Q_dash[i]),
                                         std::cref(found));
                cur_begin = cur_end;
                cur_end += chunk_size;
            }

            Q_dash[num_threads - 1].clear();
            if (cur_begin < Q.size())
                find_shortcuts_thread(this, Q, cur_begin, Q.size(), Q_dash[num_threads - 1], found);

            Benchmarker::start_one("Joining");
            for (size_t i = 0; i < num_threads - 1; i++)
                threads[i].join();
            Benchmarker::end_one("Joining");

            Benchmarker::end_one("Computing Q_dash");

            Q.clear();

            // Now compute min(Q_dash[i].get(u, v)) for all (u, v) and put it in found
            Benchmarker::start_one("Updating Q");
            HashArray<UEdge, bool> visited(false);
            for (size_t i = 0; i < num_threads; i++)
            {
                for (auto it = Q_dash[i].begin(); it != Q_dash[i].end(); it++)
                {
                    vertex_t u = it->first.from;
                    vertex_t v = it->first.to;
                    auto edge_key = UEdge(u, v);
                    if (visited.get(edge_key))
                        continue;

                    visited.insert(edge_key, true);
                    double min_val = it->second;
                    // We can start from i + 1, since {u, v} is not visited, hence Q_dash[j < i].get(u, v) == inf
                    for (size_t j = i + 1; j < num_threads; j++)
                        min_val = std::fmin(min_val, Q_dash[j].get(edge_key));

                    if (min_val < found.get(edge_key) && min_val <= delta)
                    {
                        found.insert(edge_key, min_val);
                        Q.emplace_back(DEdge(u, v, min_val));
                    }
                }
            }
            Benchmarker::end_one("Updating Q");
        }

        auto shortcuts_from = to_shortcut_array(found);
        Benchmarker::end_one("find_shortcuts_parallel");
        return shortcuts_from;
    }

    /*
     * Same as find_shortcuts_parallel, but using OpenMP instead of std::thread
     */
    shortcut_array find_shortcuts_omp(vertex_t source)
    {
        Benchmarker::start_one("find_shortcuts_omp");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::vector<DEdge> Q;
        // Immediately set the default to delta, since edges < delta are not interesting to us
        std::vector<HashArray<UEdge, double>> Q_dash(omp_get_max_threads(), HashArray<UEdge, double>(this->delta));
        Q.push_back(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Benchmarker::start_one("Computing Q_dash");
#pragma omp parallel shared(Q, Q_dash)
            {
#pragma omp for schedule(dynamic)
                for (size_t i = 0; i < Q.size(); ++i)
                {
                    size_t thread_id = omp_get_thread_num();
                    for (auto edge : graph.edges_from(Q[i].to))
                    {
                        double new_cost = Q[i].weight + edge.weight;
                        auto edge_key = UEdge(Q[i].from, edge.to);
                        if (new_cost <= Q_dash[thread_id].get(edge_key))
                        {
                            Q_dash[thread_id].insert(edge_key, new_cost);
                        }
                    }
                }
            }
            Benchmarker::end_one("Computing Q_dash");

            Q.clear();
            Benchmarker::start_one("Updating Q");
            HashArray<UEdge, bool> visited(false);
            for (size_t i = 0; i < Q_dash.size(); ++i)
            {
                for (auto it = Q_dash[i].begin(); it != Q_dash[i].end(); it++)
                {
                    vertex_t u = it->first.from;
                    vertex_t v = it->first.to;
                    auto edge_key = UEdge(u, v);
                    if (visited.get(edge_key))
                        continue;

                    visited.insert(edge_key, true);
                    double min_val = it->second;
                    for (size_t j = i + 1; j < Q_dash.size(); j++)
                        min_val = std::fmin(min_val, Q_dash[j].get(edge_key));

                    if (min_val < found.get(edge_key) && min_val <= delta)
                    {
                        found.insert(edge_key, min_val);
                        Q.emplace_back(DEdge(u, v, min_val));
                    }
                }

                Q_dash[i].clear();
            }
            Benchmarker::end_one("Updating Q");
        }

        Benchmarker::end_one("find_shortcuts_omp");
        return to_shortcut_array(found);
    }

    void solve_shortcuts_base(shortcut_array &shortcuts_from, vertex_t source)
    {
        distances = dist_vector(graph.num_vertices(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        int i = -1;
        while (!buckets->empty())
        {
            auto opt_i = buckets->first_non_empty_bucket(i);
            if (!opt_i.has_value())
                break;

            i = opt_i.value();
            for (auto it = buckets->begin_of(i); it != buckets->end_of(i); it++)
            {
                vertex_t v = *it;
                // distances[v] + e.weight <= (i + 1) * delta <=> e.weight <= (i + 1) * delta - distances[v]
                double target_weight = (i + 1) * delta - distances[v];
                for (const Edge &e : graph.edges_from(v))
                {
                    if (e.weight <= target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
                for (const Edge &e : shortcuts_from[v])
                {
                    if (e.weight <= target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
            }
            for (auto it = buckets->begin_of(i); it != buckets->end_of(i); it++)
            {
                vertex_t v = *it;
                // distances[v] + e.weight > (i + 1) * delta <=> e.weight > (i + 1) * delta - distances[v]
                double target_weight = (i + 1) * delta - distances[v];
                for (const Edge &e : graph.edges_from(v))
                {
                    if (e.weight > target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
                for (const Edge &e : shortcuts_from[v])
                {
                    if (e.weight > target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
            }
        }
    }

public:
    // Idk why I do it like this, architecture is weird...
    DeltaSteppingSolver(const Graph &g, bool use_simple = true) : graph(g)
    {
        if (use_simple)
            buckets = std::make_unique<SimpleBucketList>();
        else
            buckets = std::make_unique<PrioritizedBucketList>();
    }

    dist_vector solve(vertex_t source, double delta)
    {
        solve_base(source, delta, [this](auto begin, auto end, auto kind)
                   { return find_requests(begin, end, kind); });
        return distances;
    }

    // Same as above, but use the improvement from section 4 for faster requests findings
    dist_vector solve_parallel_simple(vertex_t source, double delta, size_t num_threads)
    {
        solve_base(source, delta, [this, num_threads](auto begin, auto end, auto kind)
                   { return find_requests_parallel(begin, end, kind, num_threads); });
        return distances;
    }

    dist_vector solve_parallel_omp(vertex_t source, double delta, size_t num_threads)
    {
        solve_base(source, delta, [this, num_threads](auto begin, auto end, auto kind)
                   { return find_requests_omp(begin, end, kind, num_threads); });
        return distances;
    }

    dist_vector solve_shortcuts(vertex_t source, double delta)
    {
        this->delta = delta;
        auto shortcuts_from = find_shortcuts_simple(source);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }

    dist_vector solve_shortcuts_parallel(vertex_t source, double delta, size_t num_threads)
    {
        this->delta = delta;
        auto shortcuts_from = find_shortcuts_parallel(source, num_threads);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }

    dist_vector solve_shortcuts_omp(vertex_t source, double delta)
    {
        this->delta = delta;
        auto shortcuts_from = find_shortcuts_omp(source);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }

    edge_weight find_delta()
    {
        auto blocked_adjacency_lists = graph.gen_blocked_adjacency_lists();

        edge_weight inf = std::numeric_limits<double>::infinity();
        HashArray<UEdge, edge_weight> found(inf);
        std::unordered_set<DEdge> Q({}), Q_next({});
        for (int i = 0; i < graph.num_vertices(); ++i)
        {
            Q.insert(DEdge({i, i, 0}));
        }
        edge_weight delta_0 = graph.get_delta_0();
        edge_weight delta_cur = delta_0;
        std::vector<std::vector<TEdge>> T(ceil(log(graph.get_delta_0() / delta_0)));
        std::unordered_set<DEdge> S({}), S_next({});
        std::multiset<DEdge> Q_dash;

        for (int i = 0; i < T.size(); ++i)
        {
            S_next.clear();
            Q_next.clear();
            for (auto t_edge : T[i])
            {
                auto block = blocked_adjacency_lists[t_edge.to][t_edge.b];
                for (auto edge : block)
                {
                    //{(u, w, x + c(v, w))};
                    Q_dash.insert(DEdge({t_edge.from, edge.to, t_edge.weight + edge.weight}));
                }
                // log((x + c(first edge of block b + 1 in v’s adjacency list))/∆0)
                // TODO: not sure how t_edge.b + 1 works here
                double c = blocked_adjacency_lists[t_edge.to][t_edge.b + 1][0].weight;
                int j = floor(log((t_edge.weight + c)) / delta_0);
                T[j].push_back(TEdge({t_edge.from, t_edge.to, t_edge.weight, t_edge.b + 1}));
            }

            while (!Q.empty())
            {
                for (auto d_edge : Q)
                {
                    // foreach  edge(v, w) ∈ E having c(v, w) < ∆cur do
                    //         Q' : = Q' ∪ { (u, w, x + c(v, w)) }
                    for (auto edge : graph.edges_from(d_edge.to))
                    {
                        if (edge.weight < delta_cur)
                        {
                            Q_dash.insert(DEdge({d_edge.from, edge.to, d_edge.weight + edge.weight}));
                        }
                    }
                }
                // semi - sort Q' by common start and destination node -> happens automatically
                // H:= {(u, v, x) : x = min{y : (u, v, y) ∈ Q' } }
                edge_weight min_weight = std::numeric_limits<edge_weight>::max();
                for (const auto &d_edge : Q_dash)
                {
                    if (d_edge.weight < min_weight)
                    {
                        min_weight = d_edge.weight;
                    }
                }
                std::unordered_set<DEdge> H;
                for (const auto &d_edge : Q_dash)
                {
                    if (d_edge.weight == min_weight)
                    {
                        H.insert(d_edge);
                    }
                }
                for (auto &d_edge : H)
                {
                    UEdge conn({d_edge.from, d_edge.to});
                    if (d_edge.weight < delta_cur)
                    {
                        Q.insert(d_edge);

                        if (found.get(conn) == inf)
                        {
                            S.insert(d_edge);
                        }
                    }
                    else
                    {
                        Q_next.insert(d_edge);
                        if (found.get(conn) == inf)
                        {
                            S_next.insert(d_edge);
                        }
                    }
                    found.insert(conn, d_edge.weight);
                }
                Q_dash.clear();
            }
            for (DEdge edge : S)
            {
                // b = first block in v’s adj.list having edges heavier than ∆cur
                int b = 0;
                bool flag = false;
                while (!flag)
                {
                    for (auto edge : blocked_adjacency_lists[edge.to][b])
                    {
                        if (edge.weight > delta_cur)
                        {
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

/*
TODO: A lot of code duplication here, which is sad, but could not figure out a clean solution.
The idea here is that we use a graph with sorted edges, which allows us to perform way less checks. 
There is a similar improvment for dijkstra, but the bet is that for parallel delta-stepping it will be more significant.
*/
class DeltaSteppingSolverSorted
{
    using shortcut_array = std::unordered_map<vertex_t, std::vector<Edge>>;

private:
    enum edge_type
    {
        light,
        heavy
    };
    dist_vector distances;
    std::unique_ptr<BucketListBase> buckets;
    const Graph& graph;
    std::vector<std::pair<std::vector<Edge>, std::vector<Edge>>> semisorted_edges;
    double delta;

    void relax(const Edge &e)
    {
        // NOTE: Benchmarking in this function makes it significantly slower
        // Due to the internals of the Benchmarker class
        // Benchmarker::start_one("relax");
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
        // Benchmarker::end_one("relax");
    }

    bool edge_is_kind(const Edge &e, edge_type kind)
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

    static void find_requests_vector(DeltaSteppingSolverSorted* sol, vertex_t v, edge_type kind, std::vector<Edge>& res) {
        if (kind == light) {
            for (const Edge &e : sol->semisorted_edges[v].first)
            {
                // Small optimization compared to the paper: we check the distances before pushing them to the vector
                // To avoid iterating over useless distances and calling relax too many times
                if (e.weight + sol->distances[v] < sol->distances[e.to])
                {
                    res.push_back({e.to, e.weight + sol->distances[v]});
                }
            }
        }
        else {
           for(const Edge& e : sol->semisorted_edges[v].second) {
                // Small optimization compared to the paper: we check the distances before pushing them to the vector
                // To avoid iterating over useless distances and calling relax too many times
                if (e.weight + sol->distances[v] < sol->distances[e.to])
                {
                    res.push_back({e.to, e.weight + sol->distances[v]});
                }
            }
        }
    }

    template <typename Iterator>
    std::vector<Edge> find_requests(Iterator begin, Iterator end, edge_type kind)
    {
        Benchmarker::start_one("find_requests");
        std::vector<Edge> res = {};
        for (auto it = begin; it != end; it++)
        {
            vertex_t v = *it;
            find_requests_vector(this, v, kind, res);
        }

        Benchmarker::end_one("find_requests");
        return std::move(res);
    }

    static void find_requests_thread(DeltaSteppingSolverSorted *sol, const std::vector<vertex_t> &R_vec,
                                     size_t start, size_t end, std::vector<Edge> &result,
                                     DeltaSteppingSolverSorted::edge_type kind)
    {
        for (size_t j = start; j < end; ++j)
        {
            vertex_t v = R_vec[j];
            find_requests_vector(sol, v, kind, result);
        }
    }

    // NOTE: For the parallel version we need random index access, so we make R a vector instead of a set
    // This is the improvement from section 4 of the paper
    template <typename Iterator>
    std::vector<Edge> find_requests_parallel(Iterator begin, Iterator end, edge_type kind, size_t num_threads)
    {
        Benchmarker::start_one("find_requests_parallel");
        // Another copy... Slow
        Benchmarker::start_one("Copying");
        std::vector<vertex_t> R_vec(begin, end);
        Benchmarker::end_one("Copying");
        size_t chunk_size = R_vec.size() / num_threads;
        std::vector<std::vector<Edge>> results(num_threads - 1);
        std::vector<std::thread> threads(num_threads - 1);
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            threads[i] = std::thread(find_requests_thread, this, std::cref(R_vec), i * chunk_size, (i + 1) * chunk_size, std::ref(results[i]), kind);
        }
        size_t last_chunk_start = (num_threads - 1) * chunk_size;
        std::vector<Edge> res;
        // Push to res immediately to avoid unnecessary copying
        find_requests_thread(this, R_vec, last_chunk_start, R_vec.size(), res, kind);

        for (size_t i = 0; i < num_threads - 1; i++)
        {
            Benchmarker::start_one("Joining");
            threads[i].join();
            Benchmarker::end_one("Joining");
        }

        // NOTE: This combining might be quite slow. Is there a faster way to parallelize?
        for (size_t i = 0; i < num_threads - 1; i++)
        {
            Benchmarker::start_one("Copying");
            if (results[i].size() > 0)
                res.insert(res.end(), results[i].begin(), results[i].end());
            Benchmarker::end_one("Copying");
        }

        Benchmarker::end_one("find_requests_parallel");
        return std::move(res);
    }

    /*
     * Same as find_requests_parallel, but using OpenMP
     */
    template <typename Iterator>
    std::vector<Edge> find_requests_omp(Iterator begin, Iterator end, edge_type kind, size_t num_threads)
    {
        Benchmarker::start_one("find_requests_omp");
        std::vector<vertex_t> R_vec(begin, end);
        std::vector<std::vector<Edge>> results(num_threads);
#pragma omp parallel for shared(R_vec, results) num_threads(num_threads) schedule(dynamic)
        for (size_t j = 0; j < R_vec.size(); ++j)
        {
            vertex_t v = R_vec[j];
            size_t thread_id = omp_get_thread_num();
            find_requests_vector(this, v, kind, results[thread_id]);
        }

        std::vector<Edge> res;
        for (size_t i = 0; i < num_threads; ++i)
        {
            res.insert(res.end(), results[i].begin(), results[i].end());
        }

        Benchmarker::end_one("find_requests_omp");
        return std::move(res);
    }

    template <typename ReqFunc>
    void solve_base(vertex_t source, double delta, ReqFunc get_requests)
    {
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
                auto requests = get_requests(buckets->begin_of(i), buckets->end_of(i), edge_type::light);
                R.insert(buckets->begin_of(i), buckets->end_of(i));
                buckets->clear_at(i);
                relax_requests(requests);
            }
            auto requests = get_requests(R.begin(), R.end(), edge_type::heavy);
            relax_requests(requests);
        }
    }

    void relax_requests(const std::vector<Edge> &requests)
    {
        Benchmarker::start_one("relax_requests");
        for (const Edge &e : requests)
        {
            relax(e);
        }
        Benchmarker::end_one("relax_requests");
    }

    shortcut_array to_shortcut_array(const HashArray<UEdge, double> &found)
    {
        shortcut_array shortcuts_from;
        for (auto it = found.begin(); it != found.end(); it++)
        {
            vertex_t u = it->first.from;
            vertex_t v = it->first.to;
            double x = it->second;
            shortcuts_from[u].push_back(Edge({v, x}));
        }
        return shortcuts_from;
    }

    shortcut_array find_shortcuts_simple(vertex_t source)
    {
        Benchmarker::start_one("find_shortcuts_simple");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::unordered_set<DEdge> Q;
        // To have them in sorted order
        HashArray<UEdge, double> Q_dash(std::numeric_limits<double>::infinity());
        Q.insert(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Q_dash.clear();
            Benchmarker::start_one("Computing Q_dash");
            for (auto d_edge : Q)
            {
                size_t u = d_edge.from;
                size_t v = d_edge.to;
                double x = d_edge.weight;
                for (auto edge : semisorted_edges[v].first)
                {
                    size_t w = edge.to;
                    double c = edge.weight;
                    // For single thread, immediately insert the minimum to avoid
                    // storing the whole multiset
                    if (x + c < Q_dash.get(UEdge(u, w)))
                    {
                        Q_dash.insert(UEdge(u, w), x + c);
                    }
                }
            }
            Benchmarker::end_one("Computing Q_dash");

            Q.clear();
            for (auto it = Q_dash.begin(); it != Q_dash.end(); it++)
            {
                vertex_t u = it->first.from;
                vertex_t v = it->first.to;
                double x = it->second;
                // filter by delta <= x < found(u, v)
                if (x >= delta || x > found.get(UEdge(u, v)))
                {
                    continue;
                }

                Q.insert(DEdge(u, v, x));
                found.insert(UEdge(u, v), x);
            }
        }

        auto shortcuts_from = to_shortcut_array(found);
        Benchmarker::end_one("find_shortcuts_simple");
        return shortcuts_from;
    }

    static void find_shortcuts_thread(DeltaSteppingSolverSorted *sol, const std::vector<DEdge> &Q, size_t idx_start, size_t idx_end,
                                      HashArray<UEdge, double> &Q_dash, const HashArray<UEdge, double> &found)
    {
        for (size_t i = idx_start; i < idx_end; i++)
        {
            size_t u = Q[i].from;
            size_t v = Q[i].to;
            double x = Q[i].weight;
            for (auto edge : sol->semisorted_edges[v].first)
            {
                size_t w = edge.to;
                double c = edge.weight;
                auto edge_key = UEdge(u, w);
                // Break immediately because the edges are semi-sorted
                // NOTE: We are immediately doing the filtering of x <= delta and x < found
                // as well as immediately computing minimums on insertion to Q_dash.
                // It is ok to filter now, since the edges that do not satisfy these conditions
                // would not be inserted into Q_dash anyway
                // double cur_best = std::fmin(found.get(edge_key), Q_dash.get(edge_key));
                // if (x + c <= sol->delta && x + c < cur_best) {
                //     Q_dash.insert(edge_key, x + c);
                // }
                if (x + c > sol->delta)
                    break; 

                if (x + c <= std::fmin(found.get(edge_key), Q_dash.get(edge_key)))
                    Q_dash.insert(edge_key, x + c);
            }
        }
    }

    std::unordered_map<vertex_t, std::vector<Edge>> find_shortcuts_parallel(vertex_t source, size_t num_threads)
    {
        Benchmarker::start_one("find_shortcuts_parallel");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::vector<DEdge> Q;
        std::vector<HashArray<UEdge, double>> Q_dash(num_threads, HashArray<UEdge, double>(std::numeric_limits<double>::infinity()));
        Q.push_back(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Benchmarker::start_one("Computing Q_dash");
            size_t chunk_size = Q.size() / num_threads;
            // Make sure the extra nodes are spread evenly across threads
            // I am not sure this helps anyhow, but trying desperately to fix a bug...
            size_t extra_nodes = Q.size() - chunk_size * num_threads;
            size_t cur_begin = 0;
            size_t cur_end = chunk_size;
            std::vector<std::thread> threads(num_threads - 1);
            for (size_t i = 0; i < num_threads - 1; i++)
            {
                if (i < extra_nodes)
                    cur_end++;

                Q_dash[i].clear();
                threads[i] = std::thread(find_shortcuts_thread, this, std::cref(Q), cur_begin,
                                         cur_end, std::ref(Q_dash[i]),
                                         std::cref(found));
                cur_begin = cur_end;
                cur_end += chunk_size;
            }

            Q_dash[num_threads - 1].clear();
            if (cur_begin < Q.size())
                find_shortcuts_thread(this, Q, cur_begin, Q.size(), Q_dash[num_threads - 1], found);

            Benchmarker::start_one("Joining");
            for (size_t i = 0; i < num_threads - 1; i++)
                threads[i].join();
            Benchmarker::end_one("Joining");

            Benchmarker::end_one("Computing Q_dash");

            Q.clear();

            // Now compute min(Q_dash[i].get(u, v)) for all (u, v) and put it in found
            Benchmarker::start_one("Updating Q");
            HashArray<UEdge, bool> visited(false);
            for (size_t i = 0; i < num_threads; i++)
            {
                for (auto it = Q_dash[i].begin(); it != Q_dash[i].end(); it++)
                {
                    vertex_t u = it->first.from;
                    vertex_t v = it->first.to;
                    auto edge_key = UEdge(u, v);
                    if (visited.get(edge_key))
                        continue;

                    visited.insert(edge_key, true);
                    double min_val = it->second;
                    // We can start from i + 1, since {u, v} is not visited, hence Q_dash[j < i].get(u, v) == inf
                    for (size_t j = i + 1; j < num_threads; j++)
                        min_val = std::fmin(min_val, Q_dash[j].get(edge_key));

                    if (min_val < found.get(edge_key) && min_val <= delta)
                    {
                        found.insert(edge_key, min_val);
                        Q.emplace_back(DEdge(u, v, min_val));
                    }
                }
            }
            Benchmarker::end_one("Updating Q");
        }

        auto shortcuts_from = to_shortcut_array(found);
        Benchmarker::end_one("find_shortcuts_parallel");
        return shortcuts_from;
    }

    /*
     * Same as find_shortcuts_parallel, but using OpenMP instead of std::thread
     */
    shortcut_array find_shortcuts_omp(vertex_t source)
    {
        Benchmarker::start_one("find_shortcuts_omp");
        HashArray<UEdge, double> found(std::numeric_limits<double>::infinity());
        std::vector<DEdge> Q;
        // Immediately set the default to delta, since edges < delta are not interesting to us
        std::vector<HashArray<UEdge, double>> Q_dash(omp_get_max_threads(), HashArray<UEdge, double>(std::numeric_limits<double>::infinity()));
        Q.push_back(DEdge(source, source, 0));
        while (!Q.empty())
        {
            Benchmarker::start_one("Computing Q_dash");
#pragma omp parallel shared(Q, Q_dash)
            {
#pragma omp for schedule(dynamic)
                for (size_t i = 0; i < Q.size(); ++i)
                {
                    size_t thread_id = omp_get_thread_num();
                    for (auto edge : semisorted_edges[Q[i].to].first)
                    {
                        double new_cost = Q[i].weight + edge.weight;
                        auto edge_key = UEdge(Q[i].from, edge.to);
                        if (new_cost > delta)
                            break;

                        if (new_cost < std::fmin(Q_dash[thread_id].get(edge_key), found.get(edge_key)))
                            Q_dash[thread_id].insert(edge_key, new_cost);
                    }
                }
            }
            Benchmarker::end_one("Computing Q_dash");

            Q.clear();
            Benchmarker::start_one("Updating Q");
            HashArray<UEdge, bool> visited(false);
            for (size_t i = 0; i < Q_dash.size(); ++i)
            {
                for (auto it = Q_dash[i].begin(); it != Q_dash[i].end(); it++)
                {
                    vertex_t u = it->first.from;
                    vertex_t v = it->first.to;
                    auto edge_key = UEdge(u, v);
                    if (visited.get(edge_key))
                        continue;

                    visited.insert(edge_key, true);
                    double min_val = it->second;
                    for (size_t j = i + 1; j < Q_dash.size(); j++)
                        min_val = std::fmin(min_val, Q_dash[j].get(edge_key));

                    if (min_val < found.get(edge_key) && min_val <= delta)
                    {
                        found.insert(edge_key, min_val);
                        Q.emplace_back(DEdge(u, v, min_val));
                    }
                }

                Q_dash[i].clear();
            }
            Benchmarker::end_one("Updating Q");
        }

        Benchmarker::end_one("find_shortcuts_omp");
        return to_shortcut_array(found);
    }

    void solve_shortcuts_base(shortcut_array &shortcuts_from, vertex_t source)
    {
        distances = dist_vector(semisorted_edges.size(), std::numeric_limits<double>::infinity());
        buckets->clear();
        relax({source, 0});
        int i = -1;
        while (!buckets->empty())
        {
            auto opt_i = buckets->first_non_empty_bucket(i);
            if (!opt_i.has_value())
                break;

            i = opt_i.value();
            for (auto it = buckets->begin_of(i); it != buckets->end_of(i); it++)
            {
                vertex_t v = *it;
                // distances[v] + e.weight <= (i + 1) * delta <=> e.weight <= (i + 1) * delta - distances[v]
                double target_weight = (i + 1) * delta - distances[v];
                for (const Edge &e : graph.edges_from(v))
                {
                    if (e.weight <= target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
                for (const Edge &e : shortcuts_from[v])
                {
                    if (e.weight <= target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
            }
            for (auto it = buckets->begin_of(i); it != buckets->end_of(i); it++)
            {
                vertex_t v = *it;
                // distances[v] + e.weight > (i + 1) * delta <=> e.weight > (i + 1) * delta - distances[v]
                double target_weight = (i + 1) * delta - distances[v];
                for (const Edge &e : graph.edges_from(v))
                {
                    if (e.weight > target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
                for (const Edge &e : shortcuts_from[v])
                {
                    if (e.weight > target_weight)
                    {
                        relax({e.to, distances[v] + e.weight});
                    }
                }
            }
        }
    }

public:
    // Idk why I do it like this, architecture is weird...
    DeltaSteppingSolverSorted(const Graph &g, double delta) : graph(g)
    {
        buckets = std::make_unique<PrioritizedBucketList>();
        this->delta = delta;
        semisorted_edges = graph.semisort_delta_parallel(delta);
    }

    dist_vector solve(vertex_t source)
    {
        solve_base(source, this->delta, [this](auto begin, auto end, auto kind)
                   { return find_requests(begin, end, kind); });
        return distances;
    }

    // Same as above, but use the improvement from section 4 for faster requests findings
    dist_vector solve_parallel_simple(vertex_t source, size_t num_threads)
    {
        solve_base(source, this->delta, [this, num_threads](auto begin, auto end, auto kind)
                   { return find_requests_parallel(begin, end, kind, num_threads); });
        return distances;
    }

    dist_vector solve_parallel_omp(vertex_t source, size_t num_threads)
    {
        solve_base(source, this->delta, [this, num_threads](auto begin, auto end, auto kind)
                   { return find_requests_omp(begin, end, kind, num_threads); });
        return distances;
    }

    dist_vector solve_shortcuts(vertex_t source)
    {
        auto shortcuts_from = find_shortcuts_simple(source);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }

    dist_vector solve_shortcuts_parallel(vertex_t source, size_t num_threads)
    {
        auto shortcuts_from = find_shortcuts_parallel(source, num_threads);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }

    dist_vector solve_shortcuts_omp(vertex_t source)
    {
        auto shortcuts_from = find_shortcuts_omp(source);
        solve_shortcuts_base(shortcuts_from, source);
        return distances;
    }
};