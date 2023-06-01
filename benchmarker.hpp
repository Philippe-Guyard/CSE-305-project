#pragma once 

#include <iostream>
#include <chrono>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

/* 
The purpose of this class is to measure the performance of random 
pieces of code and map those to a string, to later produce a summary 
of execution time for every string. For example 
Benchmark::start_one("foo"); 
{
    // code to benchmark
}
Benchmark::end_one("foo");
...
Benchmarker::print_summary(std::cout);
*/

using duration_t = long long;

class Benchmarker {
private:
    // map key -> vector of durations
    static std::unordered_map<std::string, std::vector<std::chrono::nanoseconds>> durations;
    static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> pending_starts;

    static void print_duration(std::ostream &os, duration_t duration) {
        if (duration < 10000ll) {
            // nanoseconds 
            os << duration << "ns";
            return;
        }
        duration_t duration_us = duration / 1000ll;
        if (duration_us < 10000ll) {
            // microseconds
            os << duration_us << "us";
            return;
        }
        duration_t duration_ms = duration_us / 1000ll;
        if (duration_ms < 10000ll) {
            // milliseconds
            os << duration_ms << "ms";
            return;
        }
        duration_t duration_s = duration_ms / 1000ll;
         // seconds
        os << duration_s << "s";
    }

    static std::unordered_map<std::string, duration_t> compute_totals() {
        std::unordered_map<std::string, duration_t> totals;
        for (auto& [key, durations] : durations) {
            duration_t total = 0;
            for (auto& duration : durations) {
                total += duration.count();
            }
            totals[key] = total;
        }
        return totals;
    }

    static std::unordered_map<std::string, duration_t> compute_counts() {
        std::unordered_map<std::string, duration_t> counts;
        for (auto& [key, durations] : durations) {
            counts[key] = durations.size();
        }
        return counts;
    }

    static std::unordered_map<std::string, std::pair<duration_t, duration_t>> compute_min_max() {
        std::unordered_map<std::string, std::pair<duration_t, duration_t>> min_maxes;
        for (auto& [key, durations] : durations) {
            duration_t min = durations[0].count();
            duration_t max = durations[0].count();
            for (auto& duration : durations) {
                if (duration.count() < min) {
                    min = duration.count();
                }
                if (duration.count() > max) {
                    max = duration.count();
                }
            }
            min_maxes[key] = std::make_pair(min, max);
        }
        return min_maxes;
    }
public:
    static void print_summary(std::ostream& os) {
        std::unordered_map<std::string, duration_t> totals = compute_totals();
        std::unordered_map<std::string, duration_t> counts = compute_counts();
        std::unordered_map<std::string, std::pair<duration_t, duration_t>> min_maxes = compute_min_max();
        for (auto& [key, total] : totals) {
            os << key << ": ";
            print_duration(os, total);
            os << " (" << counts[key] << " runs, min: ";
            print_duration(os, min_maxes[key].first);
            os << ", max: ";
            print_duration(os, min_maxes[key].second);
            os << ")\n";
        }
    }

    static void start_one(const std::string& key) {
        pending_starts[key] = std::chrono::steady_clock::now();
    }

    static void end_one(const std::string& key) {
        auto start = pending_starts[key];
        auto end = std::chrono::steady_clock::now();
        durations[key].push_back(end - start);
        pending_starts.erase(key);
    }

    static void clear() {
        durations.clear();
        pending_starts.clear();
    }
};

std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> Benchmarker::pending_starts;
std::unordered_map<std::string, std::vector<std::chrono::nanoseconds>> Benchmarker::durations;