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

class Benchmarker {
private:
    // map key -> vector of durations
    static std::unordered_map<std::string, std::vector<std::chrono::duration<double>>> durations;
    static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> pending_starts;

    static std::unordered_map<std::string, double> compute_totals() {
        std::unordered_map<std::string, double> totals;
        for (auto& [key, durations] : durations) {
            double total = 0;
            for (auto& duration : durations) {
                total += duration.count();
            }
            totals[key] = total;
        }
        return totals;
    }

    static std::unordered_map<std::string, double> compute_counts() {
        std::unordered_map<std::string, double> maxes;
        for (auto& [key, durations] : durations) {
            double max = 0;
            for (auto& duration : durations) {
                if (duration.count() > max) {
                    max = duration.count();
                }
            }
            maxes[key] = max;
        }
        return maxes;
    }

    static std::unordered_map<std::string, std::pair<double, double>> compute_min_max() {
        std::unordered_map<std::string, std::pair<double, double>> min_maxes;
        for (auto& [key, durations] : durations) {
            double min = durations[0].count();
            double max = durations[0].count();
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
        std::unordered_map<std::string, double> totals = compute_totals();
        std::unordered_map<std::string, double> counts = compute_counts();
        std::unordered_map<std::string, std::pair<double, double>> min_maxes = compute_min_max();
        for (auto& [key, total] : totals) {
            os << key << ": " << total << "s (" << counts[key] << " runs, min: " << min_maxes[key].first << "s, max: " << min_maxes[key].second << "s)\n";
        }
    }

    static void start_one(const std::string& key) {
        pending_starts[key] = std::chrono::steady_clock::now();
    }

    static void end_one(const std::string& key) {
        auto start = pending_starts[key];
        auto end = std::chrono::steady_clock::now();
        auto duration = end - start;
        durations[key].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(duration));
        pending_starts.erase(key);
    }

    static void clear() {
        durations.clear();
        pending_starts.clear();
    }
};

std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> Benchmarker::pending_starts;
std::unordered_map<std::string, std::vector<std::chrono::duration<double>>> Benchmarker::durations;