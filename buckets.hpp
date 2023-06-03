#pragma once

#include <optional>
#include <unordered_set>
#include <vector>
#include <map>

#include "graph.hpp"

using bucket_t = std::unordered_set<vertex_t>;

class BucketListBase {
private: 
    size_t total_elements;
public:
    BucketListBase() {
        total_elements = 0;
    }
    ~BucketListBase() {}

    size_t size() const {
        return total_elements;
    }
    bool empty() {
        return size() == 0;
    }

    virtual std::optional<size_t> first_non_empty_bucket(int min_i = -1) const {}

    virtual void insert(size_t bucket_idx, const vertex_t& v) {
        ++total_elements;
    }
    virtual void erase(size_t bucket_idx, const vertex_t& v) {
        if (total_elements == 0) {
            throw std::runtime_error("BucketListBase::erase: empty bucket list");
        }
        --total_elements;
    }
    virtual void clear_at(size_t bucket_idx) {
        total_elements -= size_of(bucket_idx);
    }
    virtual void clear() {
        total_elements = 0;
    }

    virtual size_t size_of(size_t bucket_idx) const {}
    virtual bucket_t::const_iterator begin_of(size_t bucket_idx) const {}
    virtual bucket_t::const_iterator end_of(size_t bucket_idx) const {}
};

/*
Just the simplest possible implementation of a bucket list: 
A simple vector of unordered_set of fixed size (no cyclical reusage)
*/
class SimpleBucketList : public BucketListBase {
private:
    std::vector<bucket_t> buckets;
public:
    const size_t DEFAULT_BUCKETS = 10000;

    SimpleBucketList() : BucketListBase() {
        buckets = std::vector<bucket_t>(DEFAULT_BUCKETS);
    }

    std::optional<size_t> first_non_empty_bucket(int min_i = -1) const override {
        for (int i = min_i + 1; i < buckets.size(); i++)
        {
            if (!buckets[i].empty())
            {
                return i;
            }
        }
        return std::nullopt;
    }

    void insert(size_t bucket_idx, const vertex_t& v) override {
        BucketListBase::insert(bucket_idx, v);
        buckets[bucket_idx].insert(v);
    }
    void erase(size_t bucket_idx, const vertex_t& v) override {
        BucketListBase::erase(bucket_idx, v);
        buckets[bucket_idx].erase(v);
    }
    void clear_at(size_t bucket_idx) override {
        BucketListBase::clear_at(bucket_idx);
        buckets[bucket_idx].clear();
    }
    void clear() override {
        BucketListBase::clear();
        buckets.clear();
    }

    size_t size_of(size_t bucket_idx) const override {
        return buckets[bucket_idx].size();
    }
    bucket_t::const_iterator begin_of(size_t bucket_idx) const override {
        return buckets[bucket_idx].begin();
    }
    bucket_t::const_iterator end_of(size_t bucket_idx) const override {
        return buckets[bucket_idx].end();
    }
};

/*
An attempt to mimic the priority queue implementation of dijkstra.cpp 
i.e get O(n log n) time, with minimal memory usage 
*/
class PrioritizedBucketList: public BucketListBase {
private:
    std::map<size_t, bucket_t> buckets;
public: 
    PrioritizedBucketList() : BucketListBase() {}

    std::optional<size_t> first_non_empty_bucket(int min_i = -1) const override {
        auto it = buckets.lower_bound(min_i + 1);
        if (it == buckets.end()) {
            return std::nullopt;
        }

        return it->first;
    }

    void insert(size_t bucket_idx, const vertex_t& v) override {
        BucketListBase::insert(bucket_idx, v);
        buckets[bucket_idx].insert(v);
    }

    void erase(size_t bucket_idx, const vertex_t& v) override {
        BucketListBase::erase(bucket_idx, v);
        buckets[bucket_idx].erase(v);
        if (buckets[bucket_idx].empty()) {
            buckets.erase(bucket_idx);
        }
    }

    void clear_at(size_t bucket_idx) override {
        BucketListBase::clear_at(bucket_idx);
        // TODO: Do we need this ?
        buckets[bucket_idx].clear();
        buckets.erase(bucket_idx);
    }

    void clear() override {
        BucketListBase::clear();
        buckets.clear();
    }

    size_t size_of(size_t bucket_idx) const override {
        if (!buckets.count(bucket_idx)) {
            return 0;
        }

        return buckets.at(bucket_idx).size();
    }

    bucket_t::const_iterator begin_of(size_t bucket_idx) const override {
        return buckets.at(bucket_idx).begin();
    }

    bucket_t::const_iterator end_of(size_t bucket_idx) const override {
        return buckets.at(bucket_idx).end();
    }
};