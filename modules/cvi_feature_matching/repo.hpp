#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include "optional.hpp"

template <size_t DIM>
class Repo {
public:
    typedef std::vector<float> feature_t;
    virtual ~Repo() {}

    struct match_result {
        bool matched;
        size_t id;
        double score;
    };

    virtual size_t add(const std::string &name, const feature_t &features) = 0;
    virtual void build() = 0;
    virtual match_result match(const feature_t &features, double threshold) = 0;
    virtual size_t size() const = 0;
    virtual std::vector<size_t> id_list() const = 0;
    virtual mp::optional<std::pair<std::string, feature_t>> get(size_t id) const = 0;
    virtual mp::optional<std::string> get_name(size_t id) const = 0;
    std::string get_match_name(const match_result &match) {
        return match.matched ? get_name(match.id).value_or("") : "";
    }
};

template <size_t DIM>
class MutableRepo : public Repo<DIM> {
public:
    using typename Repo<DIM>::feature_t;
    virtual ~MutableRepo() {}

    virtual double compute(size_t id, const feature_t &features) = 0;
    virtual void set(size_t id, const std::string &name, const feature_t &features) = 0;
    virtual void remove(size_t id) = 0;
    virtual void remove_all() = 0;
};