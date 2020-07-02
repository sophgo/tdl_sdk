#pragma once

#include "repo.hpp"
#include <sstream>
#include <cmath>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <map>
#include "path.hpp"
#include "file_stat.hpp"

template <size_t DIM>
class CachedRepo : public MutableRepo<DIM> {
public:
    using typename MutableRepo<DIM>::feature_t;
    static constexpr size_t dim = DIM;

protected:
    size_t next_id_ = 1;
    std::string path_;
    bool isDequantized_;
    std::map<size_t, std::pair<std::string, feature_t>> faces_;

public:
    explicit CachedRepo(const std::string &path, const bool isDequantized = true) :
        path_(path),
        isDequantized_(isDequantized) {
        mkpath(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        load();
    }

    CachedRepo(CachedRepo &&rhs)
        : next_id_(rhs.next_id_),
        path_(std::move(rhs.path_)),
        isDequantized_(rhs.isDequantized_),
        faces_(std::move(rhs.faces_)) {}

    virtual ~CachedRepo() {}

    size_t add(const std::string &name, const feature_t &features) override {
        std::cout << "CachedRepo::add " << name << std::endl;
        size_t id = next_id_++;
        faces_[id] = std::make_pair(name, features);
        write_features(face_file_name(id), features);
        write_info(id, name);

        return id;
    }

    void build() override {};

    typename Repo<DIM>::match_result match(const feature_t &features, double threshold) override {
        typename Repo<DIM>::match_result best{false, 0, 0};
        // std::cout << "face cnt = " << faces_.size() << std::endl;
        for (auto &pair : faces_) {
            double score = compute_score(pair.second.second, features);
            std::cout << "score = "<<score << std::endl;
            if (score > best.score) {
                best.score = score;
                best.id = pair.first;
                if (score >= threshold) {
                    best.matched = true;
                }
            }
        }

        return best;
    }

    double compute(size_t id, const feature_t &features) override {
        auto iter = faces_.find(id);
        if (iter == faces_.end()) {
            return 0;
        }

        return compute_score(iter->second.second, features);
    }

    double compute1v1(const feature_t &features1, const feature_t &features2) {
        return compute_score(features1, features2);
    }

    size_t size() const override {
        return faces_.size();
    }

    std::vector<size_t> id_list() const override {
        std::vector<size_t> ids;
        ids.reserve(faces_.size());
        for (auto &pair : faces_) {
            ids.emplace_back(pair.first);
        }
        return ids;
    }

    void set(size_t id, const std::string &name, const feature_t &features) override {
        faces_[id] = std::make_pair(name, features);
        write_info(id, name);
        write_features(face_file_name(id), features);
    }

    mp::optional<std::pair<std::string, feature_t>> get(size_t id) const override {
        mp::optional<std::pair<std::string, feature_t>> result;
        auto iter = faces_.find(id);
        if (iter != faces_.end()) {
            result.emplace(iter->second);
        }
        return result;
    }

    mp::optional<std::string> get_name(size_t id) const override {
        mp::optional<std::string> result;
        auto iter = faces_.find(id);
        if (iter != faces_.end()) {
            result.emplace(iter->second.first);
        }
        return result;
    }

    void remove(size_t id) override {
        faces_.erase(id);
        ::utils::file::remove(face_file_name(id));
        ::utils::file::remove(info_file_name(id));
    }

    void remove_all() override {
        for (auto &pair : faces_) {
            ::utils::file::remove(face_file_name(pair.first));
            ::utils::file::remove(info_file_name(pair.first));
        }
        faces_.clear();
    }

private:
    std::string face_file_name(size_t id) const {
        return path_ + "/" + std::to_string(id) + ".face";
    }

    std::string info_file_name(size_t id) const {
        return path_ + "/" + std::to_string(id) + ".info";
    }

    void load() {
        DIR *dir = opendir(path_.data());
        dirent *ent;

        if (dir == nullptr) {
            return;
        }

        std::vector<std::string> legacy_entries;

        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            if ((filename.length() > 5) && (filename.substr(filename.length() - 5) == ".face")) {
                std::string name = filename.substr(0, filename.length() - 5);
                size_t id = atoi(name.data());

                if (name == std::to_string(id) && ::utils::file::is_regular(path_ + "/" + name + ".info")) {
                    next_id_ = std::max<size_t>(id + 1, next_id_);

                    auto info_content = read_info(path_ + "/" + name + ".info");
                    auto features = read_features(path_ + "/" + filename);
                    faces_[id] = std::make_pair(std::move(info_content), std::move(features));
                } else {
                    legacy_entries.emplace_back(name);
                }
            }
        }

        closedir(dir);

        handle_legacy(legacy_entries);
    }

    void handle_legacy(const std::vector<std::string> &entries) {
        for (auto &name : entries) {
            size_t id = next_id_++;
            while (::utils::file::exists(face_file_name(id)) || ::utils::file::exists(info_file_name(id))) {
                id = next_id_++;
            }

            rename((path_ + "/" + name + ".face").data(), face_file_name(id).data());
            write_info(id, name);

            auto features = read_features(id);
            faces_[id] = std::make_pair(name, std::move(features));
        }
    }

    feature_t read_features(size_t id) {
        return read_features(face_file_name(id));
    }

    feature_t read_features(const std::string &filename) {
        feature_t features;
        float tmp;
        FILE *fp = fopen(filename.data(), "r");
        if (fp != nullptr) {
            while (fscanf(fp, "%f", &tmp) == 1) {
                features.push_back(tmp);
            }
        }
        fclose(fp);

        return features;
    }

    void write_features(const std::string &filename, const feature_t &features) {
        FILE *fp = fopen(filename.data(), "w");
        if (fp != nullptr) {
            for (const float &value : features) {
                if (isDequantized_) {
                    fprintf(fp, "%f\n", value);
                } else {
                    fprintf(fp, "%d\n", int(value));
                }
            }
            fclose(fp);
        }
    }

    void write_info(int id, const std::string &name) {
        std::ofstream fstream(info_file_name(id));
        fstream << name << std::endl;
        fstream.close();
    }

    std::string read_info(int id) {
        return read_info(info_file_name(id));
    }

    std::string read_info(const std::string &filename) {
        std::stringbuf name;
        std::ifstream fstream(filename);

        fstream.get(name, '\n');
        fstream.close();

        return name.str();
    }

    double compute_score(const feature_t &lhs, const feature_t &rhs) const {
        double lhs_sq_mag = 0;
        double rhs_sq_mag = 0;
        double cdot = 0;
        for (size_t i = 0; i < std::min<size_t>(lhs.size(), rhs.size()); i++) {
            cdot += lhs[i] * rhs[i];
            lhs_sq_mag += lhs[i] * lhs[i];
            rhs_sq_mag += rhs[i] * rhs[i];
        }
        if (lhs_sq_mag == 0 || rhs_sq_mag == 0) {
            return 0;
        }

        return cdot / sqrt(lhs_sq_mag) / sqrt(rhs_sq_mag);
    }
};
