#pragma once

#include <cstdint>
#include <string>
#include "memory/bytes.hpp"
#include "memory/encoding.hpp"

class cvmat_bgr_image_t;

class generic_image_t {
    friend class cvmat_bgr_image_t;
    friend class memory::Encoding<generic_image_t>;

public:
    using size_type = uint32_t;
    enum FORMAT : size_type {
        NONE,
        BGR,
        JPG,
    };
    struct metadata_t {
        FORMAT format;
        union {
            struct {
                size_type width;
                size_type height;
            } bgr_info;
        };
    };

    metadata_t metadata;
    memory::Bytes data;

    generic_image_t() {
        metadata.format = NONE;
    }

    generic_image_t(const metadata_t &meta, memory::Bytes &&data) : metadata(meta), data(std::move(data)) {
        auto result = validate();
        if (!result.empty()) {
            throw std::invalid_argument(result);
        }
    }

    std::string validate() {
        return "";
    }
};

namespace memory {

template <>
class Encoding<generic_image_t> {
public:
    static std::list<memory::Bytes> encode(generic_image_t &&data) {
        auto list = memory::Encoding<generic_image_t::metadata_t>::encode(std::move(data.metadata));
        list.emplace_back(std::move(data.data));
        return list;
    }

    static void decode(std::list<memory::Bytes> &list, generic_image_t &output) {
        memory::Encoding<generic_image_t::metadata_t>::decode(list, output.metadata);

        if (list.empty()) {
            throw std::out_of_range("");
        }
        output.data = std::move(list.front());
        list.pop_front();
    }
};

}  // namespace memory