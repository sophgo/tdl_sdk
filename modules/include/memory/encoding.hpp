#pragma once

#include <type_traits>
#include <vector>
#include <string>
#include <tuple>
#include <list>
#include "bytes.hpp"

namespace memory {

/**
 * memory::Encoding specifies how various types are serialized into and deserialized from segments of binary data.
 * Each element can be serialized into one or more segments of data.
 * The primary template of memory::Encoding handles serialization of POD types, which simply copies memory content of POD objects.
 */
template <typename T>
class Encoding {
    static_assert(std::is_pod<T>::value, "encoding of non-POD types must be specialized");

public:
    static std::list<Bytes> encode(T &&data) {
        std::vector<char> v(sizeof(T));
        memcpy(v.data(), &data, sizeof(T));
        return {Bytes(std::move(v))};
    }

    static void decode(std::list<Bytes> &list, T &output) {
        if (list.empty()) {
            throw std::out_of_range("");
        }
        if (list.front().size() != sizeof(T)) {
            throw std::invalid_argument("length of segment: is different from size of T");
        }
        memcpy(&output, list.front().data(), sizeof(T));
        list.pop_front();
    }
};

template <>
class Encoding<std::string> {
public:
    static std::list<Bytes> encode(std::string &&data) {
        return {Bytes(std::vector<char>(data.begin(), data.end()))};
    }

    static void decode(std::list<Bytes> &list, std::string &output) {
        if (list.empty()) {
            throw std::out_of_range("");
        }
        Iterable<char> iter(std::move(list.front()));
        list.pop_front();

        output.assign(iter.begin(), iter.end());
    }
};

template <typename T>
class Encoding<Iterable<T>> {
public:
    static std::list<Bytes> encode(Iterable<T> &&data) {
        return {data.release()};
    }

    static void decode(std::list<Bytes> &list, Iterable<T> &output) {
        if (list.empty()) {
            throw std::out_of_range("");
        }

        output = std::move(list.front());
        list.pop_front();
    }
};

template <size_t Index, typename... Ts>
class TupleEncoding {
    using element_type = typename std::tuple_element<Index - 1, std::tuple<Ts...>>::type;
public:
    static void encode(std::list<Bytes> &list, std::tuple<Ts...> &data) {
        TupleEncoding<Index - 1, Ts...>::encode(list, data);
        list.splice(list.end(), Encoding<element_type>::encode(std::move(std::get<Index - 1>(data))));
    }
    static void decode(std::list<Bytes> &list, std::tuple<Ts...> &output) {
        TupleEncoding<Index - 1, Ts...>::decode(list, output);
        Encoding<element_type>::decode(list, std::get<Index - 1>(output));
    }
};

template <typename... Ts>
class TupleEncoding<0, Ts...> {
public:
    static void encode(std::list<Bytes> &list, std::tuple<Ts...> &data) {
    }
    static void decode(std::list<Bytes> &list, std::tuple<Ts...> &output) {
    }
};

template <typename... Ts>
class Encoding<std::tuple<Ts...>> {
public:
    static std::list<Bytes> encode(std::tuple<Ts...> &&data) {
        std::list<Bytes> list;
        TupleEncoding<sizeof...(Ts), Ts...>::encode(list, data);
        return list;
    }
    static void decode(std::list<Bytes> &list, std::tuple<Ts...> &output) {
        TupleEncoding<sizeof...(Ts), Ts...>::decode(list, output);
    }
};

/**
 * Encoding of a vector of POD types, the whole vector is serialized into a data segment.
 */
template <typename T, bool IsPOD>
class EncodeVector;

template <typename T>
class EncodeVector<T, true> {
public:
    static std::list<Bytes> encode(std::vector<T> &&data) {
        return {Bytes(std::move(data))};
    }

    static void decode(std::list<Bytes> &list, std::vector<T> &output) {
        if (list.empty()) {
            throw std::out_of_range("");
        }
        Iterable<T> objects(std::move(list.front()));
        list.pop_front();
        output.resize(objects.size());
        memcpy(output.data(), objects.begin(), sizeof(T) * output.size());
    }
};

/**
 * Encoding of a vector of non-POD types, each element in the vector is serialized into a data segment, preceeding a segment indicates the number of elements.
 */
template <typename T>
class EncodeVector<T, false> {
public:
    static std::list<Bytes> encode(std::vector<T> &&data) {
        std::list<Bytes> result = {Bytes(Bytes::copy_t(), static_cast<uint32_t>(data.size()))};
        for (auto &ele : data) {
            result.splice(result.end(), Encoding<T>::encode(std::move(ele)));
        }
        return result;
    }

    static void decode(std::list<Bytes> &list, std::vector<T> &output) {
        if (list.empty()) {
            throw std::out_of_range("");
        }
        size_t size = *Iterable<uint32_t>(std::move(list.front())).begin();
        list.pop_front();
        output.resize(size);
        for (int i = 0; i < size; i++) {
            Encoding<T>::decode(list, output[i]);
        }
    }
};

template <typename T>
class Encoding<std::vector<T>> {
public:
    static std::list<Bytes> encode(std::vector<T> &&data) {
        return EncodeVector<T, std::is_pod<T>::value>::encode(std::move(data));
    }

    static void decode(std::list<Bytes> &list, std::vector<T> &output) {
        EncodeVector<T, std::is_pod<T>::value>::decode(list, output);
    }
};

}