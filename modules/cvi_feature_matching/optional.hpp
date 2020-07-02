#pragma once

#include <exception>
#include <tuple>

namespace mp {

struct nullopt_t {};

struct inplace_t {};

class bad_optional_access : public std::exception {
public:
    using std::exception::exception;
};

template <typename... Ts>
class optional {
    template <size_t Index>
    using element_type = typename std::tuple_element<Index, std::tuple<Ts...>>::type;

public:
    optional() : has_value_(false) {}
    optional(nullopt_t) : has_value_(false) {}
    optional(const optional &other) : has_value_(false) {
        if (other.has_value_) {
            emplace(other.value_);
        }
    }
    optional(optional &&other) : has_value_(false) {
        if (other.has_value_) {
            emplace(std::move(other.value_));
            other.reset();
        }
    }
    template <typename... Us>
    optional(inplace_t, Us &&...us) : has_value_(false) {
        emplace(std::forward<Us>(us)...);
    }

    ~optional() {
        reset();
    }

    optional &operator=(optional &&other) {
        reset();
        if (other.has_value_) {
            emplace(std::move(other.value_));
            other.reset();
        }
    }
    optional &operator=(const optional &other) {
        reset();
        if (other.has_value_) {
            emplace(other.value_);
        }
    }

    bool operator==(const optional &other) const {
        if (has_value_ != other.has_value_) {
            return false;
        }

        return !has_value_ || value_ == other.value_;
    }

    bool has_value() const {
        return has_value_;
    }

    template <size_t Index>
    const element_type<Index> &value() const {
        if (!has_value_) {
            throw bad_optional_access();
        }
        return std::get<Index>(value_);
    }

    template <size_t Index>
    element_type<Index> &value() {
        if (!has_value_) {
            throw bad_optional_access();
        }
        return std::get<Index>(value_);
    }

    template <size_t Index>
    element_type<Index> value_or(element_type<Index> fallback) const {
        if (has_value_) {
            return std::get<Index>(value_);
        } else {
            return fallback;
        }
    }

    void reset() {
        if (has_value_) {
            using T = std::tuple<Ts...>;
            value_.~T();
            has_value_ = false;
        }
    }

    void set(std::tuple<Ts...> &&input) {
        reset();
        has_value_ = true;
        using T = std::tuple<Ts...>;
        ::new (&value_) T(std::move(input));
    }

    template <typename... Us>
    void emplace(Us &&...us) {
        reset();
        has_value_ = true;
        using T = std::tuple<Ts...>;
        ::new (&value_) T(std::forward<Us>(us)...);
    }

private:
    bool has_value_;
    union {
        std::tuple<Ts...> value_;
    };
};

/**
 * mp::optional manages an optional contained value, i.e. a value that may or may not be present.
 */
template <typename T>
class optional<T> {
public:
    /// Constructs an object that does not contain a value.
    optional() : has_value_(false) {}
    /// Constructs an object that does not contain a value.
    optional(nullopt_t) : has_value_(false) {}
    optional(const optional &other) : has_value_(false) {
        if (other.has_value_) {
            emplace(other.value_);
        }
    }
    optional(optional &&other) : has_value_(false) {
        if (other.has_value_) {
            emplace(std::move(other.value_));
            other.reset();
        }
    }
    /**
     * Constructs an optional object that contains a value, initialized as if arguments (except the first) is passed to constructor of contained type.
     */
    template <typename... Us>
    optional(inplace_t, Us &&...us) : has_value_(false) {
        emplace(std::forward<Us>(us)...);
    }

    ~optional() {
        reset();
    }

    optional &operator=(optional &&other) {
        reset();
        if (other.has_value_) {
            emplace(std::move(other.value_));
            other.reset();
        }
    }
    optional &operator=(const optional &other) {
        reset();
        if (other.has_value_) {
            emplace(other.value_);
        }
    }

    /**
     * @return true if both object do not contain values or both contain values and the values are equal
     */
    bool operator==(const optional &other) const {
        if (has_value_ != other.has_value_) {
            return false;
        }

        return !has_value_ || value_ == other.value_;
    }

    /**
     * @return true if this object contains a value
     */
    bool has_value() const {
        return has_value_;
    }

    /**
     * a shortcut for accessing members of the contained value
     * @return the pointer to contained value
     * @exception bad_optional_access if no value present
     */
    const T *operator->() const {
        if (!has_value_) {
            throw bad_optional_access();
        }
        return &value_;
    }

    /**
     * a shortcut for accessing members of the contained value
     * @return the pointer to contained value
     * @exception bad_optional_access if no value present
     */
    T *operator->() {
        if (!has_value_) {
            throw bad_optional_access();
        }
        return &value_;
    }

    /**
     * @return a reference to contained value
     * @exception bad_optional_access if no value present
     */
    template <size_t Index = 0>
    const T &value() const {
        static_assert(Index == 0, "index out of range");
        if (!has_value_) {
            throw bad_optional_access();
        }
        return value_;
    }

    /**
     * @return a reference to contained value
     * @exception bad_optional_access if no value present
     */
    template <size_t Index = 0>
    T &value() {
        static_assert(Index == 0, "index out of range");
        if (!has_value_) {
            throw bad_optional_access();
        }
        return value_;
    }

    /**
     * @return a copy of the contained value, return first argument if no value present
     */
    template <size_t Index = 0>
    T value_or(T fallback) const {
        static_assert(Index == 0, "index out of range");
        if (has_value_) {
            return value_;
        } else {
            return fallback;
        }
    }

    /**
     * destruct the contained value, set the optional object to empty state
     */
    void reset() {
        if (has_value_) {
            value_.~T();
            has_value_ = false;
        }
    }

    void set(std::tuple<T> &&input) {
        reset();
        has_value_ = true;
        ::new (&value_) T(std::move(std::get<0>(input)));
    }

    /**
     * destruct the contained value.
     * construct a new instance with arugments
     */
    template <typename... Us>
    void emplace(Us &&...us) {
        reset();
        has_value_ = true;
        ::new (&value_) T(std::forward<Us>(us)...);
    }

private:
    bool has_value_;
    union {
        T value_;
    };
};

}