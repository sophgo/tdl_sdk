#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <type_traits>
#include <cstring>
#include <ostream>

namespace memory {

/**
 * \class memory::Bytes
 * memory::Bytes is a class that manages arbitrary binary data and frees managed space automatically.
 * Copied instances of Bytes may share the same memory space.
 * \code{.cpp}
 * vector<char> char_vector;
 * Bytes from_char_vector(move(char_vector));
 *
 * vector<int> int_vector;
 * Bytes from_int_vector(move(int_vector));
 *
 * // assume the matrix is continuous
 * cv::Mat image;
 * Bytes from_cv_mat(move(image), image.ptr(0), image.elemSize() * image.total());
 *
 * \endcode
 *
 */
    class Bytes {
        class RAIIBase {
        public:
            virtual ~RAIIBase() {}
        };

        template <typename T>
        class RAII : public RAIIBase {
        public:
            template <typename U>
            RAII(U &&obj) : object(std::forward<U>(obj)) {}
            ~RAII() {}

            T object;
        };

        /// A shared_ptr managing a object that frees managed memory space on destruction.
        std::shared_ptr<RAIIBase> resource;
        /// A ptr pointing to the managed memory space.
        void *ptr;
        /// The size of the managed memory space in bytes.
        size_t length;

    public:
        struct copy_t {};

        /// Create an instance that manages no memory space.
        Bytes() {
            ptr = nullptr;
            length = 0;
        }

        /// Copy the input POD object to allocated space and make the Bytes instance manage that memory space.
        template <typename T>
        Bytes(copy_t, const T &data) {
            static_assert(std::is_pod<typename std::decay<T>::type>::value, "T must be POD");
            auto raii = std::make_shared<RAII<T>>(data);
            ptr = &(raii->object);
            length = sizeof(T);
            resource = raii;
        }

        /**
         * Create a Bytes instance that manages memory space pointed by ptr of which length is given.
         * The memory space will be freed when the moved obj is destructed.
         */
        template <typename T>
        Bytes(T &&obj, void *ptr, size_t length) : resource(std::make_shared<RAII<T>>(std::forward<T>(obj))), ptr(ptr), length(length) {}

        /// Create a Bytes instance that manages memory space allocated by the vector
        template <typename T>
        Bytes(std::vector<T> &&v) {
            auto raii = std::make_shared<RAII<std::vector<T>>>(std::move(v));
            ptr = raii->object.data();
            length = sizeof(T) * raii->object.size();
            resource = raii;
        }

        Bytes(const Bytes &other) = default;
        Bytes(Bytes &&other) = default;

        Bytes &operator=(const Bytes &other) = default;
        Bytes &operator=(Bytes &&other) = default;

        /**
         * @return a pointer pointed to the managed memory space.
         */
        void *data() {
            if (ptr == nullptr) {
                throw std::out_of_range("this memory::Bytes is empty");
            }
            return ptr;
        }

        /**
         * @return a pointer pointed to the managed memory space.
         */
        void *data() const {
            if (ptr == nullptr) {
                throw std::out_of_range("this memory::Bytes is empty");
            }
            return ptr;
        }

        /**
         * @return the size of the managed memory space (in bytes).
         */
        size_t size() const {
            return length;
        }

        /**
         * Split the memory space managed by the current Bytes instance into two.
         * First slice_length bytes of memory space will be managed by the returned Bytes instance.
         * The remaining memory space will be managed by this instance.
         * The whole memory space will be freed after both instances and other instances created from these two instances.
         * @param slice_length the length of the first Bytes after splitting
         * @return a Bytes instance
         */
        Bytes slice_front(size_t slice_length) {
            if (length < slice_length) {
                throw std::out_of_range("slice_length given is greater than length of Bytes");
            }
            Bytes slice;
            slice.resource = resource;
            slice.ptr = ptr;
            slice.length = slice_length;

            ptr = reinterpret_cast<char*>(ptr) + slice_length;
            length -= slice_length;

            return slice;
        }
    };

/**
 * \class memory::Iterable
 * memory::Iterable is a helper class that allows iterating through data managed by memory::Bytes instance as a continuous sequence of POD data.
 * @param T a POD data type
 *
 * \code{.cpp}
 * vector<int> int_vector;
 * Bytes from_int_vector(move(int_vector));
 * Iterable<int> int_iterable(move(from_int_vector));
 *
 * // iterates with for loop
 * for (int i = 0; i < int_iterable.size(); i++) {
 *     printf("%d ", int_iterable[i]);
 * }
 *
 * // iterates with for-each loop
 * for (auto n : int_iterable) {
 *     printf("%d ", n);
 * }
 *
 * \endcode
 */
    template <typename T>
    class Iterable {
        static_assert(std::is_pod<T>::value, "T must be POD");
    public:
        /// create an empty Iterable object
        Iterable() {}
        /// create an Iterable that iterates data managed by the memory::Bytes instance
        Iterable(const Bytes &other) : data(other) {}
        /// create an Iterable that iterates data managed by the memory::Bytes instance
        Iterable(Bytes &&other) : data(std::move(other)) {}
        Iterable(const Iterable &) = default;
        Iterable(Iterable &&) = default;

        /// create an Iterable that iterates data managed by the memory::Bytes instance
        Iterable &operator=(Bytes &&other) {
            data = std::move(other);
            return *this;
        }
        Iterable &operator=(Iterable &&other) {
            data = std::move(other.data);
            return *this;
        }

        T *begin() {
            return reinterpret_cast<T*>(data.data());
        }
        const T *begin() const {
            return reinterpret_cast<const T*>(data.data());
        }

        T *end() {
            return reinterpret_cast<T*>(data.data()) + size();
        }
        const T *end() const {
            return reinterpret_cast<T*>(data.data()) + size();
        }

        /**
         * returns the index-th element in the Iterable instance.
         * \exception out_of_range
         */
        T &operator[](size_t index) {
            if (index >= size()) {
                throw std::out_of_range("index is out of range");
            }
            return *(begin() + index);
        }

        /**
         * returns the index-th element in the Iterable instance.
         * \exception out_of_range
         */
        const T &operator[](size_t index) const {
            if (index >= size()) {
                throw std::out_of_range("index is out of range");
            }
            return *(begin() + index);
        }

        /// @return the number of elements in the Iterable instance
        size_t size() const {
            return data.size() / sizeof(T);
        }

        /// @return the underlying Bytes instance and take ownership from this instance
        Bytes release() {
            Bytes ret(std::move(data));
            return ret;
        }

    private:
        /// Bytes instance managing the underlying memory space
        Bytes data;
    };

    /// helper function that prints the length and first 8 bytes in the memory space.
    inline std::ostream &operator<<(std::ostream &s, const Bytes &b) {
        s << "b" << b.size() << "[";
        for (size_t i = 0; i < b.size(); i++) {
            if (i >= 8) {
                s << "...";
                break;
            }
            unsigned char ch = *(reinterpret_cast<unsigned char*>(b.data()) + i);
            if (32 <= ch && ch < 127) {
                s << ch;
            } else {
                s << "\\x" << std::hex << static_cast<int>(ch);
            }
        }
        s << "]";
        return s;
    }

} // namespace memory
