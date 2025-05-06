#ifndef PACKET_HPP
#define PACKET_HPP

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>

class PacketBase {
 public:
  virtual ~PacketBase() {}
  virtual std::type_index getTypeIndex() const = 0;
  virtual const void* getRawPtr() const = 0;
};

template <typename T>
class TypedPacket : public PacketBase {
 public:
  explicit TypedPacket(const T& val) : value_(val) {}
  std::type_index getTypeIndex() const override { return typeid(T); }
  const void* getRawPtr() const override { return &value_; }
  const T& get() const { return value_; }

 private:
  T value_;
};

class Packet {
 public:
  Packet() {}

  template <typename T>
  static Packet make(const T& value) {
    Packet pkt;
    pkt.data_.reset(new TypedPacket<T>(value));
    return pkt;
  }
  template <typename T>
  bool is() const {
    return data_ && data_->getTypeIndex() == typeid(T);
  }
  template <typename T>
  const T& get() const {
    if (!data_) throw std::runtime_error("Packet is empty.");
    if (data_->getTypeIndex() != typeid(T))
      throw std::runtime_error("Packet type mismatch.");
    return static_cast<const TypedPacket<T>*>(data_.get())->get();
  }

 private:
  std::shared_ptr<PacketBase> data_;
};

#endif
