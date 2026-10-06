#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

enum class DType {
    FP32,
    FP16,
    BF16,
    INT32,
    INT64,
    UINT8
};

enum class Device {
    CPU,
    CUDA
};

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);
const char* device_name(Device device);

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape, DType dtype, Device device);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    static Tensor zeros(const std::vector<int>& shape, DType dtype, Device device);
    static Tensor empty(const std::vector<int>& shape, DType dtype, Device device);

    Tensor to(Device target_device) const;
    void copy_from(const Tensor& src);

    void* data_ptr();
    const void* data_ptr() const;

    float* data_as_float();
    const float* data_as_float() const;

    int dim() const;
    int size(int d) const;
    const std::vector<int>& shape() const;
    const std::vector<int>& strides() const;
    int64_t num_elements() const;
    size_t size_bytes() const;
    DType dtype() const;
    Device device() const;
    bool is_cuda() const;
    bool is_cpu() const;
    bool is_valid() const;

    Tensor reshape(const std::vector<int>& new_shape) const;

    void print_info(const std::string& label = "") const;
    void print_data(int max_elements = 20) const;

private:
    void allocate();
    void free_memory();
    void compute_strides();

    void* data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int64_t num_elements_;
    DType dtype_;
    Device device_;
    bool owns_data_;
};
