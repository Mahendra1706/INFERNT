#include "tensor.h"
#include "cuda_utils.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32:  return 4;
        case DType::FP16:  return 2;
        case DType::BF16:  return 2;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::UINT8: return 1;
    }
    return 0;
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FP32:  return "FP32";
        case DType::FP16:  return "FP16";
        case DType::BF16:  return "BF16";
        case DType::INT32: return "INT32";
        case DType::INT64: return "INT64";
        case DType::UINT8: return "UINT8";
    }
    return "UNKNOWN";
}

const char* device_name(Device device) {
    switch (device) {
        case Device::CPU:  return "CPU";
        case Device::CUDA: return "CUDA";
    }
    return "UNKNOWN";
}

Tensor::Tensor()
    : data_(nullptr), num_elements_(0), dtype_(DType::FP32),
      device_(Device::CPU), owns_data_(false) {}

Tensor::Tensor(const std::vector<int>& shape, DType dtype, Device device)
    : data_(nullptr), shape_(shape), dtype_(dtype), device_(device),
      owns_data_(true) {
    num_elements_ = 1;
    for (int s : shape_) {
        if (s <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
        num_elements_ *= s;
    }
    compute_strides();
    allocate();
}

Tensor::~Tensor() {
    free_memory();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), num_elements_(other.num_elements_),
      dtype_(other.dtype_), device_(other.device_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.num_elements_ = 0;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_memory();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        num_elements_ = other.num_elements_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.num_elements_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor Tensor::zeros(const std::vector<int>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    if (device == Device::CPU) {
        std::memset(t.data_, 0, t.size_bytes());
    } else {
        CUDA_CHECK(cudaMemset(t.data_, 0, t.size_bytes()));
    }
    return t;
}

Tensor Tensor::empty(const std::vector<int>& shape, DType dtype, Device device) {
    return Tensor(shape, dtype, device);
}

Tensor Tensor::to(Device target_device) const {
    if (target_device == device_) {
        throw std::runtime_error("Tensor is already on the target device");
    }

    Tensor result(shape_, dtype_, target_device);

    if (device_ == Device::CPU && target_device == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(result.data_, data_, size_bytes(),
                              cudaMemcpyHostToDevice));
    } else if (device_ == Device::CUDA && target_device == Device::CPU) {
        CUDA_CHECK(cudaMemcpy(result.data_, data_, size_bytes(),
                              cudaMemcpyDeviceToHost));
    }

    return result;
}

void Tensor::copy_from(const Tensor& src) {
    if (num_elements_ != src.num_elements_ || dtype_ != src.dtype_) {
        throw std::runtime_error("Tensor shape/dtype mismatch in copy_from");
    }

    cudaMemcpyKind kind;
    if (device_ == Device::CPU && src.device_ == Device::CPU) {
        std::memcpy(data_, src.data_, size_bytes());
        return;
    } else if (device_ == Device::CUDA && src.device_ == Device::CPU) {
        kind = cudaMemcpyHostToDevice;
    } else if (device_ == Device::CPU && src.device_ == Device::CUDA) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    CUDA_CHECK(cudaMemcpy(data_, src.data_, size_bytes(), kind));
}

void* Tensor::data_ptr() { return data_; }
const void* Tensor::data_ptr() const { return data_; }

float* Tensor::data_as_float() {
    if (dtype_ != DType::FP32) {
        throw std::runtime_error("Tensor is not FP32");
    }
    return static_cast<float*>(data_);
}

const float* Tensor::data_as_float() const {
    if (dtype_ != DType::FP32) {
        throw std::runtime_error("Tensor is not FP32");
    }
    return static_cast<const float*>(data_);
}

int Tensor::dim() const { return static_cast<int>(shape_.size()); }
int Tensor::size(int d) const { return shape_.at(d); }
const std::vector<int>& Tensor::shape() const { return shape_; }
const std::vector<int>& Tensor::strides() const { return strides_; }
int64_t Tensor::num_elements() const { return num_elements_; }

size_t Tensor::size_bytes() const {
    return static_cast<size_t>(num_elements_) * dtype_size(dtype_);
}

DType Tensor::dtype() const { return dtype_; }
Device Tensor::device() const { return device_; }
bool Tensor::is_cuda() const { return device_ == Device::CUDA; }
bool Tensor::is_cpu() const { return device_ == Device::CPU; }
bool Tensor::is_valid() const { return data_ != nullptr && num_elements_ > 0; }

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int64_t new_elements = 1;
    for (int s : new_shape) {
        new_elements *= s;
    }
    if (new_elements != num_elements_) {
        throw std::runtime_error("Reshape: element count mismatch");
    }

    Tensor result;
    result.data_ = data_;
    result.shape_ = new_shape;
    result.num_elements_ = num_elements_;
    result.dtype_ = dtype_;
    result.device_ = device_;
    result.owns_data_ = false;
    result.compute_strides();
    return result;
}

void Tensor::print_info(const std::string& label) const {
    if (!label.empty()) {
        printf("%s: ", label.c_str());
    }
    printf("Tensor(shape=[");
    for (int i = 0; i < dim(); i++) {
        if (i > 0) printf(", ");
        printf("%d", shape_[i]);
    }
    printf("], dtype=%s, device=%s, elements=%ld, bytes=%zu)\n",
           dtype_name(dtype_), device_name(device_),
           (long)num_elements_, size_bytes());
}

void Tensor::print_data(int max_elements) const {
    if (!is_cpu()) {
        printf("[Tensor on GPU — transfer to CPU to print data]\n");
        return;
    }
    if (dtype_ != DType::FP32) {
        printf("[print_data only supports FP32 currently]\n");
        return;
    }

    const float* d = data_as_float();
    int count = std::min(static_cast<int>(num_elements_), max_elements);
    printf("[");
    for (int i = 0; i < count; i++) {
        if (i > 0) printf(", ");
        printf("%.6f", d[i]);
    }
    if (count < num_elements_) {
        printf(", ... (%ld more)", (long)(num_elements_ - count));
    }
    printf("]\n");
}

void Tensor::allocate() {
    size_t bytes = size_bytes();
    if (bytes == 0) return;

    if (device_ == Device::CPU) {
        data_ = malloc(bytes);
        if (!data_) {
            throw std::runtime_error("CPU memory allocation failed");
        }
    } else {
        cudaError_t err = cudaMalloc(&data_, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU allocation failed for %zu bytes: %s\n",
                    bytes, cudaGetErrorString(err));
            throw std::runtime_error("GPU memory allocation failed");
        }
    }
}

void Tensor::free_memory() {
    if (data_ && owns_data_) {
        if (device_ == Device::CPU) {
            ::free(data_);
        } else {
            cudaFree(data_);
        }
    }
    data_ = nullptr;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}
