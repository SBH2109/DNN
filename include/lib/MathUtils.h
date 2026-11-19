#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);

// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

template <typename T>
std::vector<double> normalize(const std::vector<T>& values) {
    if (values.empty()) {
        return {};
    }

    const T maxValue = *std::max_element(values.begin(), values.end());
    if (maxValue == 0) {
        throw std::invalid_argument("maxValue must be greater than 0.");
    }

    std::vector<double> valuesNormalized(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        valuesNormalized[i] = static_cast<double>(values[i]) / static_cast<double>(maxValue);
    }
    return valuesNormalized;
}

template <typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue) {
    if (maxValue == 0) {
        throw std::invalid_argument("maxValue must be greater than 0.");
    }

    std::vector<double> valuesNormalized(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        valuesNormalized[i] = static_cast<double>(values[i]) / static_cast<double>(maxValue);
    }
    return valuesNormalized;
}
}
