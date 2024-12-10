#include "core/types.h"
#include "nn.hpp"

#include <cmath>

enum ActivationKind {
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Elu,
    SoftMax,
    Gelu,
};

f32 sigmoid(f32 x) {
    return 1.0f / (1.0f + expf(-x));
}

f32 sigmoid_derivative(f32 x) {
    f32 s = sigmoid(x);
    return s * (1.0f - s);
}

f32 tanh_derivative(f32 x) {
    f32 t = tanhf(x);
    return 1.0f - t * t;
}

f32 relu(f32 x) {
    return x > 0.0f ? x : 0.0f;
}

f32 relu_derivative(f32 x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

f32 leaky_relu(f32 x, f32 alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

f32 leaky_relu_derivative(f32 x, f32 alpha = 0.01f) {
    return x > 0.0f ? 1.0f : alpha;
}

f32 elu(f32 x, f32 alpha = 1.0f) {
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

f32 elu_derivative(f32 x, f32 alpha = 1.0f) {
    return x > 0.0f ? 1.0f : alpha * expf(x);
}

f32 gelu(f32 x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

f32 gelu_derivative(f32 x) {
    const f32 cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    const f32 pdf = exp(-0.5f * x * x) / sqrtf(2.0f * M_PI);
    return cdf + x * pdf;
}

void softmax(f32 *input, f32 *output, usize size) {
    f32 max_val = input[0];

    if (size > 1) {
        for (usize i = 1; i < size; i++) {
            max_val = fmax(input[i], max_val);
        }
    }

    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (usize i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void softmax_derivative(f32 *output, f32 *jacobian, usize size) {
    for (usize i = 0; i < size; i++) {
        for (usize j = 0; j < size; j++) {
            if (i == j) {
                jacobian[i * size + j] = output[i] * (1.0f - output[i]);
            } else {
                jacobian[i * size + j] = -output[i] * output[j];
            }
        }
    }
}
