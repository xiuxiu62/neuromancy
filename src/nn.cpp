#include "nn.hpp"

#include "core/logger.h"
#include "core/types.h"

#include <cmath>
#include <cstdlib>
#include <limits>

f32 sigmoid(f32 x) {
    return 1.0f / (1.0f + std::expf(-x));
}

f32 sigmoid_derivative(f32 x) {
    f32 s = sigmoid(x);
    return s * (1 - s);
}

// LAYER
void init(Layer &layer, usize input_size, usize output_size) {
    const usize weight_storage_size = sizeof(f32) * (input_size * output_size);
    const usize bias_storage_size = sizeof(f32) * output_size;
    const usize output_storage_size = bias_storage_size;
    const usize delta_storage_size = output_storage_size;
    const usize input_storage_size = sizeof(f32) * input_size;
    const usize total_storage_size =
        weight_storage_size + bias_storage_size + output_storage_size + delta_storage_size + input_storage_size;

    f32 *storage = static_cast<f32 *>(malloc(total_storage_size));
    layer = {
        .input_size = input_size,
        .output_size = output_size,
        .weights = storage,
        .biases = storage + weight_storage_size / sizeof(f32),
        .output = storage + (weight_storage_size + bias_storage_size) / sizeof(f32),
        .deltas = storage + (weight_storage_size + bias_storage_size + output_storage_size) / sizeof(f32),
        .inputs = storage +
                  (weight_storage_size + bias_storage_size + output_storage_size + delta_storage_size) / sizeof(f32),
    };
}

void deinit(Layer &layer) {
    free(layer.weights);
}

void prime(Layer &layer) {
    const usize bias_storage_size = sizeof(f32) * layer.output_size;
    const usize output_storage_size = bias_storage_size;
    const usize delta_storage_size = output_storage_size;

    // Randomize weights between -1 and 1
    f32 weight_scale = std::sqrtf(2.0f / (layer.input_size + layer.output_size));
    for (usize i = 0; i < layer.input_size * layer.output_size; i++) {
        layer.weights[i] = (static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX) * 2.0f - 1.0f) * weight_scale;
    }

    // Zero out remainder
    memset(reinterpret_cast<void *>(layer.biases), 0, bias_storage_size);
    memset(reinterpret_cast<void *>(layer.output), 0, output_storage_size);
    memset(reinterpret_cast<void *>(layer.deltas), 0, delta_storage_size);
}

void forward(Layer &layer, f32 *input) {
    // Store input for backprop
    memcpy(layer.inputs, input, sizeof(f32) * layer.input_size);

    for (usize i = 0; i < layer.output_size; i++) {
        f32 sum = layer.biases[i];
        for (usize j = 0; j < layer.input_size; j++) {
            sum += input[j] * layer.weights[j * layer.output_size + i];
        }

        layer.output[i] = sigmoid(sum);
    }
}

void backward(Layer &layer, f32 *next_layer_deltas, f32 learning_rate) {
    // Calculate deltas for current layer
    if (next_layer_deltas) {
        for (usize i = 0; i < layer.output_size; i++) {
            layer.deltas[i] = next_layer_deltas[i] * sigmoid_derivative(layer.output[i]);
        }
    }

    // Update weights and biases
    for (usize i = 0; i < layer.output_size; i++) {
        for (usize j = 0; j < layer.input_size; j++) {
            usize weight_id = j * layer.output_size + i;
            layer.weights[weight_id] -= learning_rate * layer.deltas[i] * layer.inputs[j];
        }

        layer.biases[i] -= learning_rate * layer.deltas[i];
    }
}

void serialize(Layer &layer, char **buffer) {
    memcpy(*buffer, &layer.input_size, sizeof(usize));
    *buffer += sizeof(usize);

    memcpy(*buffer, &layer.output_size, sizeof(usize));
    *buffer += sizeof(usize);

    usize weight_size = sizeof(f32) * layer.input_size * layer.output_size;
    memcpy(*buffer, layer.weights, weight_size);

    *buffer += weight_size;
    usize bias_size = sizeof(f32) * layer.output_size;

    memcpy(*buffer, layer.biases, bias_size);
    *buffer += bias_size;
}

void deserialize(Layer &layer, char **buffer) {
    usize weight_size = sizeof(f32) * layer.input_size * layer.output_size;
    memcpy(layer.weights, *buffer, weight_size);
    *buffer += weight_size;

    usize bias_size = sizeof(f32) * layer.output_size;
    memcpy(layer.biases, *buffer, bias_size);
    *buffer += bias_size;
}

// NETWORK
void init(Network &network, usize layer_count, usize *layer_sizes, f32 learning_rate = 0.1f) {
    network = {
        .layers = static_cast<Layer *>(malloc(sizeof(Layer) * (layer_count - 1))),
        .layer_count = layer_count - 1,
        .learning_rate = learning_rate,
    };

    for (usize i = 0; i < layer_count - 1; i++) {
        init(network.layers[i], layer_sizes[i], layer_sizes[i + 1]);
    }
}

void deinit(Network &network) {
    for (usize i = 0; i < network.layer_count; i++) {
        deinit(network.layers[i]);
    }
    free(network.layers);
}

void prime(Network &network) {
    for (usize i = 0; i < network.layer_count; i++) {
        prime(network.layers[i]);
    }
}

void forward(Network &network, f32 *input) {
    for (usize i = 0; i < network.layer_count; i++) {
        forward(network.layers[i], input);
        input = network.layers[i].output;
    }
}

void backward(Network &network) {
    for (isize i = network.layer_count - 1; i >= 0; i--) {
        f32 *deltas = nullptr;

        if (i < network.layer_count - 1) {
            deltas = network.layers[i + 1].deltas;
        }

        backward(network.layers[i], deltas, network.learning_rate);
    }
}

void train(Network &network, f32 *input, f32 *target) {
    forward(network, input);

    // Calculate output layer deltas
    Layer &output_layer = network.layers[network.layer_count - 1];

    for (usize i = 0; i < output_layer.output_size; i++) {
        f32 output = output_layer.output[i];
        output_layer.deltas[i] = sigmoid_derivative(output) * (output - target[i]);
    }

    backward(network);
}

f32 calculate_loss(Network &network, f32 *target) {
    Layer &output_layer = network.layers[network.layer_count - 1];
    f32 loss = 0.0f;

    for (usize i = 0; i < output_layer.output_size; i++) {
        f32 error = target[i] - output_layer.output[i];
        loss += error * error;
    }

    return loss * 0.5f;
}

usize calculate_total_serialization_size(Network &network) {
    usize total_size = sizeof(usize); // For the number of layers
    for (usize i = 0; i < network.layer_count; ++i) {
        total_size += 2 * sizeof(usize); // For input and output sizes
        total_size += network.layers[i].input_size * network.layers[i].output_size * sizeof(f32); // For weights
        total_size += network.layers[i].output_size * sizeof(f32);                                // For biases
    }

    return total_size;
}

char *serialize(Network &network) {
    char *original_buffer = static_cast<char *>(malloc(calculate_total_serialization_size(network)));
    char *buffer = original_buffer;

    memcpy(buffer, &network.layer_count, sizeof(usize));
    buffer += sizeof(usize);

    memcpy(buffer, &network.learning_rate, sizeof(f32));
    buffer += sizeof(f32);

    for (usize i = 0; i < network.layer_count; i++) {
        memcpy(buffer, &network.layers[i].input_size, sizeof(usize));
        buffer += sizeof(usize);

        memcpy(buffer, &network.layers[i].output_size, sizeof(usize));
        buffer += sizeof(usize);
    }

    for (usize i = 0; i < network.layer_count; i++) {
        info("[%lld] Serializing layer:%zu", buffer - original_buffer, i);
        serialize(network.layers[i], &buffer);
    }

    return original_buffer;
}

void deserialize(Network &network, char *buffer) {
    char *original_buffer = buffer;

    memcpy(&network.layer_count, buffer, sizeof(usize));
    buffer += sizeof(usize);

    memcpy(&network.learning_rate, buffer, sizeof(f32));
    buffer += sizeof(f32);

    network.layers = static_cast<Layer *>(malloc(sizeof(Layer) * network.layer_count));

    // init(network, 3, config.layer_sizes, config.learning_rate);

    for (usize i = 0; i < network.layer_count; i++) {
        memcpy(&network.layers[i].input_size, buffer, sizeof(usize));
        buffer += sizeof(usize);

        memcpy(&network.layers[i].output_size, buffer, sizeof(usize));
        buffer += sizeof(usize);

        init(network.layers[i], network.layers[i].input_size, network.layers[i].output_size);
    }

    for (usize i = 0; i < network.layer_count; ++i) {
        info("[%lld] Deserializing layer:%zu", buffer - original_buffer, i);
        deserialize(network.layers[i], &buffer);
    }
}
