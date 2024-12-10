#include "nn.hpp"

#include "core/file.h"
#include "core/logger.h"
#include "core/types.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <ratio>
#include <string>

f32 sigmoid(f32 x) {
    return 1.0f / (1.0f + expf(-x));
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
    f32 weight_scale = sqrtf(2.0f / (layer.input_size + layer.output_size));
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
    usize total_size = sizeof(usize) + sizeof(f32); // For the layer_count and learning_rate
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

    // Metadata
    memcpy(buffer, &network.layer_count, sizeof(usize));
    buffer += sizeof(usize);

    memcpy(buffer, &network.learning_rate, sizeof(f32));
    buffer += sizeof(f32);

    // Layer sizes
    for (usize i = 0; i < network.layer_count; i++) {
        memcpy(buffer, &network.layers[i].input_size, sizeof(usize));
        buffer += sizeof(usize);

        memcpy(buffer, &network.layers[i].output_size, sizeof(usize));
        buffer += sizeof(usize);
    }

    // Layer data
    for (usize i = 0; i < network.layer_count; i++) {
        serialize(network.layers[i], &buffer);
    }

    return original_buffer;
}

void deserialize(Network &network, char *buffer) {
    char *original_buffer = buffer;

    // Read metadata
    memcpy(&network.layer_count, buffer, sizeof(usize));
    buffer += sizeof(usize);

    memcpy(&network.learning_rate, buffer, sizeof(f32));
    buffer += sizeof(f32);

    // Allocate layers
    network.layers = static_cast<Layer *>(malloc(sizeof(Layer) * network.layer_count));

    // Read layer sizes and initialize layers
    for (usize i = 0; i < network.layer_count; i++) {
        memcpy(&network.layers[i].input_size, buffer, sizeof(usize));
        buffer += sizeof(usize);

        memcpy(&network.layers[i].output_size, buffer, sizeof(usize));
        buffer += sizeof(usize);

        init(network.layers[i], network.layers[i].input_size, network.layers[i].output_size);
    }

    // Read layer data
    for (usize i = 0; i < network.layer_count; i++) {
        deserialize(network.layers[i], &buffer);
    }
}

bool save(Network &network, const char *path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        error("Failed to open file: %s", path);
        return false;
    }

    char *data = serialize(network);
    usize len = calculate_total_serialization_size(network);

    file.write(data, len);
    file.close();
    free(data);

    return true;
}

bool load(Network &network, const char *path) {
    char *data;
    isize bytes_read = read_file(path, &data);

    if (bytes_read < 0) {
        return false;
    }

    usize first_value;
    memcpy(&first_value, data, sizeof(usize));

    if (bytes_read < sizeof(usize) + sizeof(f32)) { // Minimum size
        error("File too small");
        free(data);
        return false;
    }

    deserialize(network, data);
    free(data);
    return true;
}

struct MomentumOptimizerData {
    f32 momentum;
    f32 *weight_velocity;
    f32 *bias_velocity;
};

struct RMSPropOptimizerData {
    f32 decay_rate;
    f32 epsilon;
    f32 *weight_cache;
    f32 *bias_cache;
};

struct AdamOptimizerData {
    f32 beta1;
    f32 beta2;
    f32 epsilon;
    usize t;
    f32 *m_weights;
    f32 *m_biases;
    f32 *v_weights;
    f32 *v_biases;
};

struct Optimizer {
    enum Kind { SGD, Momentum, RMSProp, Adam };

    Kind kind;
    f32 learning_rate;
    union {
        MomentumOptimizerData momentum;
        RMSPropOptimizerData rms_prop;
        AdamOptimizerData adam;
    } data;
};

void init_sgd_optimizer(Optimizer &opt, f32 learning_rate) {
    opt.kind = Optimizer::SGD;
    opt.learning_rate = learning_rate;
}

void init_momentum_optimizer(Optimizer &opt, Layer &layer, f32 learning_rate, f32 momentum) {
    const usize weight_size = layer.input_size * layer.output_size;
    const usize bias_size = layer.output_size;

    f32 *storage = static_cast<f32 *>(calloc(weight_size + bias_size, sizeof(f32)));

    opt.kind = Optimizer::Momentum;
    opt.learning_rate = learning_rate;
    opt.data.momentum = {
        .momentum = momentum,
        .weight_velocity = storage,
        .bias_velocity = storage + weight_size,
    };
}

void init_rmsprop_optimizer(Optimizer &opt, Layer &layer, f32 learning_rate, f32 decay_rate = 0.9f,
                            f32 epsilon = 1e-8f) {
    const usize weight_size = layer.input_size * layer.output_size;
    const usize bias_size = layer.output_size;

    f32 *storage = static_cast<f32 *>(calloc(weight_size + bias_size, sizeof(f32)));

    opt.kind = Optimizer::RMSProp;
    opt.learning_rate = learning_rate;
    opt.data.rms_prop = {
        .decay_rate = decay_rate,
        .epsilon = epsilon,
        .weight_cache = storage,
        .bias_cache = storage + weight_size,
    };
}

void init_adam_optimizer(Optimizer &opt, Layer &layer, f32 learning_rate, f32 beta1 = 0.9f, f32 beta2 = 0.999f,
                         f32 epsilon = 1e-8f) {
    const usize weight_size = layer.input_size * layer.output_size;
    const usize bias_size = layer.output_size;

    f32 *storage = static_cast<f32 *>(calloc((weight_size + bias_size) * 2, sizeof(f32)));

    opt.kind = Optimizer::Adam;
    opt.learning_rate = learning_rate;
    opt.data.adam = {
        .beta1 = beta1,
        .beta2 = beta2,
        .epsilon = epsilon,
        .t = 0,
        .m_weights = storage,
        .m_biases = storage + weight_size,
        .v_weights = storage + weight_size + bias_size,
        .v_biases = storage + (weight_size * 2) + bias_size,
    };
}

void deinit(Optimizer &opt) {
    switch (opt.kind) {
    // case Optimizer::SGD:
    //     break;
    case Optimizer::Momentum:
        free(reinterpret_cast<void *>(opt.data.momentum.weight_velocity));
        break;
    case Optimizer::RMSProp:
        free(reinterpret_cast<void *>(opt.data.rms_prop.weight_cache));
        break;
    case Optimizer::Adam:
        free(reinterpret_cast<void *>(opt.data.adam.m_weights));
        break;
    default:
        break;
    };
}

void update_weights_sdg(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size);
void update_weights_momentum(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size);
void update_weights_rms_prop(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size);
void update_weights_adam(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size);

void update_weights(Optimizer &opt, Layer &layer) {
    const usize weight_size = layer.input_size * layer.output_size;
    const usize bias_size = layer.output_size;

    switch (opt.kind) {
    case Optimizer::SGD:
        update_weights_sdg(opt, layer, weight_size, bias_size);
        break;
    case Optimizer::Momentum:
        update_weights_momentum(opt, layer, weight_size, bias_size);
        break;
    case Optimizer::RMSProp:
        update_weights_rms_prop(opt, layer, weight_size, bias_size);
        break;
    case Optimizer::Adam:
        update_weights_adam(opt, layer, weight_size, bias_size);
        break;
    default:
        break;
    }
}

void update_weights_sdg(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size) {
    for (usize i = 0; i < weight_size; i++) {
        f32 gradient = layer.deltas[i % layer.output_size] * layer.inputs[i / layer.output_size];
        layer.weights[i] -= gradient * opt.learning_rate;
    }

    for (usize i = 0; i < bias_size; i++) {
        layer.biases[i] -= layer.deltas[i] * opt.learning_rate;
    }
}

void update_weights_momentum(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size) {
    const MomentumOptimizerData &data = opt.data.momentum;
    for (usize i = 0; i < weight_size; i++) {
        f32 &velocity = data.weight_velocity[i];
        f32 gradient = layer.deltas[i % layer.output_size] * layer.inputs[i / layer.output_size];
        velocity = velocity * data.momentum - gradient * opt.learning_rate;
        layer.weights[i] += velocity;
    }

    for (usize i = 0; i < bias_size; i++) {
        f32 &velocity = data.bias_velocity[i];
        velocity = velocity * data.momentum - layer.deltas[i] * opt.learning_rate;
        layer.biases[i] += velocity;
    }
}

void update_weights_rms_prop(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size) {
    const RMSPropOptimizerData &data = opt.data.rms_prop;
    for (usize i = 0; i < weight_size; i++) {
        f32 &weight = data.weight_cache[i];
        f32 gradient = layer.deltas[i % layer.output_size] * layer.inputs[i / layer.output_size];
        weight = weight * data.decay_rate + gradient * gradient * (1 - data.decay_rate);
        layer.weights[i] -= gradient * opt.learning_rate / (sqrtf(weight) + data.epsilon);
    }

    for (usize i = 0; i < bias_size; i++) {
        f32 &bias = data.bias_cache[i];
        f32 delta = layer.deltas[i];
        bias = bias * data.decay_rate + delta * delta * (1 - data.decay_rate);
        layer.biases[i] -= layer.deltas[i] * opt.learning_rate / (sqrtf(bias) + data.epsilon);
    }
}

void update_weights_adam(Optimizer &opt, Layer &layer, usize weight_size, usize bias_size) {
    AdamOptimizerData &data = opt.data.adam;

    data.t++;
    const f32 alpha = sqrtf(1 - powf(data.beta2, data.t)) * opt.learning_rate / (1 - powf(data.beta1, data.t));

    for (usize i = 0; i < weight_size; i++) {
        f32 &m_weight = data.m_weights[i];
        f32 &v_weight = data.v_weights[i];
        f32 gradient = layer.deltas[i % layer.output_size] * layer.inputs[i / layer.output_size];
        m_weight = m_weight * data.beta1 + (1 - data.beta1) * gradient;
        v_weight = v_weight * data.beta2 + (1 - data.beta2) * gradient * gradient;
        layer.weights[i] = m_weight * alpha / (sqrtf(v_weight) * data.epsilon);
    }

    for (usize i = 0; i < bias_size; i++) {
        f32 &m_bias = data.m_biases[i];
        f32 &v_bias = data.v_biases[i];
        f32 delta = layer.deltas[i];
        m_bias = m_bias * data.beta1 + (1 - data.beta1) * delta;
        v_bias = v_bias * data.beta2 + (1 - data.beta2) * delta * delta;
        layer.biases[i] = m_bias * alpha / (sqrtf(v_bias) * data.epsilon);
    }
}
