#pragma once

#include "core/logger.h"
#include "core/types.h"
#include "nn.hpp"

namespace gate_net {
struct TrainingData {
    f32 input[2];
    f32 target;
};

constexpr TrainingData and_gate[4] = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 0.0f},
    {{1.0f, 0.0f}, 0.0f},
    {{1.0f, 1.0f}, 1.0f},
};

constexpr TrainingData or_gate[4] = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 1.0f},
};

constexpr TrainingData xor_gate[4] = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f},
};

struct ModelConfig {
    f32 learning_rate;
    usize *layer_sizes;
    TrainingData *training_data;
    usize epochs;
};

void test_network(Network &network, TrainingData *training_data, usize training_data_count);

void run(ModelConfig config);
}; // namespace gate_net
