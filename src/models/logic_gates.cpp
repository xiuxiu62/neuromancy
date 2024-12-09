#include "gate_net.hpp"

#include <cstring>

namespace gate_net {

void test_network(Network &network, TrainingData *training_data, usize training_data_count) {
    for (usize i = 0; i < training_data_count; i++) {
        const TrainingData data = training_data[i];
        f32 input[2];
        f32 target = data.target;

        memcpy(input, data.input, sizeof(f32) * 2);

        forward(network, input);
        f32 prediction = network.layers[network.layer_count - 1].output[0];
        info("Input: [%.1f, %.1f] -> Expected: %lld -> Predicted: %lld", input[0], input[1], (u64)target,
             std::llroundf(prediction));
    }
}

void run(ModelConfig config) {
    Network network;
    init(network, 3, config.layer_sizes, config.learning_rate);

    for (usize epoch = 0; epoch < config.epochs; epoch++) {
        for (usize i = 0; i < 4; i++) {
            auto &example = config.training_data[i];

            train(network, example.input, &example.target);
        }
        if (epoch % 100 == 0) {
            f32 total_loss = 0;
            for (usize i = 0; i < 4; i++) {
                auto &example = config.training_data[i];

                forward(network, example.input);
                total_loss += calculate_loss(network, &example.target);
            }
            // info("Epoch %zu: Loss = %f", epoch, total_loss / 4);
        }
    }

    test_network(network, config.training_data, 4);
}
}; // namespace gate_net
