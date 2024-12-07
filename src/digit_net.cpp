#include "core/logger.h"
#include "digit_net.hpp"
#include "nn.hpp"
#include <corecrt_malloc.h>

namespace digit_net {

void test_network(Network &network, Digit *training_data, usize training_data_count) {
    for (usize i = 0; i < training_data_count; i++) {
        Digit &example = training_data[i];
        f32 pixels[Digit::PIXEL_COUNT];
        memcpy(pixels, example.pixels, sizeof(example.pixels));
        forward(network, pixels);

        // Find predicted digit (highest output activation)
        usize predicted = 0;
        f32 max_activation = network.layers[network.layer_count - 1].output[0];
        for (usize j = 1; j < Digit::DIGIT_COUNT; j++) {
            if (network.layers[network.layer_count - 1].output[j] > max_activation) {
                max_activation = network.layers[network.layer_count - 1].output[j];
                predicted = j;
            }
        }

        // Find actual digit from label
        usize actual = 0;
        for (usize j = 0; j < Digit::DIGIT_COUNT; j++) {
            if (example.label[j] > 0.5f) {
                actual = j;
                break;
            }
        }

        const char ico = actual == predicted ? '+' : '-';
        info("Digit %zu - [%c] Expected: %zu, Predicted: %zu (confidence: %.2f)", i, ico, actual, predicted,
             max_activation);
    }
}

void run(ModelConfig config) {
    Network network;
    init(network, 3, config.layer_sizes, config.learning_rate);
    prime(network);

    for (usize epoch = 0; epoch < config.epochs; epoch++) {
        for (usize i = 0; i < Digit::DIGIT_COUNT; i++) {
            Digit &example = config.training_data[i];
            train(network, example.pixels, example.label);
        }

        if (epoch % 1000 == 0) {
            f32 total_loss = 0;
            for (usize i = 0; i < Digit::DIGIT_COUNT; i++) {
                Digit &example = config.training_data[i];
                forward(network, example.pixels);
                total_loss += calculate_loss(network, example.label);
            }
            info("Epoch %zu: Loss = %f", epoch, total_loss / Digit::DIGIT_COUNT);
        }
    }

    // char *buf = serialize(network);
    // deinit(network);

    // deserialize(network, buf);
    // free(buf);

    test_network(network, config.training_data, Digit::DIGIT_COUNT);
    test_network(network, config.testing_data, 5);
}
}; // namespace digit_net
