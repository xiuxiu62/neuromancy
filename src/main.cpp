#include "backend/opencl.hpp"
#include "digit_net.hpp"
#include "gate_net.hpp"
#include "nn.hpp"

int main() {
    // usize layer_sizes[] = {2, 12, 1};
    // gate_net::TrainingData training_data[4];
    // memcpy(training_data, gate_net::xor_gate, sizeof(gate_net::TrainingData) * 4);

    // usize layer_sizes[] = {digit_net::Digit::PIXEL_COUNT, 64, 32, digit_net::Digit::DIGIT_COUNT};
    usize layer_sizes[] = {digit_net::Digit::PIXEL_COUNT, 48, 64, 32, digit_net::Digit::DIGIT_COUNT};

    digit_net::ModelConfig config = {
        .learning_rate = 0.01f,
        .layer_sizes = layer_sizes,
        .training_data = digit_net::data_set,
        .testing_data = digit_net::test_data,
        // .epochs = 250000,
        .epochs = 10000,
    };

    run(config);

    return 0;
}

// int main() {
//     OpenCLContext ctx;
//     init(ctx);

//     usize layer_sizes[] = {2, 12, 1};
//     Network network;
//     init(network, 3, layer_sizes, 0.1);

//     f32 test_input[2] = {0.0f, 1.0f};
//     forward(ctx, network.layers[0], test_input);
//     forward(ctx, network.layers[1], test_input);

//     f32 output_gradient[1] = {1.0f};
//     backward(ctx, network.layers[1], output_gradient, network.learning_rate);
//     backward(ctx, network.layers[0], network.layers[1].deltas, network.learning_rate);

//     // for (usize i = 0; i < network.layers[0].input_size; i++) {
//     //     info("Output[%zu]: %f", i, network.layers[0].inputs[i]);
//     // }

//     // for (usize i = 0; i < network.layers[0].output_size; i++) {
//     //     info("Output[%zu]: %f", i, network.layers[0].output[i]);
//     // }

//     for (usize i = 0; i < network.layers[1].output_size; i++) {
//         info("Output[%zu]: %f", i, network.layers[1].output[i]);
//     }

//     for (usize i = 0; i < network.layers[1].output_size; i++) {
//         info("Output[%zu].delta: %f", i, network.layers[1].deltas[i]);
//     }

//     return 0;
// }

// int main() { // Initialize OpenCL
//     OpenCLContext ctx;
//     init(ctx);
//     usize layer_sizes[] = {2, 12, 1};
//     Network network;
//     init(network, 3, layer_sizes, 0.1);

//     // Train XOR gate
//     f32 inputs[][2] = {
//         {0.0f, 0.0f},
//         {0.0f, 1.0f},
//         {1.0f, 0.0f},
//         {1.0f, 1.0f},
//     };
//     f32 targets[] = {0.0f, 1.0f, 1.0f, 0.0f}; // XOR truth table

//     // Training loop
//     for (usize epoch = 0; epoch < 5000; epoch++) {
//         f32 total_loss = 0;
//         for (usize i = 0; i < 4; i++) {
//             train(ctx, network, inputs[i], &targets[i]);
//             total_loss += calculate_loss(network, &targets[i]);
//         }
//         if (epoch % 100 == 0) {
//             info("Epoch %zu: Loss = %f", epoch, total_loss / 4);
//         }
//     }

//     // Test the network
//     for (usize i = 0; i < 4; i++) {
//         forward(ctx, network.layers[0], inputs[i]);
//         forward(ctx, network.layers[1], network.layers[0].output);
//         info("Input: [%.1f, %.1f] -> Output: %.3f (Expected: %.1f)", inputs[i][0], inputs[i][1],
//              network.layers[1].output[0], targets[i]);
//     }

//     return 0;
// }
