#pragma once

#include "core/types.h"

#include <cmath>
#include <cstdlib>
#include <limits>

struct Layer {
    usize input_size, output_size;
    f32 *weights;
    f32 *biases;
    f32 *output;

    f32 *deltas;
    f32 *inputs;
};

struct Network {
    Layer *layers;
    usize layer_count;
    f32 learning_rate;
};

f32 sigmoid(f32 x);

f32 sigmoid_derivative(f32 x);

// LAYER
void init(Layer &layer, usize input_size, usize output_size);
void deinit(Layer &layer);
void prime(Layer &layer);
void forward(Layer &layer, f32 *input);
void backward(Layer &layer, f32 *next_layer_deltas, f32 learning_rate);

void serialize(Layer &layer, char **buffer);
void deserialize(Layer &layer, char **buffer);

// NETWORK
void init(Network &network, usize layer_count, usize *layer_sizes, f32 learning_rate);
void deinit(Network &network);
void prime(Network &network);
void forward(Network &network, f32 *input);
void backward(Network &network);
void train(Network &network, f32 *input, f32 *target);
f32 calculate_loss(Network &network, f32 *target);

char *serialize(Network &network);
void deserialize(Network &network, char *buffer);
