#pragma once

#include "core/file.h"
#include "nn.hpp"

#include <CL/cl.h>
#include <corecrt_wctype.h>
#include <cstdlib>

#define bind_buf(IDENT, TYPE, SIZE, PERMS)                                                                             \
    cl_mem IDENT = clCreateBuffer(ctx.context, PERMS, sizeof(TYPE) * SIZE, nullptr, nullptr);
#define bind_float_buf(IDENT, SIZE, PERMS) bind_buf(IDENT, f32, SIZE, PERMS);

#define enqueue_write_buf(IDENT, HOST_BUFFER, SIZE, TYPE)                                                              \
    clEnqueueWriteBuffer(ctx.queue, IDENT, CL_TRUE, 0, sizeof(TYPE) * SIZE, HOST_BUFFER, 0, nullptr, nullptr);
#define enqueue_float_write_buf(IDENT, HOST_BUFFER, SIZE) enqueue_write_buf(IDENT, HOST_BUFFER, SIZE, f32);

#define set_arg(NAME, VALUE, POSITION, TYPE) clSetKernelArg(ctx.kernels.NAME, POSITION, sizeof(TYPE), &VALUE);

struct OpenCLContext {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    struct {
        cl_kernel matmul;
        cl_kernel forward;
        cl_kernel backward;
    } kernels;
};

char *load_kernel_source(const char *path);

void init(OpenCLContext &ctx) {
    clGetPlatformIDs(1, &ctx.platform, nullptr);
    clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, nullptr);

    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, nullptr);
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, nullptr);

    char *src;
    isize bytes_read = read_file("compute/nn.cl", &src);

    ctx.program = clCreateProgramWithSource(ctx.context, 1, const_cast<const char **>(&src), nullptr, nullptr);
    clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr);
    free(const_cast<void *>(reinterpret_cast<const void *>(src)));

#define bind_kernel(NAME) ctx.kernels.NAME = clCreateKernel(ctx.program, #NAME, nullptr);
    bind_kernel(matmul);
    bind_kernel(backward);
}

void forward(OpenCLContext &ctx, Layer &layer, f32 *input) {

    memcpy(layer.inputs, input, sizeof(f32) * layer.input_size);

    bind_float_buf(d_input, layer.input_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_weights, layer.input_size * layer.output_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_biases, layer.output_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_output, layer.output_size, CL_MEM_WRITE_ONLY);

    enqueue_float_write_buf(d_input, input, layer.input_size);
    enqueue_float_write_buf(d_weights, layer.weights, layer.input_size * layer.output_size);
    enqueue_float_write_buf(d_biases, layer.biases, layer.output_size);

    set_arg(matmul, d_input, 0, cl_mem);
    set_arg(matmul, d_weights, 1, cl_mem);
    set_arg(matmul, d_biases, 2, cl_mem);
    set_arg(matmul, d_output, 3, cl_mem);
    set_arg(matmul, layer.input_size, 4, u32);  // M
    set_arg(matmul, layer.output_size, 5, u32); // N
    // set_arg(matmul, layer.input_size, 6, u32);  // K

    usize global_work_size = layer.output_size;
    // usize global_work_size[2] = {layer.input_size, layer.output_size};
    clEnqueueNDRangeKernel(ctx.queue, ctx.kernels.matmul, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(ctx.queue, d_output, CL_TRUE, 0, sizeof(f32) * layer.output_size, layer.output, 0, nullptr,
                        nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_biases);
    clReleaseMemObject(d_output);
}

void backward(OpenCLContext &ctx, Layer &layer, f32 *next_layer_deltas, f32 learning_rate) {
    bind_float_buf(d_output_gradients, layer.output_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_weights, layer.input_size * layer.output_size, CL_MEM_READ_WRITE);
    bind_float_buf(d_layer_outputs, layer.output_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_layer_inputs, layer.input_size, CL_MEM_READ_ONLY);
    bind_float_buf(d_input_gradients, layer.input_size, CL_MEM_WRITE_ONLY);
    bind_float_buf(d_weight_gradients, layer.input_size * layer.output_size, CL_MEM_WRITE_ONLY);
    bind_float_buf(d_bias_gradients, layer.output_size, CL_MEM_WRITE_ONLY);

    enqueue_float_write_buf(d_output_gradients, next_layer_deltas, layer.output_size);
    enqueue_float_write_buf(d_weights, layer.weights, layer.input_size * layer.output_size);
    enqueue_float_write_buf(d_layer_outputs, layer.output, layer.output_size);
    enqueue_float_write_buf(d_layer_inputs, layer.inputs, layer.input_size);
    set_arg(backward, d_output_gradients, 0, cl_mem);
    set_arg(backward, d_weights, 1, cl_mem);
    set_arg(backward, d_layer_outputs, 2, cl_mem);
    set_arg(backward, d_layer_inputs, 3, cl_mem);
    set_arg(backward, d_input_gradients, 4, cl_mem);
    set_arg(backward, d_weight_gradients, 5, cl_mem);
    set_arg(backward, d_bias_gradients, 6, cl_mem);
    set_arg(backward, layer.input_size, 7, u32);
    set_arg(backward, layer.output_size, 8, u32);

    usize global_work_size = layer.input_size;
    clEnqueueNDRangeKernel(ctx.queue, ctx.kernels.backward, 1, nullptr, &global_work_size, nullptr, 0, nullptr,
                           nullptr);

    f32 *storage = static_cast<f32 *>(malloc(sizeof(f32) * (layer.input_size * layer.output_size + layer.output_size)));
    f32 *weight_updates = storage;
    f32 *bias_updates = storage + (layer.input_size * layer.output_size);

    clEnqueueReadBuffer(ctx.queue, d_input_gradients, CL_TRUE, 0, sizeof(f32) * layer.input_size, layer.deltas, 0,
                        nullptr, nullptr);
    clEnqueueReadBuffer(ctx.queue, d_weight_gradients, CL_TRUE, 0, sizeof(f32) * layer.input_size * layer.output_size,
                        weight_updates, 0, nullptr, nullptr);
    clEnqueueReadBuffer(ctx.queue, d_bias_gradients, CL_TRUE, 0, sizeof(f32) * layer.output_size, bias_updates, 0,
                        nullptr, nullptr);

    for (usize i = 0; i < layer.input_size * layer.output_size; i++) {
        layer.weights[i] -= learning_rate * weight_updates[i];
    }

    for (usize i = 0; i < layer.output_size; i++) {
        layer.biases[i] -= learning_rate * bias_updates[i];
    }

    clReleaseMemObject(d_output_gradients);
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_layer_outputs);
    clReleaseMemObject(d_layer_inputs);
    clReleaseMemObject(d_input_gradients);
    clReleaseMemObject(d_weight_gradients);
    clReleaseMemObject(d_bias_gradients);

    free(storage);
}

void train(OpenCLContext &ctx, Network &network, f32 *input, f32 *target) {
    // Forward pass
    for (usize i = 0; i < network.layer_count; i++) {
        f32 *layer_input = (i == 0) ? input : network.layers[i - 1].output;
        forward(ctx, network.layers[i], layer_input);
    }

    Layer &output_layer = network.layers[network.layer_count - 1];
    for (usize i = 0; i < output_layer.output_size; i++) {
        f32 output = output_layer.output[i];
        output_layer.deltas[i] = sigmoid_derivative(output) * (target[i] - output);
    }

    // info("Target: %f", target[0]);
    // info("Ouput: %f", output_layer.output[0]);
    // info("Delta: %f", output_layer.deltas[0]);

    for (isize i = network.layer_count - 1; i >= 0; i--) {
        f32 *deltas = nullptr;
        if (i < network.layer_count - 1) {
            deltas = network.layers[i + 1].deltas;
        }
        backward(ctx, network.layers[i], deltas, network.learning_rate);
    }
}

// REVISION

struct OpenCLLayer {
    cl_mem weights;
    cl_mem biases;
    cl_mem output;
    cl_mem deltas;
    usize input_size;
    usize output_size;
};

struct OpenCLNetwork {
    OpenCLLayer *layers;
    usize layer_count;
    f32 learning_rate;
};

struct OpenCLTrainingBuffers {
    cl_mem input_data;  // [batch_size * input_size]
    cl_mem target_data; // [batch_size * output_size]
    usize batch_size;
    usize input_size;
    usize output_size;
};

// TODO: Initialize OpenCL network resource directly from config, rather than pre-initializing a network host-side

void init(OpenCLContext &ctx, OpenCLLayer &layer, const Layer &host_layer) {
    layer.input_size = host_layer.input_size;
    layer.output_size = host_layer.output_size;

    layer.weights = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(f32) * layer.input_size * layer.output_size,
                                   nullptr, nullptr);
    layer.biases = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(f32) * layer.output_size, nullptr, nullptr);
    layer.output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(f32) * layer.output_size, nullptr, nullptr);
    layer.deltas = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(f32) * layer.output_size, nullptr, nullptr);

    clEnqueueWriteBuffer(ctx.queue, layer.weights, CL_TRUE, 0, sizeof(f32) * layer.input_size * layer.output_size,
                         layer.weights, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctx.queue, layer.biases, CL_TRUE, 0, sizeof(f32) * layer.output_size, layer.biases, 0, nullptr,
                         nullptr);
}

void init(OpenCLContext &ctx, OpenCLNetwork &network, const Network &host_network) {
    network.layer_count = host_network.layer_count;
    network.learning_rate = host_network.learning_rate;
    network.layers = static_cast<OpenCLLayer *>(malloc(sizeof(OpenCLLayer) * network.layer_count));
}

// TODO: Load chunks of training data in, training incrementally if a region of the training data exceeds the vram cap
// void train(OpenCLContext &ctx, OpenCLNetwork &network, const f32 *training_data, const f32 *target_data,
//            usize frame_count, usize batch_size) {
// }
