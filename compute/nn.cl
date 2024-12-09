#define WORK_GROUP_SIZE 16

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
	float s = sigmoid(x);
    return s * (1 - s);
}

void matrix_multiply(
	__global const float* A, // Input matrix
	__global const float* B, // Weight matrix
	__global const float* bias,
	__global float* C, // Output matrix
	const int M, // Input size
	const int N, // Output size
	const bool apply_sigmoid // Wether to apply sigmoid activation
) {
	const int row = get_global_id(0);
	const int col = get_global_id(1);

	if (row < M && col < N) {
		float sum = bias[col];

		// Compute matrix multiplication
		for (int k = 0; k < M; k++) {
			sum += A[row * M + k] * B[k * N + col];
		}

		C[row * N + col] = apply_sigmoid ? sigmoid(sum) : sum;
	}
}

__kernel void apply_activation(	
	__global float* data,
	const int size
) {
	const int id = get_global_id(0);
	if (id < size) {
		data[id] = sigmoid(data[id]);
	}
}

__kernel void compute_gradients(
	__global const float* output_gradients,
	__global const float* weights,
	__global const float* layer_outputs,
	__global const float* layer_inputs,
	__global float* input_gradients,
	__global float* weight_gradients,
	__global float* bias_gradients,
	const int input_size,
	const int output_size
) {
	const int id = get_global_id(0);
	const int local_id = get_global_id(0);

	__local float local_gradients[WORK_GROUP_SIZE];

	if (id < input_size) {
		float gradient_sum = 0.0f;

		// Compute gradients with respect to inputs
		for (int j = 0; j < output_size; j++) {
			float output_grad = output_gradients[j];
			float output_val = layer_outputs[j];
			float deriv = sigmoid_derivative(output_val);
			float weight_val = weights[id * output_size + j];

			gradient_sum += output_grad * deriv * weight_val;

			// Compute weight gradients
			weight_gradients[id * output_size + j] = output_grad * deriv * layer_inputs[id];
		}

		input_gradients[id] = gradient_sum;

		// Compute bias gradients
		if (id < output_size) {
			bias_gradients[id] = output_gradients[id] * sigmoid_derivative(layer_outputs[id]);
		}
	}
}

__kernel void matmul(
	__global const float* input,
	__global const float* weights,
	__global const float* biases,
	__global float* output,
	const int M, // Input size
	const int N  // Output size
	// const int K  // Batch size
) {
	const int row = get_global_id(0);
	const int col = get_global_id(1);

	// if (row < M && col < N) {
	if (row < N) {
		float sum = biases[row];
		for (int i = 0; i < M; i++) {
			sum += input[i] * weights[i * N + row];
		}
		output[row] = sigmoid(sum);
	}
}

__kernel void backward(
	__global const float* output_gradients,
	__global const float* weights,
	__global const float* layer_outputs,
	__global const float* layer_inputs,
	__global float* input_gradients,
	__global float* weight_gradients,
	__global float* bias_gradients,
	const int input_size,
	const int output_size
) {
	const int id = get_global_id(0);

	if (id < input_size) {
		float sum = 0.0f;
		for (int j = 0; j < output_size; j++) {
			float output = layer_outputs[j];
			float grad = output_gradients[j] * sigmoid_derivative(output);
			sum += grad * weights[id * output_size + j];
			weight_gradients[id * output_size + j] = grad * layer_inputs[id];
		}

		input_gradients[id] = sum;
		if (id < output_size) {
			bias_gradients[id] = output_gradients[id] * sigmoid_derivative(layer_outputs[id]);
		}
	}
}
