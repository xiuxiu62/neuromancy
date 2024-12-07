float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
	float s = sigmoid(x);
    return s * (1 - s);
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
