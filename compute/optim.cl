// SGD
__kernel void sgd_update(
	__global float* weights,	
	__global const float* inputs,	
	__global const float* deltas,	
	const float learning_rate,
	const uint input_size,
	const uint output_size
) {
	size_t i = get_global_id(0);
	if (i >= input_size * output_size) return;

	uint output_id = i % output_size;
	uint input_id = i / output_size;
	float gradient = deltas[output_id] * inputs[input_id];
	weights[i] -= gradient * learning_rate;
}

__kernel void sgd_update_biases(
	__global float* biases,
	__global const float* deltas,
	const float learning_rate
) {
	size_t i = get_global_id(0);
	biases[i] -= deltas[i] * learning_rate;
}

// Momentum
__kernel void momentum_update(
	__global float* weights,	
	__global float* velocities,
	__global const float* inputs,	
	__global const float* deltas,	
	const float learning_rate,
	const float momentum,
	const uint input_size,
	const uint output_size
) {
	size_t i = get_global_id(0);
	if (i >= input_size * output_size) return;

	uint output_id = i % output_size;
	uint input_id = i / output_size;
	float gradient = deltas[output_id] * inputs[input_id];

	velocities[i] = velocities[i] * momentum - gradient * learning_rate; 
	weights[i] += velocities[i];
}

__kernel void momentum_update_biases(
	__global float* biases,
	__global float* velocities,
	__global const float* deltas,
	const float learning_rate,
	const float momentum
) {
	size_t i = get_global_id(0);
	velocities[i] = velocities[i] * momentum - deltas[i] * learning_rate;
	biases[i] += velocities[i];
} 

// RMSprop
__kernel void rmsprop_update(
	__global float* weights,	
	__global float* cache,
	__global const float* inputs,	
	__global const float* deltas,	
	const float learning_rate,
	const float decay_rate,
	const float epsilon,
	const uint input_size,
	const uint output_size
) {
	size_t i = get_global_id(0);
	if (i >= input_size * output_size) return;

	uint output_id = i % output_size;
	uint input_id = i / output_size;
	float gradient = deltas[output_id] * inputs[input_id];

	cache[i] = cache[i] * decay_rate + (1.0f - decay_rate) * gradient * gradient;
	weights[i] -= gradient * learning_rate / (sqrt(cache[i]) + epsilon);
}

__kernel void rmsprop_update_biases(
	__global float* biases,
	__global float* cache,
	__global const float* deltas,
	const float learning_rate,
	const float decay_rate,
	const float epsilon
) {
	size_t i = get_global_id(0);
	float gradient = deltas[i];

	cache[i] = cache[i] * decay_rate + (1.0f - decay_rate) * gradient * gradient;
	biases[i] -= gradient * learning_rate / (sqrt(cache[i]) + epsilon);
}



// Adam
__kernel void adam_update(
	__global float* weights,	
	__global float* m,
	__global float* v,
	__global const float* inputs,	
	__global const float* deltas,	
	const float learning_rate,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float t_scaled,
	const uint input_size,
	const uint output_size
) {
	size_t i = get_global_id(0);
	if (i >= input_size * output_size) return;

	uint output_id = i % output_size;
	uint input_id = i / output_size;
	float gradient = deltas[output_id] * inputs[input_id];

	m[i] = m[i] * beta1 + (1.0f - beta1) * gradient;
	v[i] = v[i] * beta2 + (1.0f - beta2) * gradient * gradient;

	float m_hat = m[i] / (1.0f - pow(beta1, t_scaled));
	float v_hat = v[i] / (1.0f - pow(beta2, t_scaled));

	weights[i] = m_hat * learning_rate / (sqrt(v_hat) + epsilon);
}

__kernel void adam_update_biases(
	__global float* biases,
	__global float* m,
	__global float* v,
	__global const float* deltas,
	const float learning_rate,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float t_scaled
) {
	size_t i = get_global_id(0);
	float gradient = deltas[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * gradient;
    v[i] = beta2 * v[i] + (1.0f - beta2) * gradient * gradient;
    
    float m_hat = m[i] / (1.0f - pow(beta1, t_scaled));
    float v_hat = v[i] / (1.0f - pow(beta2, t_scaled));
    
    biases[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}
