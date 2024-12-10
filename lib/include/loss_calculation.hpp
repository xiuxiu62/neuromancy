#include "core/types.h"

#include <cmath>

const f32 EPSILON = 1e-7f;

enum LossFunction {
    MSE,
    MAE,
    Huber,
    CrossEntropy,
    BinaryCrossEntropy,
    Hinge,
};

f32 mse_loss(f32 *predicted, f32 *target, usize size) {
    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        f32 diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

void mse_loss_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    for (usize i = 0; i < size; i++) {
        derivative[i] = 2.0f * (predicted[i] - target[i]) / size;
    }
}

f32 mae_loss(f32 *predicted, f32 *target, usize size) {
    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        sum += fabsf(predicted[i] - target[i]);
    }
    return sum / size;
}

void mae_loss_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    for (usize i = 0; i < size; i++) {
        derivative[i] = predicted[i] > target[i] ? 1.0f : -1.0f;
    }
}

f32 huber_loss(f32 *predicted, f32 *target, usize size, f32 delta) {
    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        f32 diff = fabsf(predicted[i] - target[i]);
        if (diff <= delta) {
            sum += diff * diff * 0.5f;
        } else {
            sum += delta * (diff - delta * 0.5f);
        }
    }
    return sum / size;
}

void huber_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size, f32 delta) {
    for (usize i = 0; i < size; i++) {
        f32 diff = predicted[i] - target[i];
        if (fabsf(diff) <= delta) {
            derivative[i] = diff;
        } else {
            derivative[i] = (diff > 0.0f ? 1.0f : -1.0f) * delta;
        }
        derivative[i] /= size;
    }
}

f32 cross_entropy_loss(f32 *predicted, f32 *target, usize size) {
    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        f32 p = fmaxf(fminf(predicted[i], 1.0f - EPSILON), EPSILON);
        sum -= target[i] * logf(p);
    }
    return sum / size;
}

void cross_entropy_loss_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    for (usize i = 0; i < size; i++) {
        f32 p = fmaxf(fminf(predicted[i], 1.0f - EPSILON), EPSILON);
        derivative[i] = -target[i] / (p * size);
    }
}

f32 binary_cross_entropy_loss(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    f32 sum = 0.0f;

    for (usize i = 0; i < size; i++) {
        f32 p = fmaxf(fminf(predicted[i], 1.0f - EPSILON), EPSILON);
        sum -= target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p);
    }
    return sum / size;
}

void binary_cross_entropy_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    const f32 epsilon = 1e-7f;

    for (usize i = 0; i < size; i++) {
        f32 p = fmaxf(fminf(predicted[i], 1.0f - epsilon), epsilon);
        derivative[i] = (p - target[i]) / (p * (1.0f - p) * size);
    }
}

f32 hinge_loss(f32 *predicted, f32 *target, usize size) {
    f32 sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        sum += fmaxf(0.0f, 1.0f - target[i] * predicted[i]);
    }
    return sum / size;
}

void hinge_derivative(f32 *predicted, f32 *target, f32 *derivative, usize size) {
    for (usize i = 0; i < size; i++) {
        if (target[i] * predicted[i] < 1.0f) {
            derivative[i] = -target[i] / size;
        } else {
            derivative[i] = 0.0f;
        }
    }
}
