#pragma once

#include "core/types.h"

#include <chrono>

struct Clock {
    std::chrono::high_resolution_clock::time_point start_time, previous_time;
    f64 elapsed, delta_t;
    f32 time_scale;
    usize days;
};

extern Clock global_clock;

Clock clock_create();
void clock_update(Clock &clock);
void global_clock_init();
