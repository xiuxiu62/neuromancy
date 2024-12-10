#pragma once

#include "core/types.h"

struct Timer {
    f64 duration;
    f64 current_time;
    bool repeat;
};

Timer timer_create(f64 duration, bool repeat = false);

void timer_update(Timer *timer, f64 delta_t);

bool timer_is_finished(const Timer *timer);

void timer_reset(Timer *timer);

// completion percentage of a timer (0 to 1).
f64 timer_progress(const Timer *timer);
