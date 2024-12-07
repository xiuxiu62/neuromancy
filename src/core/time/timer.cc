#include "core/time/timer.h"

#include <cmath>

Timer timer_create(f64 duration, bool repeat) {
    return {
        .duration = duration,
        .current_time = 0,
        .repeat = repeat,
    };
}

void timer_update(Timer *timer, f64 delta_t) {
    timer->current_time += delta_t;
    if (timer_is_finished(timer) && timer->repeat) timer_reset(timer);
}

bool timer_is_finished(const Timer *timer) {
    return timer->current_time >= timer->duration;
}

void timer_reset(Timer *timer) {
    timer->current_time = 0.0;
}

f64 timer_progress(const Timer *timer) {
    if (timer->duration <= 0.0) return 0.0;
    return std::fmin(timer->current_time / timer->duration, 1.0);
}
