#ifndef CURRENT_TIME_H
#define CURRENT_TIME_H

#include <chrono>
#include <cstdint>

class Timer {
    std::chrono::high_resolution_clock m_clock;

public:
    uint64_t milliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>
        (m_clock.now().time_since_epoch()).count();
    }

    uint64_t microseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>
        (m_clock.now().time_since_epoch()).count();
    }

    uint64_t nanoseconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
        (m_clock.now().time_since_epoch()).count();
    }
};

#endif  /* CURRENT_TIME_H */