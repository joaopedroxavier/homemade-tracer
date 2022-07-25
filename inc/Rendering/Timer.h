#pragma once

#include <Common.h>

namespace Rendering {

class Timer {
public:
    Timer();

    uint64_t StartTime() const;
    uint64_t CurrentTime() const;
    uint64_t LastTime() const;
    float Frequency() const;

    template<typename TUpdate>
    void Tick(const TUpdate& update) {
        GetTime(_currentTime);
        _lastTime = _currentTime;

        update();
    }

    void ResetTime();
    void GetTime(uint64_t& time) const;

private:
    uint64_t    _startTime;
    uint64_t    _currentTime;
    uint64_t    _lastTime;

    float       _frequency;
};

} // namespace Rendering
