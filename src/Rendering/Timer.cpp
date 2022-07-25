#include <Timer.h>
#include <wrl.h>

namespace Rendering {

Timer::Timer() : _startTime(), _currentTime(), _lastTime(), _frequency() {
    _frequency = Frequency();
    ResetTime();
}

uint64_t Timer::StartTime() const {
    return _startTime;
}

uint64_t Timer::CurrentTime() const {
    return _currentTime;
}

uint64_t Timer::LastTime() const {
    return _lastTime;
}

float Timer::Frequency() const {
    LARGE_INTEGER f;

    if (!QueryPerformanceFrequency(&f)) {
        throw new std::exception("Call on QueryPerformanceFrequency failed.");
    }

    return float(f.QuadPart);
}

void Timer::ResetTime() {
    GetTime(_startTime);
    _currentTime = _startTime;
    _lastTime = _startTime;
}

void Timer::GetTime(uint64_t& time) const {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);

    time = t.QuadPart;
}


} // namespace Rendering
