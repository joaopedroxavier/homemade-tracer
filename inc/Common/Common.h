#pragma once

#include <wrl.h>

#include <random>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <tuple>

#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif

// STL
#include <algorithm>
#include <cassert>
#include <chrono>

namespace Common {

float GetRandomFloat();

} // namespace Common
