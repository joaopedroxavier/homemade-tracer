#include "Common.h"

namespace Common {

float getRandomFloat() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(e);
}

void ShowMessageOnConsole(std::string s) {
    OutputDebugString(s.c_str());
}

} // namespace Common