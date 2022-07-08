#pragma once

#include "Vector3.h"

namespace Geometry {

class Ray {
public:
    Ray() {}
    Ray(const Vector3& a, const Vector3& b): src(a), dir(b/(!b)) {}

    Vector3 source() const { return src; }
    Vector3 direction() const { return dir; }

    Vector3 at(float k) const { return src + k*dir; }
private:
    Vector3 src;
    Vector3 dir;
};

} // namespace Geometry
