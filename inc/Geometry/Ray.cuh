#pragma once

#include <Vector3.cuh>

namespace Geometry {

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const Vector3& a, const Vector3& b): src(a), dir(b/(!b)) {}

    __device__ Vector3 source() const { return src; }
    __device__ Vector3 direction() const { return dir; }

    __device__ Vector3 at(float k) const { return src + k*dir; }
private:
    Vector3 src;
    Vector3 dir;
};

} // namespace Geometry
