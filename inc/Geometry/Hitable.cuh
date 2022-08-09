#pragma once

#include <Vector3.cuh>
#include <HitRecord.cuh>
#include <Ray.cuh>

namespace Geometry {

class Hitable {
public:
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const = 0;
};

} // namespace Geometry
