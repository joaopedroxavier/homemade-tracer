#pragma once

#include <Hitable.cuh>

namespace Geometry {

class HitableList : public Hitable {
public:
    __device__ HitableList(): list(), size(0) {}
    __device__ HitableList(Hitable **l, int n): list(l), size(n) {}

    __device__ virtual void setMaterial(Material::Scatterable* mat) {}
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const;

    Hitable **list;
    int size;
};

} // namespace Geometry
