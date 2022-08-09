#pragma once

#include <Hitable.cuh>

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

class Sphere : public Hitable {
public:
    __device__ Sphere() : mat(nullptr), center(), radius(0) {}
    __device__ Sphere(Material::Scatterable* mat, Vector3 c, float r) : mat(mat), center(c), radius(r) {}
    ~Sphere() {
        delete mat;
    }

    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const;
private:
    Material::Scatterable* mat;
    Vector3 center;
    float radius;
};

} // namespace Geometry