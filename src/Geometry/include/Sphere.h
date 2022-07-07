#pragma once

#include "Hitable.h"

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

class Sphere : public Hitable {
public:
    Sphere() : mat(nullptr), center(), radius(0) {}
    Sphere(Material::Scatterable* mat, Vector3 c, float r) : mat(mat), center(c), radius(r) {}

    virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const;
private:
    Material::Scatterable* mat;
    Vector3 center;
    float radius;
};

} // namespace Geometry