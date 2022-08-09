#pragma once

#include <Vector3.cuh>

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

struct HitRecord {
    float t;
    Vector3 hitPoint;
    Vector3 surfaceNormal;
    Material::Scatterable* material;
};

} // namespace Geometry
