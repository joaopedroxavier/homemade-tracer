#pragma once

#include <Vector3.h>

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

// HitRecord
//
// t: parameter for where it hits the object
// hitPoint: point where it hits
// surfaceNormal: the surface normal direction, not normalized
struct HitRecord {
    HitRecord() : t(0), hitPoint(), surfaceNormal(), material(nullptr) {}

    float t;
    Vector3 hitPoint;
    Vector3 surfaceNormal;
    Material::Scatterable* material;
};

} // namespace Geometry
