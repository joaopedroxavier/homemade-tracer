#pragma once

#include "Ray.h"
#include "HitRecord.h"

namespace Material {

struct ScatterRecord {
    bool didScatter;
    Geometry::Vector3 attenuation;
    Geometry::Ray scatteredRay;
};

class Scatterable {
public:
    virtual ScatterRecord scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& record) const = 0;
};

} // namespace Material
