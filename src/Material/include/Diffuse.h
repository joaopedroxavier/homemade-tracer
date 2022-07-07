#pragma once

#include "Ray.h"
#include "Vector3.h"
#include "Hitable.h"
#include "Scatterable.h"

namespace Material {

class Diffuse : public Scatterable {
public:
    Diffuse() : albedo() {}
    Diffuse(Geometry::Vector3 c) : albedo(c) {}

    virtual ScatterRecord scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& record) const;
private:
    Geometry::Vector3 albedo;
};

} // namespace Material
