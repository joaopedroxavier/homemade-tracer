#pragma once

#include "Scatterable.h"

namespace Material {

class Metallic : public Scatterable {
public:
    Metallic() : albedo() {}
    Metallic(const Geometry::Vector3 a) : albedo(a) {}

    virtual ScatterRecord scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& record) const;
private:
    Geometry::Vector3 albedo;
};

} // namespace Material