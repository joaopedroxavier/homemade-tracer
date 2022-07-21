#pragma once

#include "Scatterable.h"

namespace Material {

class Glass : public Scatterable {
public:
    Glass() : refractionIndex() {}
    Glass(float r) : refractionIndex(r) {}

    virtual ScatterRecord scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& record) const;
private:
    float refractionIndex;
};

} // namespace Material