#include <cstdlib>
#include <ctime>
#include <cmath>

#include "Diffuse.h"

namespace Material {

Geometry::Vector3 randomInUnitSphere() {
    srand (static_cast <unsigned> (time(0)));

    float r = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
    float theta = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
    float phi = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);

    return {
        r * cos(theta) * cos(phi),
        r * cos(theta) * sin(phi),
        r * sin(theta)
    };
}

Geometry::Ray Diffuse::reflect(const Geometry::HitRecord& record) const {
    Geometry::Vector3 reflection = record.hitPoint + (record.surfaceNormal/ (!record.surfaceNormal)) + randomInUnitSphere();
    return { record.hitPoint, reflection - record.hitPoint };
}

} // namespace Material