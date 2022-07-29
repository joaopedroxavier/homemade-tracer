#include <Common.h>
#include <Diffuse.h>

const float PI = acos(-1);

namespace Material {

Geometry::Vector3 randomInUnitSphere() {
    float r = Common::GetRandomFloat();
    float theta = 2 * PI * Common::GetRandomFloat();
    float phi = 2 * PI * Common::GetRandomFloat();

    return {
        r * cos(theta) * cos(phi),
        r * cos(theta) * sin(phi),
        r * sin(theta)
    };
}

ScatterRecord Diffuse::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& record) const {
    Geometry::Vector3 target = record.hitPoint + record.surfaceNormal + randomInUnitSphere();
    Geometry::Ray scattered = Geometry::Ray(record.hitPoint, target - record.hitPoint);

    return {
        true,
        albedo,
        Geometry::Ray(record.hitPoint, target - record.hitPoint)
    };
}

} // namespace Material