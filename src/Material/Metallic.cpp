#include "Metallic.h"

namespace Material {

ScatterRecord Metallic::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& record) const {
    Geometry::Ray scatteredRay = Geometry::Ray(
        record.hitPoint, 
        Geometry::reflect(r.direction(), record.surfaceNormal)
    );

    return {
        scatteredRay.direction() * record.surfaceNormal > 0,
        albedo,
        scatteredRay
    };
}
    
} // namespace Material