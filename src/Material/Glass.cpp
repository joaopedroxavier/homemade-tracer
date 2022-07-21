#include "Glass.h"
#include "HitRecord.h"
#include "Common.h"

namespace Material {

bool internalReflect(const Geometry::Vector3& l, 
        const Geometry::Vector3& n,
        float r /* refraction index ratio between the media */ ) {
    Geometry::Vector3 ul = l / (!l);
    Geometry::Vector3 un = n / (!n);
    float dot = ul * un;
    
    return (1.0 - r * r * (1 - dot * dot) < -Geometry::EPS);
}

// I'm just believing Mr Shirley for this one lol
float schlick(float cos, float r /* refraction index */) {
    if (r > 1.0) {
        cos = sqrt(1.0 - r * r * (1.0 - cos * cos));
    }
    float r0 = (1 - r) / (1 + r);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cos), 5);
}

ScatterRecord refract(Geometry::Vector3 l,
        Geometry::HitRecord rec,
        float r /* refraction index */) {
    float cos = -l * rec.surfaceNormal;
    float reflectProb = schlick(cos, r);

    if (internalReflect(l, rec.surfaceNormal, r)) {
        reflectProb = 1.0;
    }

    if (Common::getRandomFloat() < reflectProb) {
        Geometry::Vector3 reflected = Geometry::reflect(l, rec.surfaceNormal);
        return {
            true,
            Geometry::Vector3(1.0, 1.0, 1.0),
            Geometry::Ray(
                rec.hitPoint,
                reflected
            )
        };
    } 
    float c = - rec.surfaceNormal * l;

    // Apply Snell's law, Vector form to avoid trigonometry calls
    Geometry::Vector3 refracted = 
            r * l + ((r * c) - sqrt(1 - r * r * (1 - c * c))) * rec.surfaceNormal;

    return {
        true,
        Geometry::Vector3(1.0, 1.0, 1.0),
        Geometry::Ray(
            rec.hitPoint,
            refracted
        )
    };
}

Material::ScatterRecord Glass::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& record) const {
    Geometry::Vector3 l = r.direction() / (!r.direction());
    Geometry::HitRecord auxRecord = record;
    auxRecord.surfaceNormal = record.surfaceNormal / (!record.surfaceNormal);

    if (r.direction() * record.surfaceNormal > 0) {
        auxRecord.surfaceNormal = -auxRecord.surfaceNormal;
        return refract(
            l,
            auxRecord,
            refractionIndex
        );
    }

    return refract(
        l,
        auxRecord,
        1.0 / refractionIndex
    );
}

} // namespace Material