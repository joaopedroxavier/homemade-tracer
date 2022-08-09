#include <Glass.cuh>
#include <HitRecord.cuh>

namespace Material {

__device__ bool internalReflect(const Geometry::Vector3& l,
    const Geometry::Vector3& n,
    float r /* refraction index ratio between the media */) {
    Geometry::Vector3 ul = l / (!l);
    Geometry::Vector3 un = n / (!n);
    float dot = ul * un;

    return (1.0 - r * r * (1 - dot * dot) < -1e-9f);
}

// I'm just believing Mr Shirley for this one lol
__device__ float schlick(float cos, float r /* refraction index */) {
    if (r > 1.0) {
        cos = sqrt(1.0 - r * r * (1.0 - cos * cos));
    }
    float r0 = (1 - r) / (1 + r);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cos), 5);
}

__device__ bool refract(Geometry::Vector3 l,
    Geometry::HitRecord rec,
    float r /* refraction index */,
    curandState* rng, 
    Material::ScatterRecord& scatterRecord) {
    float cos = -l * rec.surfaceNormal;
    float reflectProb = schlick(cos, r);

    if (internalReflect(l, rec.surfaceNormal, r)) {
        reflectProb = 1.0;
    }

    if (curand_uniform(rng) < reflectProb) {
        Geometry::Vector3 reflected = Geometry::reflect(l, rec.surfaceNormal);

        scatterRecord.attenuation = Geometry::Vector3(1.0, 1.0, 1.0);
        scatterRecord.scatteredRay = Geometry::Ray(
            rec.hitPoint,
            reflected
        );

        return true;
    }
    float c = -rec.surfaceNormal * l;

    // Apply Snell's law, Vector form to avoid trigonometry calls
    Geometry::Vector3 refracted =
        r * l + ((r * c) - sqrt(1 - r * r * (1 - c * c))) * rec.surfaceNormal;

    scatterRecord.attenuation = Geometry::Vector3(1.0, 1.0, 1.0);
    scatterRecord.scatteredRay = Geometry::Ray(
        rec.hitPoint,
        refracted
    );

    return true;
}

__device__ bool Glass::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& hitRecord, 
        curandState* rng,
        Material::ScatterRecord& scatterRecord) const {
    Geometry::Vector3 l = r.direction() / (!r.direction());
    Geometry::HitRecord auxRecord = hitRecord;
    auxRecord.surfaceNormal = hitRecord.surfaceNormal / (!hitRecord.surfaceNormal);

    if (r.direction() * hitRecord.surfaceNormal > 0) {
        auxRecord.surfaceNormal = -auxRecord.surfaceNormal;
        return refract(
            l,
            auxRecord,
            refractionIndex,
            rng, 
            scatterRecord
        );
    }

    return refract(
        l,
        auxRecord,
        1.0 / refractionIndex,
        rng,
        scatterRecord
    );
}

} // namespace Material