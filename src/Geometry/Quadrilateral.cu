#include <Quadrilateral.cuh>

namespace Geometry {

__device__ bool Quadrilateral::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
    float t1 = (firstPoint - r.source()) * surfaceNormal;
    float t2 = (r.direction()) * surfaceNormal;

    float t = t1 / t2;

    if (t < tMin || t > tMax) {
        return false;
    }

    Vector3 pointOnPlane = r.source() + t * r.direction();

    Vector3 PA = firstPoint - pointOnPlane;
    Vector3 PB = secondPoint - pointOnPlane;
    Vector3 PC = thirdPoint - pointOnPlane;
    Vector3 PD = fourthPoint - pointOnPlane;

    Vector3 a = PA % PB;
    Vector3 b = PB % PC;
    Vector3 c = PC % PD;
    Vector3 d = PD % PA;

    bool liesInside = (a * b > 1e-9f) &&
        (a * c > 1e-9f) &&
        (a * d > 1e-9f);

    if (liesInside) {
        record.t = t;
        record.hitPoint = pointOnPlane;
        record.surfaceNormal = surfaceNormal;
        record.material = mat;
        return true;
    }

    return false;
}

} // namespace Geometry
