#include <Quadrilateral.cuh>

namespace Geometry {

// Assumes that p already lies on the plane.
// Checks if it's inside the triangle borders.
__device__ bool Quadrilateral::liesInside(const Vector3& p) const {
    Vector3 PA = firstPoint - p;
    Vector3 PB = secondPoint - p;
    Vector3 PC = thirdPoint - p;
    Vector3 PD = fourthPoint - p;

    Vector3 a = PA % PB;
    Vector3 b = PB % PC;
    Vector3 c = PC % PD;
    Vector3 d = PD % PA;

    return (a * b > 1e-9f) &&
        (a * c > 1e-9f) &&
        (a * d > 1e-9f);
}

__device__ bool Quadrilateral::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
    float t1 = (firstPoint - r.source()) * surfaceNormal;
    float t2 = (r.direction()) * surfaceNormal;

    float t = t1 / t2;

    if (t < tMin || t > tMax) {
        return false;
    }

    Vector3 pointOnPlane = r.source() + t * r.direction();

    if (liesInside(pointOnPlane)) {
        record.t = t;
        record.hitPoint = pointOnPlane;
        record.surfaceNormal = surfaceNormal;
        record.material = mat;
        return true;
    }

    return false;
}

} // namespace Geometry
