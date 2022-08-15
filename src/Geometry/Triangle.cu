#include <Triangle.cuh>

namespace Geometry {

// Assumes that p already lies on the plane.
// Checks if it's inside the triangle borders.
__device__ bool Triangle::liesInside(const Vector3& p) const {
    Vector3 PA = firstPoint - p;
    Vector3 PB = secondPoint - p;
    Vector3 PC = thirdPoint - p;

    Vector3 u = PB % PC;
    Vector3 v = PC % PA;
    Vector3 w = PA % PB;

    return (u * v > 1e-9f) && (u * w > 1e-9f);
}

__device__ bool Triangle::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
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
