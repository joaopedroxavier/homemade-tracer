#pragma once

#include <Hitable.cuh>
#include <Vector3.cuh>

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

class Quadrilateral : public Hitable {
public:
    __device__ Quadrilateral() : 
            mat(nullptr), firstPoint(), secondPoint(), thirdPoint(), fourthPoint(), surfaceNormal() {}

    __device__ Quadrilateral(Material::Scatterable* mat, Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4) : 
            mat(mat), firstPoint(p1), secondPoint(p2), thirdPoint(p3), fourthPoint(p4) {
        Vector3 u = p2 - p1;
        Vector3 v = p3 - p1;

        Vector3 n = u % v;
        surfaceNormal = n / !n;
    }

    ~Quadrilateral() {
        delete mat;
    }

    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const;
private:
    Material::Scatterable* mat;

    // Three points that defines the triangle;
    Vector3 firstPoint;
    Vector3 secondPoint;
    Vector3 thirdPoint;
    Vector3 fourthPoint;

    Vector3 surfaceNormal;

    __device__ bool liesInside(const Vector3& p) const;
};

} // namespace Geometry
