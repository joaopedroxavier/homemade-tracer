#pragma once

#include <Hitable.cuh>
#include <Vector3.cuh>

namespace Material {
class Scatterable;
} // namespace Material

namespace Geometry {

class Triangle : public Hitable {
public:
    __host__ __device__ Triangle() : 
            mat(nullptr), firstPoint(), secondPoint(), thirdPoint(), surfaceNormal() {}

    __host__ __device__ Triangle(Vector3 p1, Vector3 p2, Vector3 p3, Material::Scatterable* mat = nullptr) : 
            mat(mat), firstPoint(p1), secondPoint(p2), thirdPoint(p3) {
        Vector3 u = p2 - p1;
        Vector3 v = p3 - p1;

        Vector3 n = u % v;
        surfaceNormal = n / !n;
    }

    ~Triangle() {
        delete mat;
    }

    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const;
private:
    Material::Scatterable* mat;

    // Three points that defines the triangle;
    Vector3 firstPoint;
    Vector3 secondPoint;
    Vector3 thirdPoint;

    Vector3 surfaceNormal;
};

} // namespace Geometry
