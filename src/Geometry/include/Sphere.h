#ifndef SPHEREH
#define SPHEREH

#include "Hitable.h"

namespace Geometry {

class Sphere : public Hitable {
public: 
    Sphere() : center(), radius(0) {}
    Sphere(Vector3 c, float r) : center(c), radius(r) {}

    virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const;
private:
    Vector3 center;
    float radius;
};

} // namespace Geometry
#endif