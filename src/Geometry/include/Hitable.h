#ifndef HITABLEH 
#define HITABLEH

#include "Vector3.h"
#include "Ray.h"

namespace Geometry {

// HitRecord
//
// t: parameter for where it hits the object
// hitPoint: point where it hits
// surfaceNormal: the surface normal direction, not normalized
struct HitRecord {
    HitRecord() : t(0), hitPoint(), surfaceNormal() {}

    float t;
    Vector3 hitPoint;
    Vector3 surfaceNormal;
};

// Hitable
// 
// Interface for all objects in scene to implement their ray hit logic
// hit: returns true if the ray hits the object, and keeps the information of hits in the record parameter 
class Hitable {
public:
    virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const = 0;
};

} // namespace Geometry

#endif