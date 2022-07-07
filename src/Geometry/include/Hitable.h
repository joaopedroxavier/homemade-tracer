#pragma once

#include "Vector3.h"
#include "HitRecord.h"
#include "Ray.h"

namespace Geometry {

// Hitable
// 
// Interface for all objects in scene to implement their ray hit logic
// hit: returns true if the ray hits the object, and keeps the information of hits in the record parameter 
class Hitable {
public:
    virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const = 0;
};

} // namespace Geometry
