#ifndef HITABLELISTH
#define HITABLELISTH

#include "Hitable.h"

namespace Geometry {

class HitableList : public Hitable {
public:
    HitableList(): list(), size(0) {}
    HitableList(Hitable **l, int n): list(l), size(n) {}

    virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const;
private:
    Hitable **list;
    int size;
};

} // namespace Geometry

#endif