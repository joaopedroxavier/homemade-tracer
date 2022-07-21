#include "HitableList.h"

namespace Geometry {

bool HitableList::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
    HitRecord tempRecord;
    bool hit = false;
    float closest = tMax;
    for (int i = 0; i < size; i++) {
        if (list[i]->hit(r, tMin, closest, tempRecord)) {
            hit = true;
            closest = tempRecord.t;
            record = tempRecord;
        }
    }
    return hit;
}

} // namespace Geometry