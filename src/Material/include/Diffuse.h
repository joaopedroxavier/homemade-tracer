#ifndef DIFFUSEH 
#define DIFFUSEH

#include "Ray.h"
#include "Vector3.h"
#include "Hitable.h"
#include "Reflectable.h"

namespace Material {

class Diffuse : public Reflectable {
public:
    Diffuse() : color() {}
    Diffuse(Geometry::Vector3 c) : color(c) {}

    virtual Geometry::Ray reflect(const Geometry::HitRecord& record) const;

private:
    Geometry::Vector3 color;
};

} // namespace Material

#endif