#ifndef REFLECTABLEH 
#define REFLECTABLEH

namespace Material {

class Reflectable {
public:
    virtual Geometry::Ray reflect(const Geometry::HitRecord& record) const = 0;
};

} // namespace Material

#endif