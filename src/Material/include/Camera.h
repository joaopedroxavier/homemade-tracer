#ifndef CAMERAH
#define CAMERAH

#include "Ray.h"
#include "Vector3.h"

namespace Material {

class Camera {
public:
    Camera() {
        origin = Geometry::Vector3(0.0, 0.0, 0.0);
        lowerLeftCorner = Geometry::Vector3(-2.0, -1.0, -1.0);
        horizontal = Geometry::Vector3(4.0, 0.0, 0.0);
        vertical = Geometry::Vector3(0.0, 2.0, 0.0);
    }

    Geometry::Ray getRay(float u, float v) { return Geometry::Ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin); }

private:
    Geometry::Vector3 origin;
    Geometry::Vector3 lowerLeftCorner;
    Geometry::Vector3 horizontal;
    Geometry::Vector3 vertical;
};

} // namespace Material

#endif