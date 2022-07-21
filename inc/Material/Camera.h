#pragma once

#include "Ray.h"
#include "Vector3.h"

namespace Material {

class Camera {
public:
    Camera(Geometry::Vector3 lookFrom, 
            Geometry::Vector3 lookAt,
            Geometry::Vector3 upDirection,
            float vFOV, 
            float ratio) : verticalFOV(vFOV), aspectRatio(ratio) {
        float angle = vFOV * 2.0 * Geometry::PI / 360.0;
        float height = 2.0 * tan(angle / 2.0);
        float width = height * aspectRatio;

        Geometry::Vector3 z = (lookFrom - lookAt) / !(lookFrom - lookAt);
        Geometry::Vector3 x = (upDirection % z) / !(upDirection % z);
        Geometry::Vector3 y = z % x;

        origin = lookFrom;
        lowerLeftCorner = origin - (width / 2.0) * x - (height / 2.0) * y - z;
        horizontal = width * x;
        vertical = height * y;
    }

    Geometry::Ray getRay(float u, float v) { 
        return Geometry::Ray(origin, lowerLeftCorner + u * horizontal + v * vertical - origin); 
    }

private:
    float verticalFOV;
    float aspectRatio;

    Geometry::Vector3 origin;
    Geometry::Vector3 lowerLeftCorner;
    Geometry::Vector3 horizontal;
    Geometry::Vector3 vertical;
};

} // namespace Material
