#include <iostream>
#include <limits>

#include "Common.h"
#include "Ray.h"
#include "Vector3.h"
#include "Hitable.h"
#include "HitableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Scatterable.h"
#include "Diffuse.h"

using namespace Common;
using namespace Geometry;
using namespace Material;

const Vector3 NO_COLOR = Vector3(0.0, 0.0, 0.0);

const float PI = acos(-1);
const float INF = 1e9;

const int MAX_DEPTH = 50;

// t: parameter to create a gradient effect on background.
Vector3 background(float t) {
    return (1.0 - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
}

Vector3 color(const Ray& r, Hitable* world, int depth) {
    if (depth > MAX_DEPTH) {
        return NO_COLOR;
    }

    HitRecord rec;
    if (world->hit(r, 0.001, INF, rec)) {
        Material::ScatterRecord sRec = rec.material->scatter(r, rec);
        if (sRec.didScatter) {
            return sRec.attenuation ^ color(sRec.scatteredRay, world, depth + 1);
        }
        else {
            return NO_COLOR;
        }
    }
    else {
        Vector3 unitDirection = r.direction().unit();
        float t = 0.5 * (unitDirection.y() + 1.0);
        return background(t);
    }
}

int main() {
    int nx = 600;
    int ny = 300;
    int ns = 20;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    Hitable* list[2];
    list[0] = new Sphere(
        new Diffuse(Vector3(0.8, 0.3, 0.3)),
        Vector3(0, -100.5, -1),
        100);
    list[1] = new Sphere(
        new Diffuse(Vector3(0.8, 0.0, 0.8)),
        Vector3(0, 0, -1),
        0.5);
    Hitable* world = new HitableList(list, 2);
    Camera cam;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            Vector3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float ru = getRandomFloat();
                float rv = getRandomFloat();

                float u = float(i + ru) / float(nx);
                float v = float(j + rv) / float(ny);

                Ray r = cam.getRay(u, v);
                col += color(r, world, 0);
            }
            col /= float(ns);
            col = Vector3(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
            int ir = int(255.99 * col.r());
            int ig = int(255.99 * col.g());
            int ib = int(255.99 * col.b());

            std::cout << ir << " " << ig << " " << ib << "\n";
            //std::cerr << returned << ": " << ir << " " << ig << " " << ib << std::endl;
        }
    }
    return 0;
}