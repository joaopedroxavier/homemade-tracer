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
#include "Metallic.h"
#include "Glass.h"

using namespace Common;
using namespace Geometry;
using namespace Material;

const Vector3 NO_COLOR = Vector3(0.0, 0.0, 0.0);

const float PI = acos(-1);
const float INF = 1e9;

const int MAX_DEPTH = 50;

// t: parameter to create a gradient effect on background.
Vector3 background(float t) {
    float brightness = 1.5;
    Vector3 col = brightness * ((1.0 - t) * Vector3(0.05137, 0.15294, 1.0) + 
                                t * Vector3(0.99215, 0.60784, 0.10980));
    return Vector3(
        std::min(col.r(), 1.0f),
        std::min(col.g(), 1.0f),
        std::min(col.b(), 1.0f)
    );
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
        Vector3 unitDirection = r.direction() / (!r.direction());
        float t = 0.5 * (unitDirection.y() + 1.0);
        return background(t);
    }
}

int main() {
    int nx = 600;
    int ny = 300;
    int ns = 100;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    Hitable* list[3];
    list[0] = new Sphere(
        new Metallic(Vector3(0.8, 0.8, 0.8)),
        Vector3(0, -100.5, -1),
        100);
    list[1] = new Sphere(
        new Glass(1.2),
        Vector3(0.6, 0.0, -1.5),
        0.5);
    list[2] = new Sphere(
        new Diffuse(Vector3(0.8, 0.0, 0.8)),
        Vector3(-0.6, 0.0, -1.5),
        0.5);
    Hitable* world = new HitableList(list, 3);
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

    std::cerr << "Successfully populated the PPM file." << std::endl;

    return 0;
}