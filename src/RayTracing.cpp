#include <iostream>
#include <limits>
#include <random>

#include "Ray.h"
#include "Vector3.h"
#include "Hitable.h"
#include "HitableList.h"
#include "Sphere.h"
#include "Camera.h"

using namespace Geometry;
using namespace Material;

const float PI = acos(-1);
const float INF = 1e9;
const int MAX_REFLECTIONS = 50;

int returned = 0;

// Random number in [0, 1)
float getRandomFloat() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(e);
}

Vector3 randomInUnitSphere() {
    float r = 1.0;
    float theta = 2 * PI * getRandomFloat();
    float phi = 2 * PI * getRandomFloat();

    return {
        r * cos(theta) * cos(phi),
        r * cos(theta) * sin(phi),
        r * sin(theta)
    };
}

Vector3 color(const Ray& r, Hitable* world, int reflections) {
    if (reflections > MAX_REFLECTIONS) {
        return Vector3(0.0, 0.0, 0.0);
    }
    HitRecord rec;
    if (world->hit(r, 0.001, INF, rec)) {
        Vector3 target = rec.hitPoint + rec.surfaceNormal + randomInUnitSphere();
        return 0.5 * color(Ray(rec.hitPoint, target - rec.hitPoint), world, reflections + 1);
    }
    else {
        Vector3 unitDirection = r.direction().unit();
        float t = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
    }
}

int main() {
    int nx = 600;
    int ny = 300;
    int ns = 20;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    Hitable* list[2];
    list[0] = new Sphere(Vector3(0, -100.5, -1), 100);
    list[1] = new Sphere(Vector3(0, 0, -1), 0.5);
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

                if (i == (nx / 2) && j == 60 && s == 0) {
                    std::cerr << "Hit" << std::endl;
                }

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