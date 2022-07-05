#include "Sphere.h"
#include <tuple>

namespace Geometry {

// solveQuadraticEquation
//
// Solves ax^2 + bx + c = 0.
// Parameters: a, b and c, the equation coefficients
// Return values: 
// - First integer contains the number of non-complex solutions
// - floathe two other values are the solutions. If there is only one non-complex solution, both values are equal
std::tuple<int, float, float> solveQuadraticEquation(float a, float b, float c) {
    float disc = b * b - 4 * a * c;

    if (disc < -EPS) {
        return std::make_tuple(0, float(0), float(0));
    }

    float ans = -b / (2 * a);
    if (disc < EPS) {
        return std::make_tuple(
            1,
            float(ans),
            float(ans)
        );
    }

    return std::make_tuple(
        2,
        float(ans - (sqrt(disc) / (2 * a))),
        float(ans + (sqrt(disc) / (2 * a)))
    );
}

bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
    Vector3 OC = r.source() - center;

    auto parameters = solveQuadraticEquation(
        r.direction() * r.direction(),
        2 * OC * r.direction(),
        OC * OC - radius * radius
    );

    if (std::get<0>(parameters) == 0) {
        return false;
    }

    float firstSol = std::get<1>(parameters);
    if ((firstSol < tMax - EPS) && (firstSol > tMin + EPS)) {
        record.t = firstSol;
        record.hitPoint = r.at(firstSol);
        record.surfaceNormal = (record.hitPoint - center) / radius;
        return true;
    }

    float secondSol = std::get<2>(parameters);
    if ((secondSol < tMax - EPS) && (secondSol > tMin + EPS)) {
        record.t = secondSol;
        record.hitPoint = r.at(secondSol);
        record.surfaceNormal = (record.hitPoint - center) / radius;
        return true;
    }

    return false;
}

} // namespace Geometry