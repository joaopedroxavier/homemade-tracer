#include <Sphere.cuh>

namespace Geometry {

struct QuadraticEquationAnswer {
    __device__ QuadraticEquationAnswer(int roots, float firstRoot, float secondRoot) : numRoots(roots), firstRoot(firstRoot), secondRoot(secondRoot) {}
    int numRoots;
    float firstRoot;
    float secondRoot;
};

__device__ QuadraticEquationAnswer solveQuadraticEquation(float a, float b, float c) {
    float disc = b * b - 4 * a * c;
    float EPS = 1e-9;

    if (disc < -EPS) {
        return QuadraticEquationAnswer(0, float(0), float(0));
    }

    float ans = -b / (2 * a);
    if (disc < EPS) {
        return QuadraticEquationAnswer(1, ans, ans);
    }

    return QuadraticEquationAnswer(
        2,
        float(ans - (sqrt(disc) / (2 * a))),
        float(ans + (sqrt(disc) / (2 * a)))
    );
}

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const {
    Vector3 OC = r.source() - center;
    float EPS = 1e-9f;

    QuadraticEquationAnswer solution = solveQuadraticEquation(
        r.direction() * r.direction(),
        2 * OC * r.direction(),
        OC * OC - radius * radius
    );

    if (solution.numRoots == 0) {
        return false;
    }

    float firstSol = solution.firstRoot;
    if ((firstSol < tMax - EPS) && (firstSol > tMin + EPS)) {
        record.t = firstSol;
        record.hitPoint = r.at(firstSol);
        record.surfaceNormal = (record.hitPoint - center) / radius;
        record.material = mat;
        return true;
    }

    float secondSol = solution.secondRoot;
    if ((secondSol < tMax - EPS) && (secondSol > tMin + EPS)) {
        record.t = secondSol;
        record.hitPoint = r.at(secondSol);
        record.surfaceNormal = (record.hitPoint - center) / radius;
        record.material = mat;
        return true;
    }

    return false;
}

} // namespace Geometry