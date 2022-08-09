#include <Vector3.cuh>

namespace Geometry {

__host__ __device__ Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - 2 * (v * n) * n;
}

} // namespace Geometry