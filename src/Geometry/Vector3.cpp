#include <Vector3.h>

namespace Geometry {

Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - 2 * (v * n) * n;
}

} // namespace Geometry