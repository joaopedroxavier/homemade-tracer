#pragma once

#include <iostream>
#include <cmath>
#include <cstdlib>

namespace Geometry {

const float EPS = float(1e-9);
const float PI = acos(-1);

class Vector3 {
public:
    Vector3() { e[0] = 0; e[1] = 0; e[2] = 0; }
    Vector3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    inline float x() const { return e[0]; }
    inline float y() const { return e[1]; }
    inline float z() const { return e[2]; }
    inline float r() const { return e[0]; }
    inline float g() const { return e[1]; }
    inline float b() const { return e[2]; }

    inline const Vector3& operator+() const;
    inline Vector3 operator-() const;
    inline float operator[](int i) const;
    inline float& operator[](int i);

    inline Vector3& operator += (const Vector3& v);
    inline Vector3& operator -= (const Vector3& v);
    inline Vector3& operator /= (const Vector3& v);
    inline Vector3& operator *= (const Vector3& v);
    inline Vector3& operator /= (float k);
    inline Vector3& operator *= (float k);
    inline float operator ! ();

private:
    float e[3];
};

inline std::ostream& operator <<(std::ostream& os, Vector3& v) {
    os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    return os;
}

inline Vector3 operator +(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() + v2.x(),
        v1.y() + v2.y(),
        v1.z() + v2.z()
    };
}

inline Vector3 operator -(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() - v2.x(),
        v1.y() - v2.y(),
        v1.z() - v2.z()
    };
}

inline Vector3 operator ^(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() * v2.x(),
        v1.y() * v2.y(),
        v1.z() * v2.z()
    };
}

inline Vector3 operator /(const Vector3& v, float k) {
    return {
        v.x() / k,
        v.y() / k,
        v.z() / k
    };
}

inline Vector3 operator *(const Vector3& v, float k) {
    return {
        v.x() * k,
        v.y() * k,
        v.z() * k
    };
}

inline Vector3 operator *(float k, const Vector3& v) {
    return v * k;
}

inline float operator *(const Vector3& v1, const Vector3& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

inline Vector3 operator %(const Vector3& v1, const Vector3& v2) {
    return {
        v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x()
    };
}

inline float operator !(const Vector3& v) {
    return v * v;
}

inline const Vector3& Vector3::operator +() const {
    return *this;
}

inline Vector3 Vector3::operator -() const {
    return {
        -this->e[0],
        -this->e[1],
        -this->e[2]
    };
}

inline float Vector3::operator [](int i) const {
    return this->e[i];
}

inline float& Vector3::operator [](int i) {
    return this->e[i];
}

inline Vector3& Vector3::operator +=(const Vector3& v) {
    this->e[0] += v.e[0];
    this->e[1] += v.e[1];
    this->e[2] += v.e[2];
    return *this;
}

inline Vector3& Vector3::operator -=(const Vector3& v) {
    this->e[0] -= v.e[0];
    this->e[1] -= v.e[1];
    this->e[2] -= v.e[2];
    return *this;
}

inline Vector3& Vector3::operator /=(const Vector3& v) {
    this->e[0] /= v.e[0];
    this->e[1] /= v.e[1];
    this->e[2] /= v.e[2];
    return *this;
}

inline Vector3& Vector3::operator *=(const Vector3& v) {
    this->e[0] *= v.e[0];
    this->e[1] *= v.e[1];
    this->e[2] *= v.e[2];
    return *this;
}

inline Vector3& Vector3::operator *=(float k) {
    this->e[0] *= k;
    this->e[1] *= k;
    this->e[2] *= k;
    return *this;
}

inline Vector3& Vector3::operator /=(float k) {
    this->e[0] /= k;
    this->e[1] /= k;
    this->e[2] /= k;
    return *this;
}

inline float Vector3::operator !() {
    return sqrt((*this) * (*this));
}

Vector3 reflect(const Vector3& v, const Vector3& n);

} // namespace Geometry
