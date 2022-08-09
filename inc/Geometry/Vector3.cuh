#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Geometry {

class Vector3 {
public:
    __host__ __device__ Vector3() { e[0] = 0; e[1] = 0; e[2] = 0; }
    __host__ __device__ Vector3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vector3& operator+() const;
    __host__ __device__ inline Vector3 operator-() const;
    __host__ __device__ inline float operator[](int i) const;
    __host__ __device__ inline float& operator[](int i);

    __host__ __device__ inline Vector3& operator += (const Vector3& v);
    __host__ __device__ inline Vector3& operator -= (const Vector3& v);
    __host__ __device__ inline Vector3& operator /= (const Vector3& v);
    __host__ __device__ inline Vector3& operator *= (const Vector3& v);
    __host__ __device__ inline Vector3& operator /= (float k);
    __host__ __device__ inline Vector3& operator *= (float k);
    __host__ __device__ inline float operator ! ();

private:
    float e[3];
};

inline std::ostream& operator <<(std::ostream& os, Vector3& v) {
    os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    return os;
}

__host__ __device__ inline Vector3 operator +(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() + v2.x(),
        v1.y() + v2.y(),
        v1.z() + v2.z()
    };
}

__host__ __device__ inline Vector3 operator -(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() - v2.x(),
        v1.y() - v2.y(),
        v1.z() - v2.z()
    };
}

__host__ __device__ inline Vector3 operator ^(const Vector3& v1, const Vector3& v2) {
    return {
        v1.x() * v2.x(),
        v1.y() * v2.y(),
        v1.z() * v2.z()
    };
}

__host__ __device__ inline Vector3 operator /(const Vector3& v, float k) {
    return {
        v.x() / k,
        v.y() / k,
        v.z() / k
    };
}

__host__ __device__ inline Vector3 operator *(const Vector3& v, float k) {
    return {
        v.x() * k,
        v.y() * k,
        v.z() * k
    };
}

__host__ __device__ inline Vector3 operator *(float k, const Vector3& v) {
    return v * k;
}

__host__ __device__ inline float operator *(const Vector3& v1, const Vector3& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

__host__ __device__ inline Vector3 operator %(const Vector3& v1, const Vector3& v2) {
    return {
        v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x()
    };
}

__host__ __device__ inline float operator !(const Vector3& v) {
    return v * v;
}

__host__ __device__ inline const Vector3& Vector3::operator +() const {
    return *this;
}

__host__ __device__ inline Vector3 Vector3::operator -() const {
    return {
        -this->e[0],
        -this->e[1],
        -this->e[2]
    };
}

__host__ __device__ inline float Vector3::operator [](int i) const {
    return this->e[i];
}

__host__ __device__ inline float& Vector3::operator [](int i) {
    return this->e[i];
}

__host__ __device__ inline Vector3& Vector3::operator +=(const Vector3& v) {
    this->e[0] += v.e[0];
    this->e[1] += v.e[1];
    this->e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator -=(const Vector3& v) {
    this->e[0] -= v.e[0];
    this->e[1] -= v.e[1];
    this->e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator /=(const Vector3& v) {
    this->e[0] /= v.e[0];
    this->e[1] /= v.e[1];
    this->e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator *=(const Vector3& v) {
    this->e[0] *= v.e[0];
    this->e[1] *= v.e[1];
    this->e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator *=(float k) {
    this->e[0] *= k;
    this->e[1] *= k;
    this->e[2] *= k;
    return *this;
}

__host__ __device__ inline Vector3& Vector3::operator /=(float k) {
    this->e[0] /= k;
    this->e[1] /= k;
    this->e[2] /= k;
    return *this;
}

__host__ __device__ inline float Vector3::operator !() {
    return sqrt((*this) * (*this));
}

__host__ __device__ Vector3 reflect(const Vector3& v, const Vector3& n);

} // namespace Geometry
