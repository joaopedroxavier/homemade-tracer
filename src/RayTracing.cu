#include <Hitable.cuh>
#include <HitableList.cuh>
#include <Diffuse.cuh>
#include <Glass.cuh>
#include <Metallic.cuh>
#include <Sphere.cuh>
#include <Triangle.cuh>
#include <Vector3.cuh>
#include <Camera.cuh>

#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <wrl.h>
#include <chrono>

#define MINIMUM(a,b) ((a < b) ? (a) : (b))

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "': " << cudaGetErrorString(result) << "\n";
        cudaDeviceReset();
        exit(99);
    }
}

bool parseCommandLineArguments(int& clientWidth, int& clientHeight, int& samples) {
    int argc;
    wchar_t** argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);

    if (argc <= 1) {
        ::LocalFree(argv);
        return true;
    }

    for (size_t i = 1; i < argc; i++) {
        if (::wcscmp(argv[i], L"-w") == 0 || ::wcscmp(argv[i], L"--width") == 0) {
            clientWidth = ::wcstol(argv[++i], nullptr, 10);
            continue;
        }
        if (::wcscmp(argv[i], L"-h") == 0 || ::wcscmp(argv[i], L"--height") == 0) {
            clientHeight = ::wcstol(argv[++i], nullptr, 10);
            continue;
        }
        if (::wcscmp(argv[i], L"-s") == 0 || ::wcscmp(argv[i], L"--samples") == 0) {
            samples = ::wcstol(argv[++i], nullptr, 10);
            continue;
        }

        ::LocalFree(argv);
        return false;
    }

    ::LocalFree(argv);
    return true;
}

// t: parameter to create a gradient effect on background.
__device__ Geometry::Vector3 background(float t) {
    Geometry::Vector3 COLOR_1 = Geometry::Vector3(255.0f, 183.0f, 183.0f);
    Geometry::Vector3 COLOR_2 = Geometry::Vector3(255.0f, 250.0f, 177.0f);
    Geometry::Vector3 col = COLOR_1 * (1.0f - t) + COLOR_2 * t;

    float brightness = 1.0f;
    col = brightness * col / 255.99f;

    return Geometry::Vector3(
        MINIMUM(col.r(), 1.0f),
        MINIMUM(col.g(), 1.0f),
        MINIMUM(col.b(), 1.0f)
    );
}


__global__ void initializeWorld(Geometry::Hitable** list,
    Geometry::Hitable** world,
    Material::Camera** camera,
    int maxX,
    int maxY) {
    list[0] = new Geometry::Sphere(
        new Material::Metallic(Geometry::Vector3(0.2f, 0.2f, 0.2f)),
        Geometry::Vector3(0.0f, -200.5f, -1.0f),
        200.0f);
    list[1] = new Geometry::Triangle(
        new Material::Metallic(Geometry::Vector3(0.3f, 0.3f, 0.3f)),
        Geometry::Vector3(4.0f, -0.5f, 0.0f),
        Geometry::Vector3(-0.2f, 3.5f, 0.2f),
        Geometry::Vector3(0.0f, -0.5f, -4.0f));
    list[2] = new Geometry::Sphere(
        new Material::Glass(1.51f),
        Geometry::Vector3(-0.5f, 0.0f, -0.5f),
        0.5f);
    list[3] = new Geometry::Sphere(
        new Material::Glass(1.51f),
        Geometry::Vector3(0.5f, 0.0f, -1.0f),
        0.5f);
    list[4] = new Geometry::Sphere(
        new Material::Metallic(Geometry::Vector3(1.0f, 1.0f, 1.0f)),
        Geometry::Vector3(-0.5f, 0.2f, -2.0f),
        0.7f);
    list[5] = new Geometry::Sphere(
        new Material::Diffuse(Geometry::Vector3(0.9f, 0.1f, 0.1f)),
        Geometry::Vector3(2.0f, 0.0f, 0.0f),
        0.5f);

    *world = new Geometry::HitableList(list, 6);
    *camera = new Material::Camera(
        Geometry::Vector3(0.0, 2.0f, 3.0f),
        Geometry::Vector3(0.0f, 0.0f, -1.5f),
        Geometry::Vector3(0.0f, 1.0f, 0.0f),
        90, float(maxX) / float(maxY));
}

__global__ void initializeRandomState(int maxX, int maxY, curandState* rng) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= maxX) || (j >= maxY)) {
        return;
    }

    int index = j * maxX + i;
    unsigned long long seed = (unsigned long long)clock64();
    curand_init(seed, index, 0, &rng[index]);
}

__device__ Geometry::Vector3 traceRay(const Geometry::Ray& r, Geometry::Hitable** world, curandState* rng, int width, int height, int maxReflections) {
    Geometry::Vector3 color = Geometry::Vector3(1.0f, 1.0f, 1.0f);
    Geometry::Ray currentRay = r;
    for (int reflection = 0; reflection < maxReflections; reflection++) {
        Geometry::HitRecord rec;
        if ((*world)->hit(currentRay, 0.001f, 1e9f, rec)) {
            Material::ScatterRecord sRec;
            if (rec.material->scatter(currentRay, rec, rng, sRec)) {
                currentRay = sRec.scatteredRay;
                color = color ^ sRec.attenuation;
            }
            else {
                return Geometry::Vector3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            Geometry::Vector3 unitDirection = currentRay.direction() / !currentRay.direction();
            float t = 0.5f * (unitDirection.y() + 1.0f);
            return color ^ background(t);
        }
    }

    return color;
}

__global__ void draw(Geometry::Vector3* buffer, Geometry::Hitable** world, Material::Camera** cam, curandState* rng, int width, int height, int numSamples) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height)) return;

    int pixel_index = j * width + i;

    curandState localRng = rng[pixel_index];
    Geometry::Vector3 col(0, 0, 0);
    for (int s = 0; s < numSamples; s++) {
        float u = float(i + curand_uniform(&localRng)) / float(width);
        float v = float(j + curand_uniform(&localRng)) / float(height);

        Geometry::Ray r = (*cam)->getRay(u, v);
        col += traceRay(r, world, &localRng, width, height, 50);
    }
    col /= float(numSamples);

    buffer[pixel_index] = Geometry::Vector3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b()));
}

int main() {
    int width = 1280, height = 720, samples = 50;
    if (!::parseCommandLineArguments(width, height, samples)) {
        std::cerr << "Usage: RayTracing.exe [options]\n\n" <<
            "options:\n" <<
            "\t-w, --width \t Sets width for generated Image\n" <<
            "\t-h, --height \t Sets height for generated Image\n" <<
            "\t-s, --samples \t Sets number of samples per pixel.\n" << std::endl;
        return 0;
    }

    std::cout << "P3\n" << width << " " << height << "\n255\n";

    int numThreadsX = 16;
    int numThreadsY = 16;

    std::cerr << "Running path tracer to generate " <<
        width << "x" << height << " image " <<
        "using " << numThreadsX << "x" << numThreadsY << "-sized thread blocks " <<
        "with " << samples << " samples per pixel" << std::endl; 

    Geometry::Vector3* pixelBuffer;
    curandState* rng;

    int numPixels = width * height;
    checkCudaErrors(cudaMallocManaged((void**)&pixelBuffer, numPixels * sizeof(Geometry::Vector3)));
    checkCudaErrors(cudaMalloc((void**)&rng, numPixels * sizeof(curandState)));

    Geometry::Hitable** list;
    Geometry::Hitable** world;
    Material::Camera** camera;

    checkCudaErrors(cudaMalloc((void**)&list, 3 * sizeof(Geometry::Hitable*)));
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(Geometry::Hitable*)));
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Material::Camera*)));

    initializeWorld << <1, 1 >> > (list, world, camera, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    auto startTime = std::chrono::high_resolution_clock::now();

    dim3 numBlocks(width / numThreadsX + 1, height / numThreadsY + 1);
    dim3 numThreads(numThreadsX, numThreadsY);

    initializeRandomState<< <numBlocks, numThreads >> > (width, height, rng);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    draw << <numBlocks, numThreads >> > (pixelBuffer, world, camera, rng, width, height, samples);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    auto stopTime = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();

    std::cerr << "Successfully populated the pixel buffer. Process took " << elapsed_time_ms << " milisseconds." << std::endl;

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t index = j * width + i;

            int ir = int(255.99f * pixelBuffer[index].r());
            int ig = int(255.99f * pixelBuffer[index].g());
            int ib = int(255.99f * pixelBuffer[index].b());

            std::cout << ir << " " << ig << " " << ib << std::endl;
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        delete list[i];
    }
    delete* world;
    delete* camera;

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(rng));
    checkCudaErrors(cudaFree(pixelBuffer));

    return 0;
}