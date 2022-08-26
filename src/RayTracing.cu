#include <Hitable.cuh>
#include <HitableList.cuh>
#include <Diffuse.cuh>
#include <Glass.cuh>
#include <Metallic.cuh>
#include <Sphere.cuh>
#include <Triangle.cuh>
#include <Quadrilateral.cuh>
#include <Vector3.cuh>
#include <Camera.cuh>

#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <wrl.h>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>

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
    Geometry::Vector3 COLOR_1 = Geometry::Vector3(0.0f, 255.0f, 255.0f);
    Geometry::Vector3 COLOR_2 = Geometry::Vector3(227.0f, 255.0f, 102.0f);
    Geometry::Vector3 col = COLOR_1 * (1.0f - t) + COLOR_2 * t;

    float brightness = 1.0f;
    col = brightness * col / 255.99f;

    return Geometry::Vector3(
        MINIMUM(col.r(), 1.0f),
        MINIMUM(col.g(), 1.0f),
        MINIMUM(col.b(), 1.0f)
    );
}

__host__ void loadObjData(std::ifstream& in, int*& faceTypes, Geometry::Vector3*& vertices, int& numVertices, int& numFaces) {
    std::vector<Geometry::Vector3> vlist;
    std::vector<Geometry::Vector3> positions(1);
    std::vector<int> faces;

    std::cerr << "Attempting to read file..." << std::endl;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream lineSS(line);
        std::string lineType;
        lineSS >> lineType;

        if (lineType == "v") {
            float x, y, z, w;
            lineSS >> x >> y >> z >> w;

            positions.push_back(Geometry::Vector3(x, y, z));
        }

        if (lineType == "f") {
            std::vector<int> refs;
            std::string refStr;
            while (lineSS >> refStr)
            {
                std::istringstream ref(refStr);
                std::string vStr, vtStr, vnStr;
                std::getline(ref, vStr, '/');
                std::getline(ref, vtStr, '/');
                std::getline(ref, vnStr, '/');
                int v = atoi(vStr.c_str());
                v = v >= 0 ? v : (int)positions.size() + v;
                refs.push_back(v);
            }

            if (refs.size() == 3) {
                vlist.push_back(positions[refs[0]]);
                vlist.push_back(positions[refs[1]]);
                vlist.push_back(positions[refs[2]]);
                faces.push_back(1);
            }
            else if (refs.size() == 4) {
                vlist.push_back(positions[refs[0]]);
                vlist.push_back(positions[refs[1]]);
                vlist.push_back(positions[refs[2]]);
                vlist.push_back(positions[refs[3]]);
                faces.push_back(2);
            }

            numFaces++;
        }
    }
    std::cerr << "Finished reading file. Total faces: " << numFaces << std::endl;

    numVertices = (int)vlist.size();

    vertices = new Geometry::Vector3[numVertices];
    faceTypes = new int[numFaces];

    for (size_t i = 0; i < numVertices; i++) {
        vertices[i] = vlist[i];
    }

    for (size_t i = 0; i < numFaces; i++) {
        faceTypes[i] = faces[i];
    }
}


__global__ void initializeWorld(
    Geometry::Hitable** world,
    Geometry::Hitable** list,
    Geometry::Vector3* vertices,
    int* faceTypes,
    int numFaces) {

    int vIdx = 0;
    for (int i = 0; i < numFaces; i++) {
        if (faceTypes[i] == 1) {
            list[i] = new Geometry::Triangle(vertices[vIdx],
                vertices[vIdx + 1],
                vertices[vIdx + 2],
                new Material::Metallic(Geometry::Vector3(1.0f, 1.0f, 1.0f))
            );
            vIdx += 3;
        }
        if (faceTypes[i] == 2) {
            list[i] = new Geometry::Quadrilateral(vertices[vIdx],
                vertices[vIdx + 1],
                vertices[vIdx + 2],
                vertices[vIdx + 3],
                new Material::Metallic(Geometry::Vector3(1.0f, 1.0f, 1.0f))
            );
            vIdx += 4;
        }
    }

    list[numFaces] = new Geometry::Sphere(
        Geometry::Vector3(0.0f, -20000.0f, 0.0f), 
        20000.0f, 
        new Material::Diffuse(Geometry::Vector3(0.4f, 0.4f, 0.9f))
    );

    *world = new Geometry::HitableList(list, numFaces+1);
}

__global__ void initializeCamera(Material::Camera** camera, int width, int height) {
    *camera = new Material::Camera(
        Geometry::Vector3(500.0, 2000.0f, 1500.0f),
        Geometry::Vector3(0.0f, 0.0f, -1.0f),
        Geometry::Vector3(0.0f, 1.0f, 0.0f),
        90, float(width) / float(height));
}

__global__ void initializeRandomState(int width, int height, curandState* rng) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) {
        return;
    }

    int index = j * width + i;
    unsigned long long seed = (unsigned long long)clock64();
    curand_init(seed, index, 0, &rng[index]);
}

__device__ Geometry::Vector3 traceRay(const Geometry::Ray& r, Geometry::Hitable* world, curandState* rng, int width, int height, int maxReflections) {
    Geometry::Vector3 color = Geometry::Vector3(1.0f, 1.0f, 1.0f);
    Geometry::Ray currentRay = r;
    for (int reflection = 0; reflection < maxReflections; reflection++) {
        Geometry::HitRecord rec;
        if ((world)->hit(currentRay, 0.001f, 1e9f, rec)) {
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
        col += traceRay(r, *world, &localRng, width, height, 50);
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

    std::ifstream in("../models/deer.obj");
    int* faceTypes;
    Geometry::Vector3* vertices;

    int numVertices = 0, numFaces = 0;
    loadObjData(in, faceTypes, vertices, numVertices, numFaces);

    Geometry::Hitable** world;
    Geometry::Hitable** list;
    Material::Camera** camera;

    checkCudaErrors(cudaMalloc((void**)&world, sizeof(Geometry::Hitable*)));
    checkCudaErrors(cudaMalloc((void**)&list, (numFaces+1) * sizeof(Geometry::Hitable*)));
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Material::Camera*)));

    int* dFaceTypes;
    Geometry::Vector3* dVertices;
    checkCudaErrors(cudaMalloc((void**)&dFaceTypes, numFaces * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dVertices, numVertices * sizeof(Geometry::Vector3)));
    checkCudaErrors(cudaMemcpy(dFaceTypes, faceTypes, numFaces * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dVertices, vertices, numVertices * sizeof(Geometry::Vector3), cudaMemcpyHostToDevice));

    initializeWorld << <1, 1 >> > (world, list, dVertices, dFaceTypes, numFaces);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    initializeCamera << <1, 1 >> > (camera, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    auto startTime = std::chrono::high_resolution_clock::now();

    dim3 numBlocks(width / numThreadsX + 1, height / numThreadsY + 1);
    dim3 numThreads(numThreadsX, numThreadsY);

    initializeRandomState << <numBlocks, numThreads >> > (width, height, rng);
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

    // free memory

    delete vertices;
    delete faceTypes;

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(dVertices));
    checkCudaErrors(cudaFree(dFaceTypes));
    checkCudaErrors(cudaFree(rng));
    checkCudaErrors(cudaFree(pixelBuffer));

    return 0;
}