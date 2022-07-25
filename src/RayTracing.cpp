#include <iostream>
#include <limits>
#include <wrl.h>
#include <shellapi.h>

#include <App.h>
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
const Vector3 COLOR_1 = Vector3(195, 20, 50);
const Vector3 COLOR_2 = Vector3(36, 11, 54);

const float INF = 1e9;

const int MAX_DEPTH = 50;


uint32_t clientWidth = 1280;
uint32_t clientHeight = 720;


// t: parameter to create a gradient effect on background.
Vector3 background(float t) {
    float brightness = 1.5;
    Vector3 col = COLOR_1 * (1.0 - t) + COLOR_2 * t;
    col = brightness * col / 255.99;

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

int runWithoutGPU() {
    int nx = 600;
    int ny = 300;
    int ns = 100;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    Hitable* list[3];
    list[0] = new Sphere(
        new Metallic(Vector3(0.5, 0.5, 0.5)),
        Vector3(0, -200.5, -1),
        200);
    list[1] = new Sphere(
        new Glass(1.2),
        Vector3(0.6, 0.0, -1.5),
        0.5);
    list[2] = new Sphere(
        new Diffuse(Vector3(0.2, 0.2, 0.2)),
        Vector3(-0.6, 0.0, -1.5),
        0.5);

    Hitable* world = new HitableList(list, 3);
    Camera cam = Camera(
        Vector3(-1.5, 1.5, 0.5),
        Vector3(0, 0, -1), 
        Vector3(0, 1, 0), 
        90, float(nx) / float(ny));

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

bool ParseCommandLineArguments() {
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

        ::LocalFree(argv);
        return false;
    }

    ::LocalFree(argv);
    return true;
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCommand) {
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    if (!ParseCommandLineArguments()) {
        return 0;
    }

    ShowMessageOnConsole("Initializing window with width " + std::to_string(clientWidth) + " and height " + std::to_string(clientHeight) + "\n");

    std::unique_ptr<App> mainApp(new App(
            instance, 
            "RayTracingExample", 
            "Real-Time 3D Rendering", 
            showCommand, 
            clientWidth, 
            clientHeight
        )
    );

    try {
        mainApp->Run();
    }
    catch (std::exception ex) {
        MessageBox(
                mainApp->GetWindowHandle(), 
                ex.what(), 
                mainApp->GetWindowTitle().c_str(), 
                MB_ABORTRETRYIGNORE
        );
    }

    return 0;
}