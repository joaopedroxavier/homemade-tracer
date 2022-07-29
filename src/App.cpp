#include <App.h>
#include <Hitable.h>
#include <HitableList.h>
#include <Metallic.h>
#include <Glass.h>
#include <Diffuse.h>
#include <Sphere.h>
#include <Vector3.h>
#include <Camera.h>

void App::Setup() {
    Geometry::Hitable** list = new Geometry::Hitable*[3];

    list[0] = new Geometry::Sphere(
        new Material::Metallic(Geometry::Vector3(0.5, 0.5, 0.5)),
        Geometry::Vector3(0, -200.5, -1),
        200);

    list[1] = new Geometry::Sphere(
        new Material::Glass(1.2),
        Geometry::Vector3(0.6, 0.0, -1.5),
        0.5);

    list[2] = new Geometry::Sphere(
        new Material::Diffuse(Geometry::Vector3(0.2, 0.2, 0.2)),
        Geometry::Vector3(-0.6, 0.0, -1.5),
        0.5);

    Geometry::Hitable* world = new Geometry::HitableList(list, 3);
    _world = std::unique_ptr<Geometry::Hitable>(std::move(world));
}

const Geometry::Vector3 NO_COLOR = Geometry::Vector3(0.0, 0.0, 0.0);
const Geometry::Vector3 COLOR_1 = Geometry::Vector3(195, 20, 50);
const Geometry::Vector3 COLOR_2 = Geometry::Vector3(36, 11, 54);

const float INF = 1e9;

const int MAX_DEPTH = 50;


uint32_t clientWidth = 1280;
uint32_t clientHeight = 720;


// t: parameter to create a gradient effect on background.
Geometry::Vector3 background(float t) {
    float brightness = 1.5;
    Geometry::Vector3 col = COLOR_1 * (1.0 - t) + COLOR_2 * t;
    col = brightness * col / 255.99;

    return Geometry::Vector3(
        std::min(col.r(), 1.0f),
        std::min(col.g(), 1.0f),
        std::min(col.b(), 1.0f)
    );
}

Geometry::Vector3 App::TraceRay(const Geometry::Ray& r, int depth) {
    if (depth > MAX_DEPTH) {
        return NO_COLOR;
    }

    Geometry::HitRecord rec;
    if (_world->hit(r, 0.001, INF, rec)) {
        Material::ScatterRecord sRec = rec.material->scatter(r, rec);
        if (sRec.didScatter) {
            return sRec.attenuation ^ TraceRay(sRec.scatteredRay, depth + 1);
        }
        else {
            return NO_COLOR;
        }
    }
    else {
        Geometry::Vector3 unitDirection = r.direction() / (!r.direction());
        float t = 0.5 * (unitDirection.y() + 1.0);
        return background(t);
    }
}

void App::Run() {
    std::cout << "P3\n" << _screenWidth << " " << _screenHeight << "\n255\n";

    Setup();

    Material::Camera cam = Material::Camera(
        Geometry::Vector3(-1.5, 1.5, 0.5),
        Geometry::Vector3(0, 0, -1), 
        Geometry::Vector3(0, 1, 0), 
        90, float(_screenWidth) / float(_screenHeight));

    for (int j = _screenHeight - 1; j >= 0; j--) {
        for (int i = 0; i < _screenWidth; i++) {
            Geometry::Vector3 col(0, 0, 0);
            for (int s = 0; s < _samplesPerPixel; s++) {
                float ru = Common::GetRandomFloat();
                float rv = Common::GetRandomFloat();

                float u = float(i + ru) / float(_screenWidth);
                float v = float(j + rv) / float(_screenHeight);

                Geometry::Ray r = cam.getRay(u, v);
                col += TraceRay(r, 0);
            }
            col /= float(_samplesPerPixel);
            col = Geometry::Vector3(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
            int ir = int(255.99 * col.r());
            int ig = int(255.99 * col.g());
            int ib = int(255.99 * col.b());

            std::cout << ir << " " << ig << " " << ib << "\n";
            //std::cerr << returned << ": " << ir << " " << ig << " " << ib << std::endl;
        }
    }

    std::cerr << "Successfully populated the PPM file." << std::endl;
}
