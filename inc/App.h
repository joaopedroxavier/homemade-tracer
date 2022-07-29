#pragma once

#include <Common.h>
#include <Hitable.h>

const int32_t defaultSamplesPerPixel = 100;

class App {
public:
    App(int32_t width, int32_t height, int32_t samplesPerPixel = defaultSamplesPerPixel) :
        _screenWidth(width),
        _screenHeight(height),
        _samplesPerPixel(samplesPerPixel),
        _world() {}

    int32_t GetScreenWidth() const { return _screenWidth; }
    int32_t GetScreenHeight() const { return _screenHeight; }

    void Setup();
    void Run();
private:
    Geometry::Vector3 TraceRay(const Geometry::Ray& r, int depth);

    int32_t                                 _screenWidth;
    int32_t                                 _screenHeight;
    int32_t                                 _samplesPerPixel;

    std::unique_ptr<Geometry::Hitable>      _world;
};

