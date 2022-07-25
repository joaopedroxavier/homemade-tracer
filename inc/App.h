#pragma once

#include <Timer.h>
#include <Renderer.h>

const int32_t defaultScreenWidth = 1280;
const int32_t defaultScreenHeight = 720;

class App {
public:
    App(HINSTANCE instance, const std::string& windowClass, const std::string& windowTitle, 
            uint32_t showCommand, int32_t width, int32_t height) :
        _instance(instance), 
        _windowClass(windowClass), 
        _windowTitle(windowTitle), 
        _showCommand(showCommand), 
        _screenWidth(width), 
        _screenHeight(height) {
            InitializeWindow();
    }

    HINSTANCE GetInstance() const { return _instance; }
    HWND GetWindowHandle() const{ return _windowHandle; }

    const WNDCLASSEX& GetWindow() const { return _window; }
    const std::string& GetWindowClass() const { return _windowClass; }
    const std::string& GetWindowTitle() const { return _windowTitle; }

    int32_t GetScreenWidth() const { return _screenWidth; }
    int32_t GetScreenHeight() const { return _screenHeight; }

    void Run();
    void InitializeWindow();

private:
    Rendering::Timer        _timer;
    Rendering::Renderer     _renderer;

    HINSTANCE               _instance;
    std::string             _windowClass;
    std::string             _windowTitle;
    uint32_t                _showCommand;
    
    HWND                    _windowHandle;
    WNDCLASSEX              _window;

    int32_t                 _screenWidth;
    int32_t                 _screenHeight;

    static LRESULT WINAPI WndProc(HWND windowHandle, UINT message, WPARAM wParam, LPARAM lParam);
};

