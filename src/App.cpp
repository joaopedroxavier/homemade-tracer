#include <App.h>

POINT WindowCenterPoint(int32_t width, int32_t height) {
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    return { (screenWidth - width) / 2, (screenHeight - height) / 2 };

}

void App::Run() {
    MSG message;
    ZeroMemory(&message, sizeof(message));

    _timer.ResetTime();

    while (message.message != WM_QUIT) {
        if (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessage(&message);
        }
        else {
            _timer.Tick([&]() {
                _renderer.Update(_timer);
            });
        }
    }
}

void App::InitializeWindow() {
    ZeroMemory(&_window, sizeof(_window));

    _window.cbSize = sizeof(WNDCLASSEX);
    _window.style = CS_CLASSDC;
    _window.lpfnWndProc = WndProc;
    _window.hInstance = _instance;
    _window.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    _window.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);
    _window.hCursor = LoadCursor(nullptr, IDC_ARROW);
    _window.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    _window.lpszClassName = _windowClass.c_str();

    RECT windowRect = { 0, 0, _screenWidth, _screenHeight };
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, false);

    RegisterClassEx(&_window);

    POINT center = WindowCenterPoint(_screenWidth, _screenHeight);
    _windowHandle = CreateWindow(
            _windowClass.c_str(),
            _windowTitle.c_str(),
            WS_OVERLAPPEDWINDOW,
            center.x,
            center.y,
            windowRect.right - windowRect.left,
            windowRect.bottom - windowRect.top,
            nullptr, 
            nullptr,
            _instance,
            nullptr,
    );

    ShowWindow(_windowHandle, _showCommand);
    UpdateWindow(_windowHandle);
}

LRESULT WINAPI App::WndProc(HWND windowHandle, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(windowHandle, message, wParam, lParam);
}
