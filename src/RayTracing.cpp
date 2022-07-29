#include <App.h>

const int32_t defaultScreenWidth = 1280;
const int32_t defaultScreenHeight = 720;

bool ParseCommandLineArguments(int32_t& clientWidth, int32_t& clientHeight) {
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

int main() {
    int32_t clientWidth = defaultScreenWidth, clientHeight = defaultScreenHeight;
    if (!::ParseCommandLineArguments(clientWidth, clientHeight)) {
        return 0;
    }

    std::unique_ptr<App> mainApp(new App(
        clientWidth,
        clientHeight
    ));

    mainApp->Run();

    return 0;
}