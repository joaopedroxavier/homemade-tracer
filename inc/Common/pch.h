#pragma once

// The min/max macros conflict with like-named member functions.
// Only use std::min and std::max defined in <algorithm>.
#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <shellapi.h>

#include <wrl.h>

// STL
#include <algorithm>
#include <cassert>
#include <chrono>

// DirectX 12
#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectXMath.h>
#include <d3dx12.h>
