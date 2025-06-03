#pragma once

#pragma warning(disable: 26495)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)

#define GLEW_STATIC
#define NOMINMAX

//constants
constexpr float _Pi = 3.14159265358979323846f;

//constexpr int WIDTH = 800;
//constexpr int HEIGHT = 450;

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;

constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;

constexpr int MAX_AGENTS = 500000;
constexpr int MAX_SEARCH_PIXELS = 10000;