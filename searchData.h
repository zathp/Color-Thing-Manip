#pragma once
#include <array>

struct SearchInput {
	float posx = 0;
	float posy = 0;
	float dir = 0;
};

struct SearchOutput {
	float avgr[3] = { 0.0f, 0.0f, 0.0f };
	float avgg[3] = { 0.0f, 0.0f, 0.0f };
	float avgb[3] = { 0.0f, 0.0f, 0.0f };
};

struct Pixel {
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;
	unsigned char a = 0;
};

struct Point {
	int x = 0;
	int y = 0;
};

struct SearchSettings {
	int ptCount = 0;
	float searchAngle = 0;
	float searchOffset = 0;
	int imWidth = 0;
	int imHeight = 0;
	int agentCount = 0;
};