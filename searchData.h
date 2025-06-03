#pragma once
#include <array>

struct SearchOutput {
	float avgr[3] = { 0.0f, 0.0f, 0.0f };
	float avgg[3] = { 0.0f, 0.0f, 0.0f };
	float avgb[3] = { 0.0f, 0.0f, 0.0f };
};

struct Point {
	int x = 0;
	int y = 0;
};

struct SolverSettings {
	int ptCount = 0;
	float searchAngle = 0;
	float searchOffset = 0;
	int imWidth = 0;
	int imHeight = 0;
	int agentCount = 0;
	bool interpolatePlacement = true;
};

struct AgentData {
	float x = 0;
	float y = 0;
	float dir = 0;
	float color[3] = { 0, 0, 0 };
};