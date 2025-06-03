#pragma once

#define _Pi 3.14159265358979323846f

#include <SFML/Graphics.hpp>
#include <vector>
#include <array>

struct AGENT_SETTINGS {
	float speed = 1.0f;
	float turnFactor = 1.0f;
	float randFactor = 1.0f;
	float biasFactor = 1.0f;
	float repulsion = 1.0f;
};

struct SEARCH_SETTINGS {
	float searchSize = 10.0f;
	float searchAngle = _Pi / 6.0f;
	float searchAngleOffset = 0.f;
};

struct OSCILLATION_SETTINGS {
	bool timeOscillationEnabled = false;
	float globalPeriodR = 50.0f;
	float globalPeriodG = 70.0f;
	float globalPeriodB = 30.0f;

	bool distAlternate = false;
	float distR = 500.0f;
	float distG = 1000.0f;
	float distB = 2000.0f;
	float distPeriodR = 30.0f;
	float distPeriodG = 70.0f;
	float distPeriodB = 50.0f;
};

struct SETTINGS {

	AGENT_SETTINGS Agents;

	OSCILLATION_SETTINGS Oscillation;

	sf::Vector3f colorFilter = { 1.f, 1.f, 1.f };

	bool audioAlternate;
	sf::Vector3f audioMod;

	int width;
	int height;

	std::vector<sf::Vector2i> normTriangle;

	bool debug;

	int maxSearchPixels = 10000;
	int maxAgents = 500000;
};