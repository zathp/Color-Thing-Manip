#pragma once
#include <SFML/Graphics.hpp>
#include "imgui.h"

#include <array>

#include "Config.hpp"
#include "Parameters.hpp"
#include "Agent.hpp"

class SearchPreview {

	sf::RenderTexture tex;
	sf::RenderTexture tex_rot;

	float previewScale = 75.0f;
	sf::Vector2f previewPos = { 100, 100 };
	sf::Vector2f previewSize = { 200, 200 };
	float _Pi = 3.14159265358979323846f;
public:
	SearchPreview();

	void update(SEARCH_SETTINGS settings);

	void drawGuiImage() {
		ImGui::Image(tex_rot.getTexture().getNativeHandle(), ImVec2(previewSize.x, previewSize.y));
	}

	//sfg::Image::Ptr getSfgImage();
};

class AudioVisualizer {

	sf::RenderTexture tex;
	sf::RenderTexture tex_rot;

	float scaleFactor = 100.0f;
	float barWidth = 20.0f;
	float barSpacing = 10.0f;
	float smoothedEnergies[3] = { 0, 0, 0 };
	float lastPeaks[3] = { 0.f, 0.f, 0.f };
	float lastCentroids[3] = { 0.f, 0.f, 0.f };
	sf::Vector2f previewSize = { 200, 100 };
	
public:
	AudioVisualizer();
	void update(const float* energies, const float* peaks, const float* centroids, const float* peakDrift, const float* centroidDrift);

	void drawGuiImage() {
		ImGui::Image(tex_rot.getTexture().getNativeHandle(), ImVec2(previewSize.x, previewSize.y));
	}
	//sfg::Image::Ptr getSfgImage();
};
