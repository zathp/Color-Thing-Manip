#pragma once

#include "SFML\Graphics.hpp"
#include "Config.hpp"

#include <array>
#include <vector>
#include <random>
#include <math.h>
#include <iostream>
#include <execution>

#include <glm.hpp>

#include "Parameters.hpp"

class Agent {
public:

    std::array<sf::Vector3f, 3> avgColors = {
        sf::Vector3f(),
        sf::Vector3f(),
        sf::Vector3f()
    };

    Agent(int id);

    void updateColorBase(sf::Color c, float rate);

    void randomizeColorBase();

    sf::Vector2f getPos();

    void updatePos(std::shared_ptr<GROUP_SETTINGS> Settings, sf::Image& im);

	void updatePos(std::shared_ptr<GROUP_SETTINGS> Settings);

    void updateDir(std::shared_ptr<GROUP_SETTINGS> Settings);

    sf::Color getFinalColor() { return colorFinal; }
	void getFinalColor(float* pos) {
		pos[0] = colorFinal.r;
		pos[1] = colorFinal.g;
		pos[2] = colorFinal.b;
	}

    float getDir();

    void searchImage(std::shared_ptr<GROUP_SETTINGS> Settings, sf::Image& im, std::vector<sf::Vector2i>& normPixels);

	int getId() {
		return id;
	}

	static void setGlobalSettings(std::shared_ptr<GLOBAL_SETTINGS> settings) {
		GlobalSettings = settings;
	}

    void applyFilter(sf::Vector3f filter);

    void applyMapping(glm::mat3 colorMap);

	void applyMappingHSV(const glm::mat3& colorMap);

	void applyHueShift(float hueShift);

	void applyColorTransformations(std::shared_ptr<GROUP_SETTINGS> Settings, float time);

	static void setColorMap(glm::mat3 colorMap) {
		Agent::colorMap = colorMap;
	}

    static sf::Color hueShift(sf::Color, float);

private:

    static std::shared_ptr<GLOBAL_SETTINGS> GlobalSettings;

    //agent properties
    sf::Vector2f prevPos;
    sf::Vector2f pos;
    sf::Vector2f vel;
    float dir;

    float lastTurnBias;

    //base color holds an unchanging color,
	//colorFiltered is the color post static filtering/transformation
    //the final color will be colorFiltered after frame by frame filters/transformations

    sf::Color colorFinal;
    sf::Color colorBase;

    unsigned int id;

    void alterDir(std::shared_ptr<GROUP_SETTINGS> Settings, float delta);

    static float cosfRatio(float val);

    static float cosfRatio(float val, float offset, float scale);

    static float compare(sf::Vector3f a, sf::Vector3f b);

    static sf::Vector3f norm(sf::Vector3f v);

	static glm::mat3 colorMap;
};