#pragma once

#include "SFML\Graphics.hpp"

#include <array>
#include <vector>

class Agent {
public:

    static float pi;

    static float speed;
    static float turnFactor;
    static float randFactor;
    static float biasFactor;
    static float repulsion;
    static float searchSize;
    static float searchAngle;
    static float searchAngleOffset;

    static float alternate;
    static float globalPeriodR;
    static float globalPeriodG;
    static float globalPeriodB;

    static float distAlternate;
    static float distR;
    static float distG;
    static float distB;
    static float distPeriodR;
    static float distPeriodG;
    static float distPeriodB;

    static float fract;

    static sf::Vector3f palette;

    static bool audioAlternate;
    static sf::Vector3f audioMod;

    static int width;
    static int height;

    static std::vector<sf::Vector2i> normTriangle;

    bool debug = false;

    std::array<sf::Vector3f, 3> avgColors = {
        sf::Vector3f(),
        sf::Vector3f(),
        sf::Vector3f()
    };

    Agent();

    void colorFilters(float time);

    void updateColorBase(sf::Color c, float rate);

    void randomizeColorBase();

    void updateColor();

    void alternateColor(float time);

    sf::Vector2f getPos();

    void updatePos(sf::Image& im);

    void updateDir();

    void mandelBrot(float time);

    float getDir();

    static void generateNormTriangle(int maxPts);

    static void generateNormTriangleSimple();

    void searchImage(sf::Image& im);

private:

    //agent properties
    sf::Vector2f pos;
    sf::Vector2f vel;
    float dir;
    sf::Color color;
    sf::Vector3f colorBase;

    float cosfRatio(float val);

    float cosfRatio(float val, float offset, float scale);

    void alterDir(float delta);

    float compare(sf::Vector3f a, sf::Vector3f b);

    sf::Vector3f norm(sf::Vector3f v);
};