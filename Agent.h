#pragma once

#include "SFML\Graphics.hpp"
#include <math.h>
#include <array>

using namespace std;
using namespace sf;

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

    static Vector3f palette;

    static int width;
    static int height;

    static vector<Vector2i> normTriangle;

    bool debug = false;

    array<Vector3f, 3> avgColors = {
        Vector3f(),
        Vector3f(),
        Vector3f()
    };

    Agent();

    void updateColor();

    void alternateColor(float time);

    Vector2f getPos();

    void updatePos(Image& im);

    void updateDir();

    void mandelBrot(float time);

    float getDir();

    static void generateNormTriangle(int maxPts);

    static void generateNormTriangleSimple();

    void searchImage(Image& im);

private:

    //agent properties
    Vector2f pos;
    Vector2f vel;
    float dir;
    Color color;
    Vector3f colorBase;

    float cosfRatio(float val);

    float cosfRatio(float val, float offset, float scale);

    void alterDir(float delta);

    float compare(Vector3f a, Vector3f b);

    Vector3f norm(Vector3f v);
};