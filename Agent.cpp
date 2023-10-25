#pragma once
#include "Agent.h"
#include <random>
#include <vector>
#include <math.h>
#include <array>
#include <iostream>
#include <execution>

using namespace std;
using namespace sf;

float Agent::pi = 3.141592653589793f;

float Agent::speed = 1.0f;
float Agent::turnFactor = 1.0f;
float Agent::randFactor = 1.0f;
float Agent::biasFactor = 2.0f;
float Agent::repulsion = 0.0f;
float Agent::searchSize = 10;
float Agent::searchAngle = pi / 6;
float Agent::searchAngleOffset = 0;

float Agent::alternate = -1.0f;
float Agent::globalPeriodR = 50.0f;
float Agent::globalPeriodG = 70.0f;
float Agent::globalPeriodB = 30.0f;

float Agent::distAlternate = -1.0f;
float Agent::distR = 500.0f;
float Agent::distG = 1000.0f;
float Agent::distB = 2000.0f;
float Agent::distPeriodR = 30.0f;
float Agent::distPeriodG = 70.0f;
float Agent::distPeriodB = 50.0f;

float Agent::audioAlternate = -1.0f;

Vector3f Agent::palette = Vector3f(1.0f, 1.0f, 1.0f);

Vector3f Agent::audioMod = Vector3f(1, 1, 1);

int Agent::width = 800;
int Agent::height = 450;

vector<Vector2i> Agent::normTriangle(0);

//public members
Agent::Agent() {
    //spawn agent with a random position, velocity, and base color
    pos.x = rand() % width;
    pos.y = rand() % height;
    dir = (float)(rand() % 10000) / 10000 * 2 * pi;
    vel = Vector2f(cos(dir), sin(dir));

    colorBase = Vector3f(
        (double)(rand() % 206 + 50),
        (double)(rand() % 206 + 50),
        (double)(rand() % 206 + 50)
    );

    updateColor();
}

void Agent::colorFilters(float time) {

    updateColor();

    if (alternate > 0 || distAlternate > 0)
        alternateColor(time);

    if (audioAlternate > 0) {
        color.r *= audioMod.x;
        color.g *= audioMod.y;
        color.b *= audioMod.z;
    }

}

void Agent::updateColor() {
    color.r = colorBase.x * palette.x;
    color.g = colorBase.y * palette.y;
    color.b = colorBase.z * palette.z;
}

void Agent::alternateColor(float time) {

    if (alternate > 0) {
        if (globalPeriodR != 0)
            color.r *= cosfRatio(
                time / (globalPeriodR * 1000)
            );
        if (globalPeriodG != 0)
            color.g *= cosfRatio(
                time / (globalPeriodG * 1000)
            );
        if (globalPeriodB != 0)
            color.b *= cosfRatio(
                time / (globalPeriodB * 1000)
            );
    }

    if (distAlternate > 0) {

        Vector2f center(width / 2, height / 2);
        float dist = sqrt(pow(center.x - pos.x, 2) + pow(center.y - pos.y, 2));

        if(distR != 0 && distPeriodR != 0)
            color.r *= cosfRatio(
                dist / distR / cosfRatio(
                    time / (distPeriodR * 1000)
                )
            );
        if (distG != 0 && distPeriodG != 0)
            color.g *= cosfRatio(
                dist / distG / cosfRatio(
                    time / (distPeriodG * 1000)
                )
            );
        if (distB != 0 && distPeriodB != 0)
            color.b *= cosfRatio(
                dist / distB / cosfRatio(
                    time / (distPeriodB * 1000)
                )
            );
    }

    
}

Vector2f Agent::getPos() { return pos; }

//place color and advance
void Agent::updatePos(Image& im) {

    im.setPixel(pos.x, pos.y, color);

    pos.x += vel.x * speed;
    pos.y += vel.y * speed;

    //wrap agent positions around screen
    while (pos.x < 0)
        pos.x += width;
    while(pos.x >= width)
        pos.x -= width;
    while(pos.y < 0)
        pos.y += height;
    while(pos.y >= height)
        pos.y -= height;
}

void Agent::updateDir() {

    //determine avg color in zones in front, left, and right of agent
    //treat colors as a 3d vector for comparing
    //normalize average color to ignore color intensity when deciding which way to go

    Vector3f me = norm(Vector3f(color.r, color.g, color.b));
    Vector3f l = norm(avgColors[0]);
    Vector3f f = norm(avgColors[1]);
    Vector3f r = norm(avgColors[2]);

    float lval = compare(me, l) - repulsion;
    float rval = compare(me, r) - repulsion;
    float mval = compare(me, f) - repulsion;

    float bias = (lval - rval) * (1 - mval) * 255;
    //biasFactor affects how heavily the bias affects the decision
    float val = ((float)(rand() % 2000) / 1000.0f - 1.0f) * randFactor + bias * biasFactor;
    alterDir( val * pi / 180.0f * turnFactor );
    debug = false;
}

void Agent::mandelBrot(float time) {
    float c_re = (pos.x - width / 2.0) * 4.0 / width / 2;
    float c_im = (pos.y - height / 2.0) * 4.0 / height / 2;
    c_re *= 1000.0f / pow(time, 1.5f);
    c_im *= 1000.0f / pow(time, 1.5f);
    c_re += -1.34855;
    c_im += -0.0456763;
    float x = 0, y = 0;
    float max = 500;
    for (float it = 0; it < max; it++) {
        float x_new = x * x - y * y + c_re;
        y = 2 * x * y + c_im;
        x = x_new;
        if (x * x + y * y > 4) {
            //color.r *= it / max * 10;
            color.g *= it / max * 100;
            color.b *= it / max * 50;
            break;
        }
    }
}

float Agent::getDir() {
    return dir;
}

int wrap(int i, int size) {
    if (i < 0)
        return size + i % size;
    if (i >= size)
        return i % size;
    return i;
}

void Agent::searchImage(Image& im) {

    float angles[3] = {
        dir + searchAngleOffset + searchAngle,
        dir,
        dir - searchAngleOffset - searchAngle
    };

    avgColors = { Vector3f(0,0,0), Vector3f(0,0,0), Vector3f(0,0,0) };

    if (normTriangle.size() <= 0)
        return;

    for (int pt = 0; pt < normTriangle.size(); pt++) {

        int ptx = normTriangle[pt].x;
        int pty = normTriangle[pt].y;

        for (int dir = 0; dir < 3; dir++) {

            int ptransx =
                pos.x
                + ptx * cosf(angles[dir])
                - pty * sinf(angles[dir])
            ;
            int ptransy =
                pos.y
                + ptx * sinf(angles[dir])
                + pty * cosf(angles[dir])
            ;

            ptransx = wrap(ptransx, width);
            ptransy = wrap(ptransy, height);

            Color c = im.getPixel(ptransx, ptransy);

            avgColors[dir].x += c.r;
            avgColors[dir].y += c.g;
            avgColors[dir].z += c.b;
        }
    }

    for (int dir = 0; dir < 3; dir++) {
        avgColors[dir].x /= normTriangle.size();
        avgColors[dir].y /= normTriangle.size();
        avgColors[dir].z /= normTriangle.size();
    }
}

//private members

float Agent::cosfRatio(float val) {
    return cosfRatio(val, 0, 1);
}

float Agent::cosfRatio(float val, float offset, float scale) {
    return (cosf(val * 2 * pi) + offset) / scale;
}

void Agent::alterDir(float delta) {
    dir = dir + delta;
    vel = Vector2f(cos(dir), sin(dir));
}

float Agent::compare(Vector3f a, Vector3f b) {
    const float maxDist = sqrt(2); //maximum difference between two 3d normalized vectors 
    float val = 1 - (
        sqrt(
            pow(a.x - b.x, 2) +
            pow(a.y - b.y, 2) +
            pow(a.z - b.z, 2)
        )
    ) / maxDist;
    return val;
}

Vector3f Agent::norm(Vector3f v) {
    const Vector3f zero(1.0f / sqrt(3), 1.0f / sqrt(3), 1.0f / sqrt(3));

    float mag = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
    
    if (mag == 0) {
        return zero;
    }
    return Vector3f(v.x / mag, v.y / mag, v.z / mag);
}

void Agent::generateNormTriangleSimple() {
    normTriangle.clear();

    normTriangle.push_back(Vector2i(
        searchSize,
        0
    ));
}

void Agent::generateNormTriangle(int maxPts) {

    normTriangle.clear();

    Vector2f pts[3] = {
        Vector2f(0, 0),
        Vector2f(
            searchSize * cos(searchAngle / 2),
            searchSize * sin(searchAngle / 2)
        ),
        Vector2f(
            searchSize * cos(-1 * searchAngle / 2),
            searchSize * sin(-1 * searchAngle / 2)
        )
    };

    float xStart = 0;
    float xEnd = pts[1].x;

    if (xEnd < xStart)
        swap(xStart, xEnd);

    float m1 = pts[1].y / pts[1].x;
    float m2 = pts[2].y / pts[2].x;

    if (m2 < m1)
        swap(m1, m2);

    for (int x = floor(xStart); x <= ceil(xEnd); x ++) {  

        float yStart = m1 * x;
        float yEnd = m2 * x;

        for (int y = floor(yStart); y <= ceil(yEnd); y++) {
            normTriangle.push_back(Vector2i(x, y));
            if (normTriangle.size() > maxPts)
                return;
        }
    }
}