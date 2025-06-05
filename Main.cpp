 
/*******************************************************************************
 * Author: Quinn Olson
 * Copyright (c) 2023 Quinn Olson
 *
 * Description:
 *   This file contains code authored by Quinn Olson for the Color Thing.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy of
 *   this software and associated documentation files (the "Software"), to deal in
 *   the Software without restriction, including without limitation the rights to use,
 *   copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 *   Software, and to permit persons to whom the Software is furnished to do so, subject
 *   to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 *   PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION
 *   OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 *   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include "Config.hpp"

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Graphics.hpp>

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <math.h>
#include <execution>
#include <functional>
#include <algorithm>

#include "glm.hpp"
#include "imgui.h"
#include "imgui-SFML.h"

#include "Parameters.hpp"
#include "GuiManager.hpp"

#include "AgentManager.hpp"
#include "CudaManager.hpp"
#include "AudioManager.hpp"

#include "Profiler.cpp"

using namespace std;
using namespace sf;
namespace fs = filesystem;

Vector2f operator*(Vector2f l, Vector2f r) {
    return Vector2f(l.x * r.x, l.y * r.y);
}

Vector2f operator*(float l, Vector2f r) {
    return Vector2f(l * r.x, l * r.y);
}

sf::Image processImageFromFile(string path) {
    sf::Image im;
    im.loadFromFile("images/" + path);
    Vector2f imScale(
        (float)WIDTH / im.getSize().x,
        (float)HEIGHT / im.getSize().y
    );

    //rescale cat image to fit screen
    sf::Image newIm;
    newIm.resize({ WIDTH, HEIGHT });
    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {

            int imx = x / imScale.x;
            int imy = y / imScale.y;

            sf::Vector2u pos = { static_cast<unsigned int>(imx), static_cast<unsigned int>(imy) };

            Color c = im.getPixel(pos);
            newIm.setPixel(pos, c);
        }
    }

    return newIm;


#include "SFML\Graphics.hpp"
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <math.h>
#include <execution>
#include <string>

#define WIDTH 800
#define WINDOW_WIDTH 1920

#define HEIGHT 450
#define WINDOW_HEIGHT 1080

using namespace std;
using namespace sf;

struct Setting {

    string name;
    float delta;
    int rule;
    float* val;

    string shaderVar;

    Setting(string name, float* val, float delta, int rule) {
        this->name = name;
        this->delta = delta;
        this->rule = rule;
        this->val = val;
    }
};

class Agent {
public:

    //static agent parameters
    inline static float speed = 1.0f;
    inline static float maxTurn = _Pi / 180.0f;
    inline static float biasFactor = 2.0f;
    inline static float searchSize = 10;
    inline static float searchAngle = _Pi / 6;
    inline static float alternate = -1.0f;
    inline static float bounce = 1.0f;
    inline static float alternatePeriodR = 9.0f;
    inline static float alternatePeriodG = 10.0f;
    inline static float alternatePeriodB = 11.0f;
    inline static Vector3f palette = Vector3f(1.0f, 1.0f, 1.0f);

    Agent() {
        //spawn agent with a random position, velocity, and base color
        pos.x = rand() % WIDTH;
        pos.y = rand() % HEIGHT;
        dir = (float)(rand() % 10000) / 10000 * 2 * _Pi;
        vel = Vector2f(cos(dir), sin(dir));
        colorBase = Vector3f(
            (float) (rand() % 206 + 50),
            (float) (rand() % 206 + 50),
            (float) (rand() % 206 + 50)
        );
        updateColor();
    }

    //alter agents base color based on palette parameters
    void updateColor() {
        color.r = colorBase.x * palette.x;
        color.g = colorBase.y * palette.y;
        color.b = colorBase.z * palette.z;
    }

    void alternateColor(float time) {
        if(alternatePeriodR != 0)
            color.r = colorBase.x * palette.x * cosf((time / (alternatePeriodR * 1000)) * 2.0f * _Pi);
        if (alternatePeriodG != 0)
            color.g = colorBase.y * palette.y * cosf((time / (alternatePeriodG * 1000)) * 2.0f * _Pi);
        if (alternatePeriodB != 0)
            color.b = colorBase.z * palette.z * cosf((time / (alternatePeriodB * 1000)) * 2.0f * _Pi);
    }

    void updatePos() {
        
        pos.x += vel.x * speed;
        pos.y += vel.y * speed;

        if (bounce > 0) {
            //ensure agent is in bounds and reflects off of boundary
            bool reflect = false;
            if (pos.x < 0) {
                pos.x = 0;
                vel.x = -1 * vel.x;
                reflect = true;
            }
            else if (pos.x >= WIDTH) {
                pos.x = WIDTH - 1;
                vel.x = -1 * vel.x;
                reflect = true;
            }

            if (pos.y < 0) {
                pos.y = 0;
                vel.y = -1 * vel.y;
                reflect = true;
            }
            else if (pos.y >= HEIGHT) {
                pos.y = HEIGHT - 1;
                vel.y = -1 * vel.y;
                reflect = true;
            }

            if (reflect) dir = atan2(vel.y, vel.x);
        }
        else {
            if (pos.x < 0) 
                pos.x = WIDTH - 1;
            else if (pos.x >= WIDTH) 
                pos.x = 0;
            if (pos.y < 0) 
                pos.y = HEIGHT - 1;
            else if (pos.y >= HEIGHT) 
                pos.y = 0;
        }
    }

    void updateDir(Image& im, bool checkOutside) {

        im.setPixel(pos.x, pos.y, color);

        //determine avg color in zones in front, left, and right of agent
        float bias = 0;

        Vector3f me(color.r, color.g, color.b);

        Vector3f l = norm(getAvgColor(im, searchSize, dir + searchAngle / 2, checkOutside));
        Vector3f f = norm(getAvgColor(im, searchSize, dir, checkOutside));
        Vector3f r = norm(getAvgColor(im, searchSize, dir - searchAngle / 2, checkOutside));

        bias += compare(me, l);
        bias -= compare(me, r);
        bias *= 1 - compare(me, f);

        //biasFactor affects how heavily the bias affects the decision
        bias *= biasFactor;

        alterDir(
            ((float) (rand() % 2000) / 1000.0f - 1.0f + bias) * maxTurn
        );
    }

private:

    //agent properties
    Vector2f pos;
    Vector2f vel;
    float dir;
    Color color;
    Vector3f colorBase;

    float sig(float x) {
        return 1.0f / (1 + pow(_Exp1, x));
    }

    void alterDir(float delta) {
        dir += delta;
        vel = Vector2f(cos(dir), sin(dir));
    }

    bool edgeFunc(Vector2i a, Vector2i b, Vector2i c) {
        return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x) >= 0;
    }
    float compare(Vector3f a, Vector3f b) {
        const float maxDist = sqrt(3.0f);
        return (
            maxDist - sqrt(
                pow(a.x - b.x, 2) +
                pow(a.y - b.y, 2) +
                pow(a.z - b.z, 2)
            )
        ) / maxDist;
    }
    Vector3f norm(Vector3i v) {
        float mag = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
        return Vector3f((float)v.x / mag, (float)v.y / mag, (float)v.z / mag);
    }
    Vector3i getAvgColor(Image& im, float dist, float dirDelta, bool checkOutside) {

        vector<Vector2i> points = {
            Vector2i(pos.x, pos.y),
            Vector2i(pos.x + dist * cos(dirDelta + _Pi / 12), pos.y + dist * sin(dirDelta + _Pi / 12)),
            Vector2i(pos.x + dist * cos(dirDelta - _Pi / 12), pos.y + dist * sin(dirDelta - _Pi / 12))
        };

        Vector2f search[2] = {
            Vector2f(
                min(min(points[0].x, points[1].x), points[2].x),
                min(min(points[0].y, points[1].y), points[2].y)
            ),
            Vector2f(
                max(max(points[0].x, points[1].x), points[2].x),
                max(max(points[0].y, points[1].y), points[2].y)
            )
        };

        int count = 0;
        int avg[3] = { 0, 0, 0 };

        for (int x = search[0].x; x <= search[1].x; x++) {
            if (x < 0 || x >= WIDTH && !checkOutside) continue;
            for (int y = search[0].y; y <= search[1].y; y++) {
                if (y < 0 || y >= HEIGHT && !checkOutside) continue;

                bool inside = true;
                inside &= edgeFunc(points[0], points[1], Vector2i(x, y));
                inside &= edgeFunc(points[1], points[2], Vector2i(x, y));
                inside &= edgeFunc(points[2], points[0], Vector2i(x, y));

                if (inside) {
                    count++;
                    Color c;
                    if(!checkOutside) c = im.getPixel(x, y);
                    else c = im.getPixel(x % WIDTH, y % HEIGHT);
                    avg[0] += c.r;
                    avg[1] += c.g;
                    avg[2] += c.b;
                }
            }
        }

        avg[0] = (float)avg[0] / count;
        avg[1] = (float)avg[1] / count;
        avg[2] = (float)avg[2] / count;

        return Vector3i(avg[0], avg[1], avg[2]);
    }

    struct Pt {
        Vector2i pos;
        Pt* next = nullptr;
        Pt(Vector2i pos) {
            this->pos = pos;
        }
    };
};

string formatFloat(float f, int n) {
    float g = pow(10.0f,n);
    string s = to_string(roundf(f * g) / g);
    int delim = s.find(".");
    string back = s.substr(delim + 1, s.length());
    if (back.length() > n) back = back.substr(0, back.length() - (back.length() - n));
    return s.substr(0, delim) + "." + back;
}

Vector2f cross(Vector2f a, Vector2f b) {
    return Vector2f(a.x * b.x, a.y * b.y);
}

string getSettingString(float val, int rule) {

    if (rule == 4 || rule == 5 || rule == 6)
        return val > 0 ? "True" : "False";

    else if (rule == 1) 
        return formatFloat(val * 180.0f / _Pi, 4);
    
    else
        return formatFloat(val, 4);
}

int main()
{
    Vector2f scale((float)WINDOW_WIDTH / WIDTH, (float)WINDOW_HEIGHT / HEIGHT);
    Vector2f invScale(1.0f / scale.x, 1.0f / scale.y);

    float desiredFPS = 1000;
    float frameTimeMS = 1000.0 / desiredFPS;
    int agentCount = 10000;

    Agent::speed = 1.0f;

    srand(time(NULL));

    RenderWindow window(VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Agent Project");

    RenderTexture rt;
    rt.create(WIDTH, HEIGHT);

    Texture tex;
    tex.create(WIDTH, HEIGHT);
    tex.setRepeated(true);

    Image im = tex.copyToImage();

    Sprite sp;
    sp.setTexture(rt.getTexture());
    sp.setScale(scale);

    Font font;
    if (!font.loadFromFile("Roboto-Light.ttf")) {
        cout << "font error" << endl;
        return -1;
    }

    //shader for pixel disperse and dimming effect
    float dimRate = 0.005;
    float disperseFactor = 0.3;
    float checkOutside = -1.0f;
    Shader shader;
    shader.loadFromFile("Shader.frag", Shader::Fragment);
    shader.setUniform("texture", tex);
    shader.setUniform("dimRate", dimRate);
    shader.setUniform("checkOutside", checkOutside > 0 ? true : false);
    shader.setUniform("disperseFactor", disperseFactor);
    shader.setUniform("imageSize", Vector2f(WIDTH, HEIGHT));

    Event event;

    Clock clock;

    Clock timer;
    int frameCount = 0;

    //spawn agents
    vector<Agent> agentList;
    for (int i = 0; i < agentCount; i++) {
        agentList.push_back(Agent());
    }

    //Agent and environment runtime parameters
    //0 - none, 1 - angle, 2 - color, 3 - shader parameter, 3 - color alternator, 4 - activate alternating color, 5 - toggle bounce, 6 - toggle outside sample
    const int settingCount = 16;
    Setting settings[settingCount] = {

        //Agent Options
        Setting("Speed:          \t", &Agent::speed, 0.05f, 0),
        Setting("Max Turn:       \t", &Agent::maxTurn, _Pi / 720.0f, 1),
        Setting("Bias:           \t", &Agent::biasFactor, 0.05f, 0),
        Setting("Search Size:    \t", &Agent::searchSize, 1.0f, 0),
        Setting("Search Angle:   \t", &Agent::searchAngle, _Pi / 360.0f, 1),
        Setting("Bounce:         \t", &Agent::bounce, 0.0f, 5),

        //Shader Options
        Setting("Dim Rate:       \t", &dimRate, 0.0005f, 3),
        Setting("Disperse:       \t", &disperseFactor, 0.005f, 3),
        Setting("Outside Sample: \t", &checkOutside, 0.0f, 6),

        //Color Options
        Setting("PaletteR:       \t", &Agent::palette.x, 0.01f, 2),
        Setting("PaletteG:       \t", &Agent::palette.y, 0.01f, 2),
        Setting("PaletteB:       \t", &Agent::palette.z, 0.01f, 2),
        Setting("Color Alternate:\t", &Agent::alternate, 0.0f, 4),
        Setting("R Alternate:    \t", &Agent::alternatePeriodR, 0.25f, 2),
        Setting("G Alternate:    \t", &Agent::alternatePeriodG, 0.25f, 2),
        Setting("B Alternate:    \t", &Agent::alternatePeriodB, 0.25f, 2)
    };
    settings[6].shaderVar = "dimRate";
    settings[7].shaderVar = "disperseFactor";
    settings[8].shaderVar = "checkOutside";

    int currentSetting = 0;

    int fps = 0;
    Text fpsCounter(std::to_string(fps), font, 20);
    fpsCounter.setFillColor(Color::White);

    CircleShape mouse;
    mouse.setRadius(10);
    mouse.setOrigin(Vector2f(mouse.getRadius() / 2, mouse.getRadius() / 2));
    mouse.setFillColor(Color::White);

    RectangleShape guiBase;
    Vector2f guiPos(0, 0);
    guiBase.setFillColor(Color(50, 50, 50, 150));
    guiBase.setSize(Vector2f(250, 515));
    guiBase.setPosition(guiPos);

    RectangleShape selectedHighlight;
    selectedHighlight.setFillColor(Color(100, 100, 100, 150));
    selectedHighlight.setSize(Vector2f(250, 20));

    Text settingsNameText[settingCount];
    Text settingsValText[settingCount];
    for (int i = 0; i < settingCount; i++) {

        settingsNameText[i] = Text(settings[i].name, font, 20);
        settingsNameText[i].setFillColor(Color::White);
        settingsNameText[i].setPosition(Vector2f(guiBase.getPosition().x, guiBase.getPosition().y + 30 * (i + 1)));

        settingsValText[i] = Text(getSettingString(*settings[i].val, settings[i].rule), font, 20);
        settingsValText[i].setFillColor(Color::White);
        settingsValText[i].setPosition(Vector2f(guiBase.getPosition().x + 160, guiBase.getPosition().y + 30 * (i + 1)));
    }

    selectedHighlight.setPosition(settingsNameText[0].getPosition());

    Clock colorAlternateTimer;

    bool hideGUI = false;

    while (window.isOpen()) {

        //Event management, handle parameter changing
        bool settingAltered = false;
        bool settingSelected = false;
        while (window.pollEvent(event)) {
            if(event.type == Event::Closed)
                window.close();

            if (event.type == Event::KeyPressed) {
                bool fast = Keyboard::isKeyPressed(Keyboard::LShift);
                if (event.key.code == Keyboard::Right) {
                    settingAltered = true;
                    *settings[currentSetting].val += settings[currentSetting].delta * (fast ? 10 : 1);
                }
                if (event.key.code == Keyboard::Left) {
                    settingAltered = true;
                    *settings[currentSetting].val -= settings[currentSetting].delta * (fast ? 10 : 1);
                }
                if (event.key.code == Keyboard::Up) {
                    settingSelected = true;
                    currentSetting = (currentSetting - 1 < 0 ? settingCount - 1 : currentSetting - 1) % settingCount;
                }
                if (event.key.code == Keyboard::Down) {
                    settingSelected = true;
                    currentSetting = (currentSetting + 1) % settingCount;
                }
                if (event.key.code == Keyboard::H) {
                    hideGUI = !hideGUI;
                }
            }
        }

        if (settingAltered || settingSelected) {

            if (settingAltered) {
                
                if (settings[currentSetting].rule == 2) {
                    for_each(execution::par_unseq, agentList.begin(), agentList.end(), [&](auto&& a) {
                        a.updateColor();
                        });
                }
                else if (settings[currentSetting].rule == 3) {
                    shader.setUniform(settings[currentSetting].shaderVar, *settings[currentSetting].val);
                }
                else if (settings[currentSetting].rule == 4) {
                    *settings[currentSetting].val *= -1.0f;
                    colorAlternateTimer.restart();
                }
                else if (settings[currentSetting].rule == 5) {
                    *settings[currentSetting].val *= -1.0f;
                }
                else if (settings[currentSetting].rule == 6) {
                    *settings[currentSetting].val *= -1.0f;
                    shader.setUniform("checkOutside", checkOutside > 0 ? true : false);
                }
            }

            if (settingSelected) {
                selectedHighlight.setPosition(settingsNameText[currentSetting].getPosition() + Vector2f(0, 2));
            }

            settingsValText[currentSetting].setString(
                getSettingString(*settings[currentSetting].val, settings[currentSetting].rule)
            );

        }

        if (clock.getElapsedTime().asMilliseconds() >= frameTimeMS) {
            clock.restart();
            
            //get a copy of current game image and feed into agents for decision making
            im = rt.getTexture().copyToImage();
            for_each(execution::par_unseq, agentList.begin(), agentList.end(), [&](auto&& a) {
                a.updatePos();
                if(Agent::alternate > 0)
                    a.alternateColor(colorAlternateTimer.getElapsedTime().asMilliseconds());
                a.updateDir(im, checkOutside > 0 ? true : false);
            });
            tex.update(im);

            //drawing
            rt.draw(Sprite(tex), &shader);

            if (Mouse::isButtonPressed(Mouse::Left)) {
                Vector2i mousePos = Mouse::getPosition(window);
                mouse.setPosition(cross(Vector2f(mousePos.x, mousePos.y), invScale));
                rt.draw(mouse);
            }
            
            rt.display();

            window.clear();

            window.draw(sp);

            if (!hideGUI) {
                window.draw(guiBase);
                window.draw(selectedHighlight);
                window.draw(fpsCounter);
                for (int i = 0; i < settingCount; i++) {
                    window.draw(settingsNameText[i]);
                    window.draw(settingsValText[i]);
                }
            }
 
            window.display();

            frameCount++;
            if (timer.getElapsedTime().asMilliseconds() > 1000) {
                fpsCounter.setString("FPS: " + to_string(frameCount));
                frameCount = 0;
                timer.restart();
            }
        }
    }
}