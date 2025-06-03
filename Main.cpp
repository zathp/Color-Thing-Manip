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
}

int main()
{

    Vector2f scale((float)WINDOW_WIDTH / WIDTH, (float)WINDOW_HEIGHT / HEIGHT);
    Vector2f invScale(1.0f / scale.x, 1.0f / scale.y);

    bool cudaCapable = true;

	shared_ptr<GLOBAL_SETTINGS> GlobalSettings = make_shared<GLOBAL_SETTINGS>();
	//GUIModifierEditor::setSettings(Settings);

    srand(time(NULL));

    //collect paths to image resources
    fs::path path = fs::current_path() / "./images";
    vector<string> imagePaths;
    if (fs::exists(path) && fs::is_directory(path)) {
        // Iterate over the contents of the directory
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {

                string fileName = entry.path().filename().string();

                string ext = fileName.substr(fileName.find_last_of(".") + 1);

                if (ext == "jpg" || ext == "png") {
                    // Add the image file name to the vector
                    imagePaths.push_back(fileName.substr(fileName.find_last_of("/") + 1));
                } 
            }
        }
    }
    else {
        std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
    }

    //--------------------------------------------------------------------------------
    // SFML Initialization
    //--------------------------------------------------------------------------------

    ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antiAliasingLevel = 4;

    float desiredFPS = 500;
    float frameTimeMS = 1000.0 / desiredFPS;

    RenderWindow window(VideoMode({ WINDOW_WIDTH, WINDOW_HEIGHT }), "Color Thing");
    window.setFramerateLimit(0);

	ImGui::SFML::Init(window);

    RenderTexture worldTex;
    worldTex.resize({WINDOW_WIDTH, WINDOW_HEIGHT}, settings);
    Sprite worldSprite(worldTex.getTexture());

    View view = worldTex.getDefaultView();
    worldTex.setView(view);

    Vector2f prevMousePos;
    Vector2f accOffset(0, 0);
    float accZoom = 1;

    RenderTexture rt;
    rt.resize({ WIDTH, HEIGHT });
    rt.clear();
    rt.display();

    Texture tex;
    tex.resize({ WIDTH, HEIGHT });
    tex.setRepeated(true);

    sf::Image im = rt.getTexture().copyToImage();
   
    Sprite sp(rt.getTexture());
    sp.setScale(scale);

    shared_ptr<Font> font = make_shared<Font>();
    if (!font->openFromFile("Roboto-Light.ttf")) {
        cout << "font error" << endl;
        return -1;
    }

	shared_ptr<Font> font_consolas = make_shared<Font>();
    if (!font_consolas->openFromFile("consolas.ttf")) {
        cout << "font error" << endl;
        return -1;
    }

    //recursiveShader for pixel disperse and dimming effect
	shared_ptr<Shader> recursiveShader = make_shared<Shader>();
    recursiveShader->loadFromFile("Shader.frag", Shader::Type::Fragment);
    recursiveShader->setUniform("texture", tex);
    recursiveShader->setUniform("dimRate", GlobalSettings->Shader.dimRate);
    recursiveShader->setUniform("disperseFactor", GlobalSettings->Shader.disperseFactor);
    recursiveShader->setUniform("imageSize", Vector2f(WIDTH, HEIGHT));

    shared_ptr<Shader> worldShader = make_shared<Shader>();
	worldShader->loadFromFile("WorldShader.frag", Shader::Type::Fragment);
    worldShader->setUniform("texture", rt.getTexture());
    worldShader->setUniform("colorMapEnabled", GlobalSettings->Shader.useColorMap);

    Clock sinceStart;

    Clock updateTimer;
    Clock frameTimer;
    Clock guiFrameTimer;

    Clock guiTimer;

    Clock fpsMeasureTimer;
    int frameCount = 0;
    int updateCount = 0;

	int profileState = 0; //0 - none, 1 - fps, 2 - fps + profiling
	Profiler profiler;

    //--------------------------------------------------------------------------------
    //cuda initialization
    //--------------------------------------------------------------------------------

    CudaManager::checkCudaDevice();

    GLint texId = tex.getNativeHandle();

    window.resetGLStates();

    //--------------------------------------------------------------------------------
    //Audio Visualization Init
    //--------------------------------------------------------------------------------

	shared_ptr<AudioManager> audioManager = make_shared<AudioManager>();
	audioManager->setSettings(GlobalSettings);

    //--------------------------------------------------------------------------------
    //Agent Initialiations 
    //--------------------------------------------------------------------------------

	shared_ptr<vector<AgentManager>> agentManagerList = make_shared<vector<AgentManager>>();

    AgentManager::setGlobalSettings(GlobalSettings);
	
    agentManagerList->emplace_back();
    agentManagerList->at(0).initializeCuda(texId);
    
    worldShader->setUniform("colorMatrix", AgentManager::getColorMapGlsl());

    //--------------------------------------------------------------------------------
    // GUI Initialization
    //--------------------------------------------------------------------------------

    GuiManager guiManager;
    guiManager.setSettings(GlobalSettings);
    guiManager.setAgentManager(agentManagerList);
    guiManager.setAudioManager(audioManager);
    guiManager.setRecursiveShader(recursiveShader);
    guiManager.setWorldShader(worldShader);
    guiManager.init();

    sf::Image matchImage = processImageFromFile(imagePaths[0]);

    int fps = 0;

    Text fpsCounter(*font_consolas, std::to_string(fps), 20);
    fpsCounter.setFillColor(Color::White);
    fpsCounter.setPosition(Vector2f(WINDOW_WIDTH - 300, 0));

	RectangleShape fpsBackground = RectangleShape(Vector2f(350, 30));
	fpsBackground.setFillColor(Color(64, 64, 64, 200));
	fpsBackground.setPosition(Vector2f(WINDOW_WIDTH - 350, 0));

    Text profilerReport(*font_consolas, "", 13);
    profilerReport.setFillColor(Color::White);
    profilerReport.setPosition(Vector2f(WINDOW_WIDTH - 350, 30));

	RectangleShape profilerBackground = RectangleShape(Vector2f(350, 150));
	profilerBackground.setFillColor(Color(64, 64, 64, 200));
	profilerBackground.setPosition(Vector2f(WINDOW_WIDTH - 350, 30));
    
	float guiUpdatesIntervalMS = 1000.0f / 10.f;
    
    bool hideGUI = false;

    //--------------------------------------------------------------------------------
    //Main loop
    //--------------------------------------------------------------------------------
    float updateAcc = 0;
    float frameAcc = 0;

    while (window.isOpen()) {

        float dtUPS = updateTimer.restart().asSeconds();
        float dtFPS = frameTimer.restart().asSeconds();

		updateAcc += dtUPS;
		frameAcc += dtFPS;

        float updateInterval = 1.0f / std::max(static_cast<float>(GlobalSettings->targetUPS), 1.0f);
        float frameInterval = 1.0f / std::max((int)GlobalSettings->targetFPS, 1);

        // Running average frame time (exponential moving average)
        static float emaFrameTime = frameInterval;
        const float emaSmoothing = 0.9f; // Lower = faster response to drops

        emaFrameTime = emaSmoothing * emaFrameTime + (1.0f - emaSmoothing) * dtFPS;

        profiler.start("Total Update");

        //Event management---------------------------------
        //-------------------------------------------------
        bool skipSFGUI = false;
        bool windowClose = false;
        while (std::optional<sf::Event> event = window.pollEvent()) {

			ImGui::SFML::ProcessEvent(window, *event);

            if (event->is<sf::Event::Closed>()) {
                window.close();
                windowClose = true;
            }

            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {

                //hide gui
                if (keyPressed->scancode == sf::Keyboard::Scancode::H) {
                    hideGUI = !hideGUI;
                    skipSFGUI = true;
                }

                //reset view
                if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
                    view.setCenter(Vector2f(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2));
                    view.setSize(Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
                    accZoom = 1;
                    worldTex.setView(view);
                }

                if (keyPressed->scancode == sf::Keyboard::Scancode::P) {
					profileState = (profileState + 1) % 3; //cycle through profile states
                }
            }

            else if (const auto* resizeEvent = event->getIf<sf::Event::Resized>()) {
                Vector2f scale(
					(float)WINDOW_WIDTH / resizeEvent->size.x,
                    (float)WINDOW_HEIGHT / resizeEvent->size.y
                );
                //guiManager.setScale(scale);
            }

            //view zooming
            if (const auto* mouseWheel = event->getIf<sf::Event::MouseWheelScrolled>())
            {
                if (!ImGui::GetIO().WantCaptureMouse) {
                    if (mouseWheel->delta > 0) {
                        view.zoom(0.9f);
                        accZoom *= 0.9f;
                    }
                    else {
                        view.zoom(1.1f);
                        accZoom *= 1.1f;
                    }
                    worldTex.setView(view);
                }
            }

            //pass event to SFGUI
            //if (!hideGUI && !skipSFGUI) guiManager.handleEvent(event);
        }
        if (windowClose) break;
        //mouse drag
        float sensitivity = 0.0f;
        if (Mouse::isButtonPressed(Mouse::Button::Right)) {
            Vector2f mousePos = window.mapPixelToCoords(Mouse::getPosition(window));
            Vector2f offset = accZoom * (prevMousePos - mousePos);
            offset.y *= 1;
            if (abs(offset.x) > sensitivity && abs(offset.y) > sensitivity) {
                view.move(offset);
                worldTex.setView(view);
            }
        }
        prevMousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

        //Update Cycle-------------------------------------
        //-------------------------------------------------
        if(updateAcc >= updateInterval && emaFrameTime <= frameInterval * 1.2f) {
            float guiDeltaTime = guiFrameTimer.restart().asSeconds();

            //get a copy of current world image and feed into agents for decision making
            profiler.start("Image Copy");
            if (!agentManagerList->at(0).getSettings()->Agents.Search.useCuda) im = rt.getTexture().copyToImage();
            profiler.stop("Image Copy");

            //map audio frequencies to color
            profiler.start("Audio Processing");
            if (GlobalSettings->Audio.audioAlternate) {
                audioManager->processAudio();
            }
            profiler.stop("Audio Processing");

            profiler.start("Agent Update");
			
            if (agentManagerList->at(0).getSettings()->Agents.Search.useCuda) {
                //CUDA update algorithm---------------------
                agentManagerList->at(0).update_cuda(sinceStart.getElapsedTime().asMilliseconds());
            }
            else { 
				//CPU update algorithm----------------------
                agentManagerList->at(0).update_cpu(im, sinceStart.getElapsedTime().asMilliseconds());
            }
            profiler.stop("Agent Update");

			//when using cpu, update the texture with the modified image
            profiler.start("World Update");
            if (!agentManagerList->at(0).getSettings()->Agents.Search.useCuda)
                tex.update(im);

            rt.draw(Sprite(tex), recursiveShader.get());
            rt.display();

            //when using cuda we need to manually overwrite the texture
            if (agentManagerList->at(0).getSettings()->Agents.Search.useCuda)
                tex.update(rt.getTexture());

            profiler.stop("World Update");
            profiler.stop("Total Update");

			updateAcc -= updateInterval;
            updateAcc = std::min(updateAcc, updateInterval * 3.f);

            updateCount++;
        }

        //Draw Cycle------------------------------------------------------------------
        //----------------------------------------------------------------------------
        if(frameAcc >= frameInterval) {
            profiler.start("World Draw");
            window.clear();

            int tileFactor = 2 * (int)round(accZoom);
            tileFactor = max(tileFactor, 1);

            worldTex.clear();

            Vector2f currentCenter = worldTex.getView().getCenter();
            int tileOffsetX = (int)round(currentCenter.x / WINDOW_WIDTH) * WINDOW_WIDTH;
            int tileOffsetY = (int)round(currentCenter.y / WINDOW_HEIGHT) * WINDOW_HEIGHT;

            for (float tilex = -1 * tileFactor * WINDOW_WIDTH; tilex <= tileFactor * WINDOW_WIDTH; tilex += WINDOW_WIDTH) {
                for (float tiley = -1 * tileFactor * WINDOW_HEIGHT; tiley <= tileFactor * WINDOW_HEIGHT; tiley += WINDOW_HEIGHT) {
                    sp.setPosition(Vector2f(tilex + tileOffsetX, tiley + tileOffsetY));
                    worldTex.draw(sp, worldShader.get());
                }
            }

            worldTex.display();

            window.draw(worldSprite);
            profiler.stop("World Draw");

            profiler.start("GUI Draw");
            //GUI drawing
            if (!hideGUI && false) {
                bool shouldRefreshGUI = false;
                if (guiTimer.getElapsedTime().asMilliseconds() >= guiUpdatesIntervalMS) {
                    shouldRefreshGUI = true;
                    guiTimer.restart();
                }

                float guiDeltaTime = guiFrameTimer.restart().asSeconds();
                //guiManager.update(guiDeltaTime, shouldRefreshGUI);

                //window.draw(guiManager.getSprite());
            }

			ImGui::SFML::Update(window, sf::seconds(guiFrameTimer.restart().asSeconds()));

            guiManager.draw();

			ImGui::SFML::Render(window);
            ImGui::EndFrame();

            if (profileState >= 1) {
				window.draw(fpsBackground);
                window.draw(fpsCounter);
            }
            if (profileState == 2) {
				window.draw(profilerBackground);
                window.draw(profilerReport);
            }
            window.display();

            frameCount++;
            if (fpsMeasureTimer.getElapsedTime().asMilliseconds() > 1000) {

                std::ostringstream ss;
                ss << "FPS: " << std::setw(5) << std::setfill(' ') << frameCount
                    << "  | UPS: " << updateCount;

                fpsCounter.setString(ss.str());

                frameCount = 0;
                updateCount = 0;
                profilerReport.setString(profiler.getReport());

                fpsMeasureTimer.restart();
            }
            profiler.stop("GUI Draw");

			frameAcc -= frameInterval;
            frameAcc = std::min(frameAcc, frameInterval * 3.f);
        }
    }

	ImGui::SFML::Shutdown();

    return 0;
}