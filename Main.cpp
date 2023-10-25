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

#define GLEW_STATIC
#define NOMINMAX

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Graphics.hpp>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <math.h>
#include <execution>
#include <string>
#include <functional>

#include "fftw3.h"
#include "RtAudio.h"

#include "agent.h"
#include "searchData.h"

//image size
#define WIDTH 800
#define HEIGHT 450

//window size
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define AUDIO_BUFFER_FRAMES 1024
#define AUDIO_SPECTRUM AUDIO_BUFFER_FRAMES / 2 + 1

struct AudioData {
    float inputBuffer[AUDIO_BUFFER_FRAMES];
    fftwf_complex spectrum[AUDIO_SPECTRUM];  // nBufferFrames / 2 + 1
};

AudioData audioData;

int recordCallback(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
    double streamTime, RtAudioStreamStatus status, void* userData) {
    AudioData* data = reinterpret_cast<AudioData*>(userData);

    memcpy(data->inputBuffer, inputBuffer, sizeof(float) * nBufferFrames);

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(nBufferFrames, data->inputBuffer, data->spectrum, FFTW_ESTIMATE);
    fftwf_execute(plan);

    fftwf_destroy_plan(plan);
    return 0;
}

extern "C" void run(int blockSize, const SearchSettings settingData, const SearchInput* inputData,
    cudaGraphicsResource * cudaResource, const Point* ptData, SearchOutput* outputData);

using namespace std;
using namespace sf;

Vector2f operator*(Vector2f l, Vector2f r) {
    return Vector2f(l.x * r.x, l.y * r.y);
}

Vector2f operator*(float l, Vector2f r) {
    return Vector2f(l * r.x, l * r.y);
}

struct Setting {

    string name;
    float delta;
    int rule;
    float* val;
    function<void()> action;

    string shaderVar;

    Text settingText;
    Text valText;

    Setting(string name, float* val, float delta, int rule, string shaderVar, function<void()> action) : Setting(name, val, delta, rule, action) {
        this->shaderVar = shaderVar;
    }

    Setting(string name, float* val, float delta, int rule, function<void()> action) {
        this->name = name;
        this->delta = delta;
        this->rule = rule;
        this->val = val;
        this->action = action;
    }

};

struct SettingGroup {

    Text groupText;
    string name;
    vector<Setting> settings;

    SettingGroup(string name) : name(name) {}

};

string formatFloat(float f, int n) {
    float g = pow(10.0f,n);
    string s = to_string(roundf(f * g) / g);
    int delim = s.find(".");
    string front = s.substr(0, delim);
    if (n == 0) return front;
    string back = s.substr(delim + 1, s.length());
    if (back.length() > n) back = back.substr(0, back.length() - (back.length() - n));
    return front + "." + back;
}

string getSettingString(float val, int rule) {
    //formatting rule: 0 - decimal number, 1 - scaled decimal, 2 - integer number, 3 - angle , 4 - boolean
    if (rule == 4)
        return val > 0 ? "True" : "False";
    else if (rule == 3)
        return formatFloat(val * 180.0f / _Pi, 2) + (char) 248;
    else if (rule == 2)
        return to_string((int)val);
    else if (rule == 1)
        return formatFloat(val * 100.0f, 2);
    else
        return formatFloat(val, 2);
}

void updatePreview(vector<ConvexShape> &searchPreview, Vector2f previewPos, float previewScale) {
    //0: left, 1: front, 2: right
    searchPreview[0].setPointCount(3);
    searchPreview[0].setFillColor(Color(255, 0, 0, 100));
    searchPreview[0].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[0].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f + Agent::searchAngleOffset + Agent::searchAngle * 3.0f / 2.0f),
        previewScale * sin(_Pi / -2.0f + Agent::searchAngleOffset + Agent::searchAngle * 3.0f / 2.0f)
    ) + previewPos);
    searchPreview[0].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f + Agent::searchAngleOffset + Agent::searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f + Agent::searchAngleOffset + Agent::searchAngle / 2.0f)
    ) + previewPos);

    searchPreview[1].setPointCount(3);
    searchPreview[1].setFillColor(Color(0, 255, 0, 100));
    searchPreview[1].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[1].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f + Agent::searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f + Agent::searchAngle / 2.0f)
    ) + previewPos);
    searchPreview[1].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f - Agent::searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f - Agent::searchAngle / 2.0f)
    ) + previewPos);

    searchPreview[2].setPointCount(3);
    searchPreview[2].setFillColor(Color(0, 0, 255, 100));
    searchPreview[2].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[2].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle / 2.0f)
    ) + previewPos);
    searchPreview[2].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle * 3.0f / 2.0f),
        previewScale * sin(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle * 3.0f / 2.0f)
    ) + previewPos);
}

float getMagnitude(int i) {
    float mag = sqrt(
        audioData.spectrum[i][0] * audioData.spectrum[i][0]
        + audioData.spectrum[i][1] * audioData.spectrum[i][1]
    );

    mag;

    return mag;
}

Vector3f processAudio(Vector3f freqFactors) {
    const float maxFreq = 44100.0f / 2;

    const float binWidth = maxFreq / AUDIO_SPECTRUM;

    float bassMaxFreq = 250;
    float lowMidMaxFreq = 500;
    float midMaxFreq = 2000;
    float upperMidMaxFreq = 4000;

    int bassBins = bassMaxFreq / binWidth;
    int lowMidBins = lowMidMaxFreq / binWidth - bassBins;
    int midBins = midMaxFreq / binWidth - bassBins - lowMidBins;
    int upperMidBins = upperMidMaxFreq / binWidth - bassBins - lowMidBins - midBins;

    float bassEnergy = 0;
    float lowMidEnergy = 0;
    float midEnergy = 0;
    float upperMidEnergy = 0;
    float highEnergy = 0;

    for (int i = 0; i < bassBins; i++)
        bassEnergy += getMagnitude(i);

    for (int i = bassBins; i < bassBins + lowMidBins; i++)
        lowMidEnergy += getMagnitude(i);

    for (int i = bassBins + lowMidBins; i < bassBins + lowMidBins + midBins; i++)
        midEnergy += getMagnitude(i);

    for (int i = bassBins + lowMidBins + midBins; i < bassBins + lowMidBins + midBins + upperMidBins; i++)
        upperMidEnergy += getMagnitude(i);

    for (int i = bassBins + lowMidBins + midBins + upperMidBins; i < AUDIO_SPECTRUM; i++) {
        highEnergy += getMagnitude(i);
    }

    float redBucket = bassEnergy;
    float greenBucket = lowMidEnergy + midEnergy;
    float blueBucket = upperMidEnergy + highEnergy;

    redBucket *= freqFactors.x;
    greenBucket *= freqFactors.y;
    blueBucket *= freqFactors.z;

    float maxEnergy = max(
        { redBucket, greenBucket, blueBucket }
    );

    return Vector3f(
        redBucket / maxEnergy,
        greenBucket / maxEnergy,
        blueBucket / maxEnergy
    );
}

int main()
{
    Vector2f scale((float)WINDOW_WIDTH / WIDTH, (float)WINDOW_HEIGHT / HEIGHT);
    Vector2f invScale(1.0f / scale.x, 1.0f / scale.y);

    bool cudaCapable = true;

    constexpr int maxAgents = 500000;

    constexpr int maxPts = 10000;

    srand(time(NULL));

    //--------------------------------------------------------------------------------
    // SFML Initialization
    //--------------------------------------------------------------------------------

    ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 4;

    float desiredFPS = 500;
    float frameTimeMS = 1000.0 / desiredFPS;

    RenderWindow window(VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Color Thing");

    RenderTexture worldTex;
    worldTex.create(WINDOW_WIDTH, WINDOW_HEIGHT);
    Sprite worldSprite(worldTex.getTexture());

    View view = worldTex.getDefaultView();
    worldTex.setView(view);

    Vector2f prevMousePos;
    Vector2f accOffset(0, 0);
    float accZoom = 1;

    RenderTexture rt;
    rt.create(WIDTH, HEIGHT);
    rt.clear();
    rt.display();

    Texture tex;
    tex.create(WIDTH, HEIGHT);
    tex.setRepeated(true);

    Image im = rt.getTexture().copyToImage();

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
    Shader shader;
    shader.loadFromFile("Shader.frag", Shader::Fragment);
    shader.setUniform("texture", tex);
    shader.setUniform("dimRate", dimRate);
    shader.setUniform("disperseFactor", disperseFactor);
    shader.setUniform("imageSize", Vector2f(WIDTH, HEIGHT));

    Event event;

    Clock clock;
    Clock colorAlternateTimer;
    Clock timer;
    int frameCount = 0;


    //--------------------------------------------------------------------------------
    //Audio Visualization Init
    //--------------------------------------------------------------------------------

    RtAudio adc;
    if (adc.getDeviceCount() < 1) {
        std::cout << "\nNo audio devices found!\n";
        exit(1);
    }

    vector<unsigned int> devices = adc.getDeviceIds();
    unsigned int deviceCount = devices.size();
    RtAudio::DeviceInfo info;

    cout << adc.getDeviceCount() << "\n";

    for (int i = 0; i < deviceCount; i++) {

        info = adc.getDeviceInfo(devices[i]);


        // Print, for example, the maximum number of output channels for each device
        cout << "device = " << i;
        cout << ":\n\ID: " << info.ID;
        cout << ":\n\tname: " << info.name;
        cout << "\n\tmaximum output channels: " << info.outputChannels << "\n";
        cout << "\n\tmaximum input channels: " << info.inputChannels << "\n";
    }


    RtAudio::StreamParameters parameters;
    parameters.deviceId = 137;
    parameters.nChannels = 1; // Mono for simplicity
    parameters.firstChannel = 0;

    unsigned int sampleRate = 44100;
    unsigned int bufferFrames = AUDIO_BUFFER_FRAMES;

    adc.openStream(nullptr, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &recordCallback, (void*)&audioData);
    adc.startStream();

    float lowFac = 1.0f;
    float midFac = 1.0f;
    float highFac = 1.0f;

    //--------------------------------------------------------------------------------
    //Agent Initialiations 
    //--------------------------------------------------------------------------------

    Agent::width = WIDTH;
    Agent::height = HEIGHT;

    //spawn agents
    float agentCount = 10000;
    vector<Agent> agentList;
    for (int i = 0; i < agentCount; i++) {
        agentList.push_back(Agent());
    }

    //vector containing indices for agents for multithreading purposes
    vector<int> indices(agentList.size());
    iota(indices.begin(), indices.end(), 0);

    //--------------------------------------------------------------------------------
    // GUI Initialization
    //--------------------------------------------------------------------------------

    Vector2f previewPos(100, 450);
    vector<ConvexShape> searchPreview = { ConvexShape(), ConvexShape(), ConvexShape() };
    updatePreview(searchPreview, previewPos, 50);

    int currentSetting = 0;
    int currentGroup = 0;
    vector<SettingGroup> groups = {
        SettingGroup("Agent Options"),
        SettingGroup("Shader Options"),
        SettingGroup("Color Options"),
        SettingGroup("Oscillation Options")
    };

    float useSimple = -1.0f;
    float useCuda = -1.0f;

    float blockSize = 32;

    //settings manipulated by GUI
    //name, val pointer, change rate, format rule, ? shader uniform name, function called on val change
    //formatting rule: 0 - decimal number, 1 - scaled decimal, 2 - integer number, 3 - angle , 4 - boolean
    groups[0].settings = {
        Setting("Agent Count:", &agentCount, 10, 2, [&]() {
            if (agentCount <= 0)
                agentCount == 1;
            else if (agentCount >= maxAgents)
                agentCount = maxAgents;
            while (agentList.size() < agentCount) {
                agentList.push_back(Agent());
                indices.push_back(agentList.size()-1);
            }
            while (agentList.size() > agentCount) {
                agentList.pop_back();
                indices.pop_back();
            }
        }),
        Setting("Speed:", &Agent::speed, 0.05f, 0, [&]() {}),
        Setting("Turn Factor:", &Agent::turnFactor, 0.01f, 0, [&]() {}),
        Setting("Randomness:", &Agent::randFactor, 0.05f, 0, [&]() {}),
        Setting("Bias:", &Agent::biasFactor, 0.05f, 0, [&]() {}),
        Setting("Repulsion:", &Agent::repulsion, 0.01f, 0, [&]() {}),
        Setting("Search Size:", &Agent::searchSize, 1.0f, 0, [&]() {
            if (useSimple > 0)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
        }),
        Setting("Search Angle:", &Agent::searchAngle, _Pi / 360.0f, 3, [&]() {
            if (useSimple > 0)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
            updatePreview(searchPreview, previewPos, 50);
        }),
        Setting("Angle Offset:", &Agent::searchAngleOffset, _Pi / 360.0f, 3, [&]() {
            updatePreview(searchPreview, previewPos, 50);
        }),
        Setting("Simple Search:", &useSimple, 0.0f, 4, [&]() {
            *groups[currentGroup].settings[currentSetting - 1].val *= -1.0f;
            if (useSimple > 0)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
        }),
        Setting("GPU Search:", &useCuda, 0.0f, 4, [&]() {
            if (cudaCapable) {
                *groups[currentGroup].settings[currentSetting - 1].val *= -1.0f;
            }
        }),
        Setting("Block Size:", &blockSize, 32.0f, 2, [&]() {
            if (blockSize < 32)
                blockSize = 32;
            if (blockSize > 1024)
                blockSize = 1024;
        }),
    };
    
    groups[1].settings = {
        Setting("Dim Rate:", &dimRate, 0.0001f, 1, "dimRate", [&]() {
            shader.setUniform(
                groups[currentGroup].settings[currentSetting - 1].shaderVar,
                *groups[currentGroup].settings[currentSetting - 1].val
            );
        }),
        Setting("Disperse:", &disperseFactor, 0.01f, 0, "disperseFactor", [&]() {
            shader.setUniform(
                groups[currentGroup].settings[currentSetting - 1].shaderVar,
                *groups[currentGroup].settings[currentSetting - 1].val
            );
        })
    };

    groups[2].settings = {
        Setting("PaletteR:", &Agent::palette.x, 0.01f, 0, [&]() {
            std::for_each(execution::par_unseq, agentList.begin(), agentList.end(), [&](auto&& a) {
                a.updateColor();
            });
        }),
        Setting("PaletteG:", &Agent::palette.y, 0.01f, 0, [&]() {
            std::for_each(execution::par_unseq, agentList.begin(), agentList.end(), [&](auto&& a) {
                a.updateColor();
            });
        }),
        Setting("PaletteB:", &Agent::palette.z, 0.01f, 0, [&]() {
            std::for_each(execution::par_unseq, agentList.begin(), agentList.end(), [&](auto&& a) {
                a.updateColor();
            });
        })
    };

    groups[3].settings = {
        Setting("Audio:", &Agent::audioAlternate, 0.0f, 4, [&]() {
            *groups[currentGroup].settings[currentSetting - 1].val *= -1.0f;
        }),
        Setting("\tLow Factor:", &lowFac, 0.01f, 0, [&]() {}),
        Setting("\tMid Factor:", &midFac, 0.01f, 0, [&]() {}),
        Setting("\tHigh Factor:", &highFac, 0.01f, 0, [&]() {}),
        Setting("Global:", &Agent::alternate, 0.0f, 4, [&]() {
            *groups[currentGroup].settings[currentSetting - 1].val *= -1.0f;
            colorAlternateTimer.restart();
        }),
        Setting("\tR Period:", &Agent::globalPeriodR, 0.1f, 0, [&]() {}),
        Setting("\tG Period:", &Agent::globalPeriodG, 0.1f, 0, [&]() {}),
        Setting("\tB Period:", &Agent::globalPeriodB, 0.1f, 0, [&]() {}),
        Setting("Distance:", &Agent::distAlternate, 0.0f, 4, [&]() {
            *groups[currentGroup].settings[currentSetting - 1].val *= -1.0f;
            colorAlternateTimer.restart();
        }),
        Setting("\tR Distance:", &Agent::distR, 1.0f, 2, [&]() {}),
        Setting("\tG Distance:", &Agent::distG, 1.0f, 2, [&]() {}),
        Setting("\tB Distance:", &Agent::distB, 1.0f, 2, [&]() {}),
        Setting("\tR Period:", &Agent::distPeriodR, 0.1f, 0, [&]() {}),
        Setting("\tG Period:", &Agent::distPeriodG, 0.1f, 0, [&]() {}),
        Setting("\tB Period:", &Agent::distPeriodB, 0.1f, 0, [&]() {}),
    };
    
    int fps = 0;
    Text fpsCounter(std::to_string(fps), font, 20);
    fpsCounter.setFillColor(Color::White);

    RectangleShape guiBase;
    Vector2f guiPos(0, 0);
    guiBase.setFillColor(Color(50, 50, 50, 150));
    guiBase.setSize(Vector2f(240, 545));
    guiBase.setPosition(guiPos);

    RectangleShape selectedHighlight;
    selectedHighlight.setFillColor(Color(100, 100, 100, 150));
    selectedHighlight.setSize(Vector2f(240, 20));

    for (int g = 0; g < groups.size(); g++) {
        groups[g].groupText = Text(groups[g].name, font, 20);
        groups[g].groupText.setFillColor(Color::Yellow);
        groups[g].groupText.setPosition(Vector2f(guiBase.getPosition().x, guiBase.getPosition().y + 30));
        for (int s = 0; s < groups[g].settings.size(); s++) {
            groups[g].settings[s].settingText = Text(groups[g].settings[s].name, font, 20);
            groups[g].settings[s].settingText.setFillColor(Color::White);
            groups[g].settings[s].settingText.setPosition(Vector2f(guiBase.getPosition().x, guiBase.getPosition().y + 30 * (s + 2)));

            groups[g].settings[s].valText = Text(getSettingString(*groups[g].settings[s].val, groups[g].settings[s].rule), font, 20);
            groups[g].settings[s].valText.setFillColor(Color::White);
            groups[g].settings[s].valText.setPosition(Vector2f(guiBase.getPosition().x + 160, guiBase.getPosition().y + 30 * (s + 2)));
        }
    }

    selectedHighlight.setPosition(groups[0].groupText.getPosition());

    bool hideGUI = false;


    //--------------------------------------------------------------------------------
    //cuda initialization
    //--------------------------------------------------------------------------------

    //test if host is cuda capable
    int cudaDeviceCount = 0;

    if (cudaGetDeviceCount(&cudaDeviceCount) != cudaSuccess) {
        cudaCapable = false;
        cout << "CUDA driver or device not found\n";
    }

    if (cudaDeviceCount == 0) {
        cudaCapable = false;
        cout << "CUDA device not found\n";
    } else {
        cout << cudaDeviceCount << " CUDA device" << (cudaDeviceCount > 1 ? "s" : "") << " found\n";
    }

    //cuda data
    
    GLint texId = tex.getNativeHandle();

    //Pixel* dev_imageData = 0;
    Point* dev_ptData = 0;
    SearchSettings* dev_settingData = 0;
    SearchInput* dev_inputData = 0;
    SearchOutput* dev_outputData = 0;
    
    //host data
    Point* ptData = 0;
    SearchSettings settingData;
    SearchInput* inputData = 0;
    SearchOutput* outputData = 0;

    struct cudaGraphicsResource* cudaResource;

    function<void()> freeCuda([&]() {

        cudaGraphicsUnregisterResource(cudaResource);

        //cudaFree(dev_imageData);
        cudaFree(dev_inputData);
        cudaFree(dev_outputData);
        cudaFree(dev_ptData);

        
    });

    function<void()> checkCuda([&]() {
        cudaError_t status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
            cout << cudaGetErrorString(status) << "\n";
            freeCuda();
            exit(-1);
        }
    });

    Agent::generateNormTriangle(maxPts);

    if (cudaCapable) {

        inputData = new SearchInput[maxAgents];
        outputData = new SearchOutput[maxAgents];
        ptData = new Point[maxPts];
        
        //set cuda device
        cudaSetDevice(0);
        checkCuda();

        cudaGraphicsGLRegisterImage(&cudaResource, texId,
            GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
        checkCuda();

        //allocate image data
        /*cudaMalloc((void**)&dev_imageData, sizeof(Pixel) * WIDTH * HEIGHT);
        checkCuda();*/

        //allocate cuda memory for i/o for 500,000 agents
        cudaMalloc((void**)&dev_inputData, sizeof(SearchInput) * maxAgents);
        checkCuda();

        cudaMalloc((void**)&dev_outputData, sizeof(SearchOutput) * maxAgents);
        checkCuda();

        //allocate normal triangle coordinate data
        cudaMalloc((void**)&dev_ptData, sizeof(Point) * maxPts);
        checkCuda();

        //load search settings
        settingData.ptCount = 0;

        settingData.searchAngle = Agent::searchAngle;
        settingData.searchOffset = Agent::searchAngleOffset;

        settingData.imWidth = WIDTH;
        settingData.imHeight = HEIGHT;

        settingData.agentCount = agentList.size();
    }

    //--------------------------------------------------------------------------------
    //Main loop
    //--------------------------------------------------------------------------------

    while (window.isOpen()) {

        //Event management, handle parameter changing
        //current setting = 0 indicates a setting group is selected and can be changed, >0 are the settings within that setting group
        bool settingAltered = false;
        bool settingSelected = false;
        while (window.pollEvent(event)) {

            if (event.type == Event::Closed) {
                window.close();
                break;
            }
                
            if (event.type == Event::KeyPressed) {

                //setting manipulation
                float multiplier = 1;
                multiplier *= Keyboard::isKeyPressed(Keyboard::LShift) ? 10 : 1;
                multiplier *= Keyboard::isKeyPressed(Keyboard::LControl) ? 10 : 1;
                if (event.key.code == Keyboard::Right) {
                    if (currentSetting == 0) {
                        currentGroup = ((currentGroup + 1) % groups.size());
                    }
                    else {
                        settingAltered = true;
                        *groups[currentGroup].settings[currentSetting - 1].val += groups[currentGroup].settings[currentSetting - 1].delta * multiplier;
                    }
                }
                if (event.key.code == Keyboard::Left) {
                    if (currentSetting == 0) {
                        currentGroup = ((currentGroup - 1) < 0 ? groups.size() - 1 : currentGroup - 1);
                    }
                    else {
                        settingAltered = true;
                        *groups[currentGroup].settings[currentSetting - 1].val -= groups[currentGroup].settings[currentSetting - 1].delta * multiplier;
                    }
                }
                if (event.key.code == Keyboard::Up) {
                    settingSelected = true;
                    currentSetting = (currentSetting - 1 < 0 ? groups[currentGroup].settings.size() : currentSetting - 1);
                }
                if (event.key.code == Keyboard::Down) {
                    settingSelected = true;
                    currentSetting = (currentSetting + 1) % (groups[currentGroup].settings.size() + 1);
                }

                //hide gui
                if (event.key.code == Keyboard::H) {
                    hideGUI = !hideGUI;
                }

                //reset view
                if (event.key.code == Keyboard::R) {
                    view.setCenter(Vector2f(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2));
                    view.setSize(Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
                    accZoom = 1;
                    worldTex.setView(view);
                }
            }

            //view zooming
            if (event.type == Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta > 0) {
                    view.zoom(0.9f);
                    accZoom *= 0.9f;
                } else {
                    view.zoom(1.1f);
                    accZoom *= 1.1f;
                }
                worldTex.setView(view);
            }
        }

        if (settingAltered || settingSelected) {
            if (settingAltered && currentSetting != 0) 
                groups[currentGroup].settings[currentSetting - 1].action();

            if (settingSelected) {
                if (currentSetting == 0) 
                    selectedHighlight.setPosition(groups[currentGroup].groupText.getPosition());
                else
                    selectedHighlight.setPosition(groups[currentGroup].settings[currentSetting - 1].settingText.getPosition() + Vector2f(0, 2));
            }

            if(currentSetting != 0)
                groups[currentGroup].settings[currentSetting - 1].valText.setString(
                    getSettingString(*groups[currentGroup].settings[currentSetting - 1].val, groups[currentGroup].settings[currentSetting - 1].rule)
            );
        }

        //mouse drag
        float sensitivity = 0.0f;
        if (Mouse::isButtonPressed(Mouse::Right)) {
            Vector2f mousePos = window.mapPixelToCoords(Mouse::getPosition(window));
            Vector2f offset = accZoom * (prevMousePos - mousePos);
            offset.y *= 1;
            if (abs(offset.x) > sensitivity && abs(offset.y) > sensitivity) {
                view.move(offset);
                worldTex.setView(view);
            }
                
        }
        prevMousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

        if (clock.getElapsedTime().asMilliseconds() >= frameTimeMS) {

            clock.restart();
            
            //get a copy of current world image and feed into agents for decision making
            im = rt.getTexture().copyToImage();
            
            //map audio frequencies to color
            if (Agent::audioAlternate > 0) {
                Agent::audioMod = processAudio(Vector3f(lowFac, midFac, highFac));
            }

            //GPU accelerated agent searches
            if (useCuda > 0) {

                //copy image data
                /*const unsigned char* pixelData = im.getPixelsPtr();
                cudaMemcpy(dev_imageData, pixelData, sizeof(unsigned char) * 4 * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
                checkCuda();*/

                //copy setting data
                settingData.ptCount = Agent::normTriangle.size();
                settingData.searchAngle = Agent::searchAngle;
                settingData.searchOffset = Agent::searchAngleOffset;
                settingData.agentCount = agentList.size();

                //copy normal triangle coordinate data
                for (int i = 0; i < Agent::normTriangle.size(); i++) {
                    ptData[i].x = Agent::normTriangle[i].x;
                    ptData[i].y = Agent::normTriangle[i].y;
                }
                cudaMemcpy(dev_ptData, ptData, sizeof(int) * 2 * Agent::normTriangle.size(), cudaMemcpyHostToDevice);
                checkCuda();

                //copy input
                cudaMemcpy(dev_inputData, inputData, sizeof(SearchInput) * agentList.size(), cudaMemcpyHostToDevice);
                checkCuda();

                //run searches
                run(round(blockSize), settingData, dev_inputData, cudaResource, dev_ptData, dev_outputData);
                checkCuda();

                cudaDeviceSynchronize();

                //copy data from gpu
                cudaMemcpy(outputData, dev_outputData, sizeof(SearchOutput) * agentList.size(), cudaMemcpyDeviceToHost);
                checkCuda();

                std::for_each(execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
                    
                    //load output data into agent
                    agentList[i].avgColors[0] =
                        Vector3f(
                            outputData[i].avgr[0],
                            outputData[i].avgg[0],
                            outputData[i].avgb[0]
                        );
                    agentList[i].avgColors[1] =
                        Vector3f(
                            outputData[i].avgr[1],
                            outputData[i].avgg[1],
                            outputData[i].avgb[1]
                        );
                    agentList[i].avgColors[2] =
                        Vector3f(
                            outputData[i].avgr[2],
                            outputData[i].avgg[2],
                            outputData[i].avgb[2]
                        );

                    //perform agent operations
                    agentList[i].colorFilters(colorAlternateTimer.getElapsedTime().asMilliseconds());
                    agentList[i].updateDir();
                    agentList[i].updatePos(im);

                    //copy new agent data for use next frame
                    inputData[i].posx = agentList[i].getPos().x;
                    inputData[i].posy = agentList[i].getPos().y;
                    inputData[i].dir = agentList[i].getDir();
                });
            }
            else {
                std::for_each(execution::par_unseq, indices.begin(), indices.end(), [&](int i) {
                    
                    //search
                    agentList[i].searchImage(im);

                    //decision making based on search results
                    agentList[i].updateDir();

                    //color filters
                    agentList[i].colorFilters(colorAlternateTimer.getElapsedTime().asMilliseconds());

                    //advance and place color
                    agentList[i].updatePos(im);

                });
            }
            
            tex.update(im);

            //draw world to render texture
            //note: this render texture is never cleared, thus changes made are persistent
            rt.draw(Sprite(tex), &shader);

            rt.display();

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
                    worldTex.draw(sp);
                }
            }

            worldTex.display();

            window.draw(worldSprite);

            //GUI drawing
            if (!hideGUI) {
                window.draw(guiBase);
                window.draw(selectedHighlight);
                window.draw(fpsCounter);
                window.draw(groups[currentGroup].groupText);
                for (int i = 0; i < groups[currentGroup].settings.size(); i++) {
                    window.draw(groups[currentGroup].settings[i].settingText);
                    window.draw(groups[currentGroup].settings[i].valText);
                }
                if(currentGroup == 0) for (int i = 0; i < 3; i++) {
                    window.draw(searchPreview[i]);
                }
            }
            
            window.display();

            frameCount++;
            if (timer.getElapsedTime().asMilliseconds() > 1000) {
                //agentList[0].debug = true;
                fpsCounter.setString("FPS: " + to_string(frameCount));
                frameCount = 0;
                timer.restart();
            }
        }
    }

    if (cudaCapable) {
        //free all cuda memory
        freeCuda();

        //reset cuda device
        cudaDeviceReset();

        delete[] inputData;
        delete[] outputData;
        delete[] ptData;
    }

    return 0;
}