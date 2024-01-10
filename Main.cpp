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

#pragma warning(disable: 26495)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)

#define GLEW_STATIC
#define NOMINMAX

 //image size
#define WIDTH 800
#define HEIGHT 450

//window size
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

//input audio parameters
#define AUDIO_FREQUENCY 44100.0f
#define AUDIO_BUFFER_FRAMES 512
#define AUDIO_SPECTRUM 257 //buffer frames / 2 + 1

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Graphics.hpp>

#include <SFGUI/SFGUI.hpp>
#include <SFGUI/Widgets.hpp>
#include <SFGUI/Image.hpp>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <filesystem>
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
#include "GUI.hpp"

struct AudioData {
    float inputBuffer[AUDIO_BUFFER_FRAMES];
    fftwf_complex spectrum[AUDIO_SPECTRUM];
};

AudioData AUDIO_BUFFER;

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
using namespace sfg;
namespace fs = filesystem;

Vector2f operator*(Vector2f l, Vector2f r) {
    return Vector2f(l.x * r.x, l.y * r.y);
}

Vector2f operator*(float l, Vector2f r) {
    return Vector2f(l * r.x, l * r.y);
}

void updatePreview( RenderTexture& tex, sfg::Image::Ptr sfgImage,
    Vector2f previewPos, float previewScale) {

    vector<ConvexShape> searchPreview = { ConvexShape(), ConvexShape(), ConvexShape() };

    //0: left, 1: front, 2: right
    searchPreview[0].setPointCount(3);
    searchPreview[0].setFillColor(Color(255, 0, 0, 150));
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
    searchPreview[1].setFillColor(Color(0, 255, 0, 150));
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
    searchPreview[2].setFillColor(Color(0, 0, 255, 150));
    searchPreview[2].setPoint(0, Vector2f(0, 0) + previewPos);
    searchPreview[2].setPoint(1, Vector2f(
        previewScale * cos(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle / 2.0f),
        previewScale * sin(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle / 2.0f)
    ) + previewPos);
    searchPreview[2].setPoint(2, Vector2f(
        previewScale * cos(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle * 3.0f / 2.0f),
        previewScale * sin(_Pi / -2.0f - Agent::searchAngleOffset - Agent::searchAngle * 3.0f / 2.0f)
    ) + previewPos);

    tex.clear(Color::Transparent);
    for (int i = 0; i < searchPreview.size(); i++) {
		tex.draw(searchPreview[i]);
	}
    tex.display();

    sfgImage->SetImage(tex.getTexture().copyToImage());
}

float getMagnitude(int i) {
    float mag = sqrt(
        AUDIO_BUFFER.spectrum[i][0] * AUDIO_BUFFER.spectrum[i][0]
        + AUDIO_BUFFER.spectrum[i][1] * AUDIO_BUFFER.spectrum[i][1]
    );

    //mag = log(mag+1);

    return mag;
}

Vector3f processAudio(array<array<float,2>,3> &bucketFrequencies, Vector3f freqFactors) {
    //bass:         250hz
    //low mid:      500hz
    //mid:          2000hz
    //upper mid:    4000hz
    //high:         4000+hz

    const float maxFreq = AUDIO_FREQUENCY / 2;
    const float binWidth = maxFreq / AUDIO_SPECTRUM;

    float energies[3] = { 0,0,0 };
    for (int i = 0; i < 3; i++) {
        int start = bucketFrequencies[i][0] / binWidth;
        int end = bucketFrequencies[i][1] / binWidth;

        for (int bin = start; bin < end; bin++)
            energies[i] += getMagnitude(bin);
    }

    energies[0] *= freqFactors.x;
    energies[1] *= freqFactors.y;
    energies[2] *= freqFactors.z;

    float maxEnergy = max(
        { energies[0], energies[1], energies[2] }
    );
    if (maxEnergy == 0) maxEnergy = 1;

    return Vector3f(
        energies[0] / maxEnergy,
        energies[1] / maxEnergy,
        energies[2] / maxEnergy
    );
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
    newIm.create(WIDTH, HEIGHT);
    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {

            int imx = x / imScale.x;
            int imy = y / imScale.y;

            Color c = im.getPixel(imx, imy);
            newIm.setPixel(x, y, c);
        }
    }

    return newIm;
}

int main()
{
    Vector2f scale((float)WINDOW_WIDTH / WIDTH, (float)WINDOW_HEIGHT / HEIGHT);
    Vector2f invScale(1.0f / scale.x, 1.0f / scale.y);

    bool cudaCapable = true;

    constexpr int maxAgents = 500000;

    constexpr int maxPts = 10000;

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
    settings.antialiasingLevel = 4;

    float desiredFPS = 500;
    float frameTimeMS = 1000.0 / desiredFPS;

    RenderWindow window(VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Color Thing");
    window.setFramerateLimit(0);

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

    sf::Image im = rt.getTexture().copyToImage();

    

    Sprite sp;
    sp.setTexture(rt.getTexture());
    sp.setScale(scale);

    shared_ptr<Font> font = make_shared<Font>();
    if (!font->loadFromFile("Roboto-Light.ttf")) {
        cout << "font error" << endl;
        return -1;
    }

    //shader for pixel disperse and dimming effect
    float dimRate = 0.005f;
    float disperseFactor = 0.3f;
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

    vector<unsigned int> devices = adc.getDeviceIds();
    vector<unsigned int> inputDevices;
    vector<string> deviceNames;
    for (int i = 0; i < devices.size(); i++) {
        RtAudio::DeviceInfo info = adc.getDeviceInfo(devices[i]);

        if (info.inputChannels > 0) {
            inputDevices.push_back(devices[i]);
            deviceNames.push_back(info.name);
        }

    }
    int deviceIndex = 0;
    int deviceId = 0;
    if (inputDevices.size() == 0)
        cout << "No audio input devices detected\n";
    else {
        cout << inputDevices.size() << " input devices detected\n";
    }

    RtAudio::StreamParameters parameters;
    if (devices.size() > 0) parameters.deviceId = devices[deviceId];
    parameters.nChannels = 1;
    parameters.firstChannel = 0;

    unsigned int bufferFrames = AUDIO_BUFFER_FRAMES;

    //settings to manipulate visualization
    float rFac = 1.0f;
    float gFac = 1.0f;
    float bFac = 1.0f;

    //frequencies associated with each color
    array<array<float, 2>, 3> frequencies = {
        array<float,2>{0, 250},
        array<float,2>{250, 2000},
        array<float,2>{2000, 44100}
    };

    //--------------------------------------------------------------------------------
    //Agent Initialiations 
    //--------------------------------------------------------------------------------

    Agent::width = WIDTH;
    Agent::height = HEIGHT;

    //spawn agents
    int agentCount = 10000;
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

    SFGUI sfgui;
    Desktop desktop;

    desktop.GetEngine().LoadThemeFromFile("style.theme");
    desktop.GetEngine().GetResourceManager().SetDefaultFont(font);

    sf::Image matchImage = processImageFromFile(imagePaths[0]);

    GUIBase gui;
    gui.addToDesktop(desktop);
    gui.Refresh();

    RenderTexture guiTexture;
    guiTexture.create(guiOptions.guiSize.x, guiOptions.guiSize.y);
    Sprite guiSprite(guiTexture.getTexture());

    bool useSimple = false;
    bool useCuda = false;
    bool imageMatch = false;

    RenderTexture searchPreviewTex;
    searchPreviewTex.create(200, 200);

    sfg::Image::Ptr searchPreviewGUIImage;
    searchPreviewGUIImage = sfg::Image::Create();
    searchPreviewGUIImage->SetImage(searchPreviewTex.getTexture().copyToImage());

    Vector2f previewPos(100, 100);
    auto updateSearchPreview = [&]() {
        updatePreview(searchPreviewTex, searchPreviewGUIImage, previewPos, 75);
    };
    
    updateSearchPreview();

    shared_ptr<GUITab> settingsTab = make_shared<GUITab>("Parameters");
    gui.addTab(settingsTab);

    //agent settings
    shared_ptr<DropDown> agentOptions = make_shared<DropDown>("Agent");
    settingsTab->addItem(*agentOptions);

    vector<shared_ptr<SettingBase<void>>> agentOptionsList;
    shared_ptr<Setting<int>> s_agentCount = make_shared<Setting<int>>(
        &agentCount, "Count", [&] {
            if (agentCount <= 0)
                agentCount = 1;
            else if (agentCount >= maxAgents)
                agentCount = maxAgents;
            while (agentList.size() < agentCount) {
                agentList.push_back(Agent());
                indices.push_back(agentList.size() - 1);
            }
            while (agentList.size() > agentCount) {
                agentList.pop_back();
                indices.pop_back();
            }
        }
    );
    agentOptions->addItem(*s_agentCount);

    shared_ptr<Setting<float>> s_agentSpeed = make_shared<Setting<float>>(
        &Agent::speed, "Speed");
    agentOptions->addItem(*s_agentSpeed);

    shared_ptr<Setting<float>> s_agentTurnFactor = make_shared<Setting<float>>(
        &Agent::turnFactor, "TurnFactor");
    agentOptions->addItem(*s_agentTurnFactor);

    shared_ptr<Setting<float>> s_agentRandomness = make_shared<Setting<float>>(
        &Agent::randFactor, "Randomness");
    agentOptions->addItem(*s_agentRandomness);

    shared_ptr<Setting<float>> s_agentBias = make_shared<Setting<float>>(
        &Agent::biasFactor, "Bias");
    agentOptions->addItem(*s_agentBias);

    shared_ptr<Setting<float>> s_agentRepulsion = make_shared<Setting<float>>(
        &Agent::repulsion, "Repulsion");
    agentOptions->addItem(*s_agentRepulsion);


    //search settings
    shared_ptr<DropDown> searchOptions = make_shared<DropDown>("Search Pattern");
    settingsTab->addItem(*searchOptions);

    searchOptions->addItem(searchPreviewGUIImage);

    shared_ptr<Setting<float>> s_searchSize = make_shared<Setting<float>>(
        &Agent::searchSize, "Distance", [&] {
            if (useSimple)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
        });
    searchOptions->addItem(*s_searchSize);

    shared_ptr<Setting<float>> s_searchAngle = make_shared<Setting<float>>(
        &Agent::searchAngle, "Angle", Agent::searchAngle * 180.0f / _Pi, [&] {
            if (useSimple)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
            updateSearchPreview();
        });
    s_searchAngle->setInputFunction([](float f) { return _Pi * f / 180.0f; });
    searchOptions->addItem(*s_searchAngle);

    shared_ptr<Setting<float>> s_searchAngleOffset = make_shared<Setting<float>>(
        &Agent::searchAngleOffset, "Angle Offset", [&] {
            if (useSimple)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
            updateSearchPreview();
        });
    s_searchAngleOffset->setInputFunction([](float f) { return _Pi * f / 180.0f; });
    searchOptions->addItem(*s_searchAngleOffset);

    shared_ptr<Setting<bool>> s_simpleSearch = make_shared<Setting<bool>>(
        &useSimple, "Simple Search", [&] {
            if (useSimple)
                Agent::generateNormTriangleSimple();
            else
                Agent::generateNormTriangle(maxPts);
        });
    searchOptions->addItem(*s_simpleSearch);

    shared_ptr<Setting<bool>> s_useCuda = make_shared<Setting<bool>>(
        &useCuda, "GPU Acceleration");
    searchOptions->addItem(*s_useCuda);


    //shader settings
    shared_ptr<DropDown> shaderOptions = make_shared<DropDown>("Shader");
    settingsTab->addItem(*shaderOptions);

    shared_ptr<Setting<float>> s_dimRate = make_shared<Setting<float>>(
        &dimRate, "Dim Rate", dimRate * 100.0f, [&] {
            shader.setUniform("dimRate", dimRate);
        });
    s_dimRate->setInputFunction([](float f) { return f / 100.0f; });
    shaderOptions->addItem(*s_dimRate);

    shared_ptr<Setting<float>> s_disperseFactor = make_shared<Setting<float>>(
        &disperseFactor, "Fuzzying", [&] {
            shader.setUniform("disperseFactor", disperseFactor);
        });
    shaderOptions->addItem(*s_disperseFactor);


    //Color Settings
    shared_ptr<DropDown> colorOptions = make_shared<DropDown>("Color Palette");
    settingsTab->addItem(*colorOptions);

    Button::Ptr resetColorButton = Button::Create("Reset Color");
    resetColorButton->GetSignal(Button::OnLeftClick).Connect([&]() {
		for_each(execution::par_unseq, agentList.begin(), agentList.end(),
			[&](auto&& a) {
				a.randomizeColorBase();
			});
	});
    colorOptions->addItem(resetColorButton);

    shared_ptr<Setting<float>> s_paletteR = make_shared<Setting<float>>(
        &Agent::palette.x, "Red", [&] {
            for_each(execution::par_unseq, agentList.begin(), agentList.end(),
            [&](auto&& a) {
                    a.updateColor();
                });
        });
    colorOptions->addItem(*s_paletteR);

    shared_ptr<Setting<float>> s_paletteG = make_shared<Setting<float>>(
        &Agent::palette.y, "Green", [&] {
            for_each(execution::par_unseq, agentList.begin(), agentList.end(),
            [&](auto&& a) {
                    a.updateColor();
                });
        });
    colorOptions->addItem(*s_paletteG);

    shared_ptr<Setting<float>> s_paletteB = make_shared<Setting<float>>(
        &Agent::palette.z, "Blue", [&] {
            for_each(execution::par_unseq, agentList.begin(), agentList.end(),
            [&](auto&& a) {
                    a.updateColor();
                });
        });
    colorOptions->addItem(*s_paletteB);

    int imageIndex = 0;
    shared_ptr<Setting<int>> s_imageIndex = make_shared<Setting<int>>(
        &imageIndex, "Image", imagePaths
    );
    colorOptions->addItem(*s_imageIndex);

    shared_ptr<Setting<bool>> s_matchImage = make_shared<Setting<bool>>(
        &imageMatch, "Image Match", [&] {
            if(imagePaths.size() == 0)
				imageMatch = false;
            else {
                matchImage = processImageFromFile(imagePaths[imageIndex]);
            }
        }
    );
    colorOptions->addItem(*s_matchImage);

    float imageMatchRate = 1.0f;

    shared_ptr<Setting<float>> s_imageMatchRate = make_shared<Setting<float>>(
        &imageMatchRate, "Image Match Factor");
    colorOptions->addItem(*s_imageMatchRate);

    //Audio Settings
    shared_ptr<DropDown> audioOptions = make_shared<DropDown>("Audio");
    settingsTab->addItem(*audioOptions);

    shared_ptr<Setting<int>> s_inputDevice = make_shared<Setting<int>>(
        &deviceIndex, "Input Device", deviceNames, [&] {

            deviceId = inputDevices[deviceIndex];

            if (adc.isStreamRunning())
                adc.stopStream();

            if (adc.isStreamOpen())
                adc.closeStream();

            parameters.deviceId = inputDevices[deviceIndex];

            if (Agent::audioAlternate > 0) {
                adc.openStream(nullptr, &parameters, RTAUDIO_FLOAT32, AUDIO_FREQUENCY, &bufferFrames, &recordCallback, (void*)&AUDIO_BUFFER);
                adc.startStream();
            }
        }
    );
    audioOptions->addItem(*s_inputDevice);

    shared_ptr<Setting<bool>> s_enableAudioInput = make_shared<Setting<bool>>(
        &Agent::audioAlternate, "Enable", [&] {
        if (Agent::audioAlternate > 0) {

            if (adc.isStreamRunning())
                adc.stopStream();

            if (adc.isStreamOpen())
                adc.closeStream();

            parameters.deviceId = inputDevices[deviceIndex];

            adc.openStream(nullptr, &parameters, RTAUDIO_FLOAT32, AUDIO_FREQUENCY, &bufferFrames, &recordCallback, (void*)&AUDIO_BUFFER);
            adc.startStream();

        }
        else {
            if (adc.isStreamRunning())
                adc.stopStream();
            if (adc.isStreamOpen())
                adc.closeStream();
        }
    });
    audioOptions->addItem(*s_enableAudioInput);




    int fps = 0;
    Text fpsCounter(std::to_string(fps), *font, 20);
    fpsCounter.setFillColor(Color::White);
    fpsCounter.setPosition(Vector2f(WINDOW_WIDTH - 100, 0));

    
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

        window.resetGLStates();

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
        bool skipSFGUI = false;
        while (window.pollEvent(event)) {

            if (event.type == Event::Closed) {
                window.close();
            }

            if (event.type == Event::KeyPressed) {

                if (event.key.code == Keyboard::Enter) {
                    KeyManager::enter = true;
                }

                //hide gui
                if (event.key.code == Keyboard::H) {
                    hideGUI = !hideGUI;
                    skipSFGUI = true;
                }
                    
                //reset view
                if (event.key.code == Keyboard::R) {
                    view.setCenter(Vector2f(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2));
                    view.setSize(Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
                    accZoom = 1;
                    worldTex.setView(view);
                }

            }

            if (event.type == Event::Resized) {
                Vector2f scale(
                    (float) WINDOW_WIDTH / event.size.width,
                    (float)WINDOW_HEIGHT / event.size.height
                );
                guiSprite.setScale(scale);
            }

            //view zooming
            if (event.type == Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta > 0) {
                    view.zoom(0.9f);
                    accZoom *= 0.9f;
                }
                else {
                    view.zoom(1.1f);
                    accZoom *= 1.1f;
                }
                worldTex.setView(view);
            }

            //pass event to SFGUI
            if(!hideGUI && !skipSFGUI) desktop.HandleEvent(event);
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
                Agent::audioMod = processAudio( frequencies, Vector3f(rFac, gFac, bFac) );
            }

            //GPU accelerated agent searches
            if (useCuda) {

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
                run(32, settingData, dev_inputData, cudaResource, dev_ptData, dev_outputData);
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

                    if (imageMatch) agentList[i].updateColorBase(
                        matchImage.getPixel(agentList[i].getPos().x, agentList[i].getPos().y),
                        imageMatchRate
                    );

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

                    if(imageMatch) agentList[i].updateColorBase(
                        matchImage.getPixel(agentList[i].getPos().x, agentList[i].getPos().y),
                        imageMatchRate
                    );

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

                desktop.Update(clock.getElapsedTime().asSeconds());

                guiTexture.clear(Color::Transparent);
                guiTexture.draw(gui.getBackground());

                sfgui.Display(guiTexture);
                guiTexture.display();

                window.draw(guiSprite);
                window.draw(fpsCounter);
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

        KeyManager::reset();
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