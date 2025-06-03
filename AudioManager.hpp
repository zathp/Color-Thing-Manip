#pragma once

#include <array>
#include <vector>

#include <SFML/Graphics.hpp>

#include "RtAudio.h"
#include "fftw3.h"

#include "GUI-Textures.hpp"
#include "Parameters.hpp"

//#include <SFGUI/Widgets.hpp>
//#include <SFGUI/Image.hpp>

#define AUDIO_FREQUENCY 44100.0f
#define AUDIO_BUFFER_FRAMES 512
#define AUDIO_SPECTRUM 257

struct AudioData {
    float inputBuffer[AUDIO_BUFFER_FRAMES];
    fftwf_complex spectrum[AUDIO_SPECTRUM];
};

class AudioManager {
    RtAudio adc;
    RtAudio::StreamParameters parameters;
    unsigned int bufferFrames = AUDIO_BUFFER_FRAMES;

    std::vector<unsigned int> inputDevices;
    std::vector<std::string> deviceNames;

    int deviceIndex = 0;
    bool audioEnabled = false;
    
    std::array<std::array<float, 2>, 3> frequencies;

    AudioData AUDIO_BUFFER;

	std::unique_ptr<AudioVisualizer> audioVisualizer;
	

    static std::shared_ptr<GLOBAL_SETTINGS> GlobalSettings;

public:

    AudioManager();

    // GUI-facing control
    const std::vector<std::string>& getDeviceNames() const;
    int getDeviceIndex() const;
    void setDeviceIndex(int index, bool restartIfActive);
    void enableAudio(bool on);
    bool isAudioEnabled() const;

    // Audio processing
    void processAudio();
    void setFrequencies(const std::array<std::array<float, 2>, 3>& freqs);
    AudioData& getBuffer();
    void drawVisualizerImage();

    // Callback for RtAudio
    static int recordCallback(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
        double streamTime, RtAudioStreamStatus status, void* userData);

    //sfg::Image::Ptr getVisualizerImage();

	static void setSettings(std::shared_ptr<GLOBAL_SETTINGS> settings) {
		GlobalSettings = settings;
	}

	sf::Vector3f getLastEnergies() {
		return sf::Vector3f(
			rollingMaxEnergy[0],
			rollingMaxEnergy[1],
			rollingMaxEnergy[2]
		);
	}

    const std::vector<std::string> featureNames = {
        "Rolling Max Energy",
        "Last Energies",
        "Peak Drift",
        "Centroid Drift"
    };

private:
    bool hasInputDevices() const;
    float getMagnitude(int i);
	float rollingMaxEnergy[3] = { 0.001f, 0.001f, 0.001f };

    float peakPos[3] = { 0.f, 0.f, 0.f };              // current peak pos (0–1)
    float rollingPeakPos[3] = { 0.f, 0.f, 0.f };       // smoothed peak pos
    float centroidPos[3] = { 0.f, 0.f, 0.f };          // current center-of-mass pos
    float rollingCentroid[3] = { 0.f, 0.f, 0.f };      // smoothed center-of-mass pos

	float lastEnergies[3] = { 0, 0, 0 };              //last cumulative energies
	float peakDrift[3] = { 0.f, 0.f, 0.f };           // peak drift
	float centroidDrift[3] = { 0.f, 0.f, 0.f };       // centroid drift

    const std::vector<float*> featureChannels = {
        rollingMaxEnergy,
        lastEnergies,
        peakDrift,
		centroidDrift
    };
};
