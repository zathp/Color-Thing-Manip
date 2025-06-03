#include "AudioManager.hpp"

using namespace std;

shared_ptr<GLOBAL_SETTINGS> AudioManager::GlobalSettings = nullptr;

AudioManager::AudioManager() {

    frequencies = {
        std::array<float, 2>{0, 250},
        std::array<float, 2>{250, 2000},
        std::array<float, 2>{2000, 44100}
    };

    auto devices = adc.getDeviceIds();
    for (int id : devices) {
        RtAudio::DeviceInfo info = adc.getDeviceInfo(id);
        if (info.inputChannels > 0) {
            inputDevices.push_back(id);
            deviceNames.push_back(info.name);
        }
    }

    parameters.nChannels = 1;
    parameters.firstChannel = 0;

    if (inputDevices.empty()) {
        cout << "No audio input devices detected\n";
    }
    else {
        cout << inputDevices.size() << " input devices detected\n";
    }

	// Initialize the audio buffer
	AUDIO_BUFFER = AudioData();

	audioVisualizer = std::make_unique<AudioVisualizer>();
}

bool AudioManager::hasInputDevices() const {
    return !inputDevices.empty();
}

const std::vector<std::string>& AudioManager::getDeviceNames() const {
    return deviceNames;
}

int AudioManager::getDeviceIndex() const {
    return deviceIndex;
}

void AudioManager::setDeviceIndex(int index, bool restartIfActive) {
    cout << "e\n";
    if (index < 0 || index >= inputDevices.size()) return;
    deviceIndex = index;

    if (restartIfActive && audioEnabled) {
        enableAudio(false);
        enableAudio(true);
    }

    std::cout << "Selected index: " << index << " -> device ID: " << inputDevices[index]
        << " (" << deviceNames[index] << ")" << std::endl;
}

void AudioManager::enableAudio(bool on) {
    if (on == audioEnabled) return;

    if (on) {
        if (!hasInputDevices()) return;
        parameters.deviceId = inputDevices[deviceIndex];

        if (adc.isStreamRunning()) adc.stopStream();
        if (adc.isStreamOpen()) adc.closeStream();

		auto info = adc.getDeviceInfo(parameters.deviceId);
        std::cout << "Opening device: " << parameters.deviceId << " - " << info.name << std::endl;

        adc.openStream(nullptr, &parameters, RTAUDIO_FLOAT32, AUDIO_FREQUENCY,
            &bufferFrames, &AudioManager::recordCallback, (void*)&AUDIO_BUFFER);
        adc.startStream();
    }
    else {
        if (adc.isStreamRunning()) adc.stopStream();
        if (adc.isStreamOpen()) adc.closeStream();
    }

    audioEnabled = on;
}

bool AudioManager::isAudioEnabled() const {
    return audioEnabled;
}

AudioData& AudioManager::getBuffer() {
    return AUDIO_BUFFER;
}

int AudioManager::recordCallback(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
    double streamTime, RtAudioStreamStatus status, void* userData) {
    AudioData* data = reinterpret_cast<AudioData*>(userData);
    memcpy(data->inputBuffer, inputBuffer, sizeof(float) * nBufferFrames);

    auto plan = fftwf_plan_dft_r2c_1d(nBufferFrames, data->inputBuffer, data->spectrum, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    return 0;
}

float AudioManager::getMagnitude(int i) {
    float real = AUDIO_BUFFER.spectrum[i][0];
    float imag = AUDIO_BUFFER.spectrum[i][1];
    return sqrt(real * real + imag * imag);
}

void AudioManager::processAudio() {
    const float maxFreq = AUDIO_FREQUENCY / 2;
    const float binWidth = maxFreq / AUDIO_SPECTRUM;

    float energies[3] = { 0.f, 0.f, 0.f };

    const float alpha = 0.2f;  // smoothing for energies and drifts

    for (int i = 0; i < 3; i++) {
        int start = frequencies[i][0] / binWidth;
        int end = frequencies[i][1] / binWidth;

        start = std::max(0, start);
        end = std::min(end, AUDIO_SPECTRUM - 1);
        int length = std::max(end - start, 1);

        // For centroid + peak
        float sumMag = 0.f;
        float weightedSum = 0.f;
        float maxMag = 0.f;
        int peakBinIndex = start;

        for (int bin = start; bin < end; ++bin) {
            float mag = getMagnitude(bin);
            energies[i] += mag;
            sumMag += mag;
            weightedSum += (bin - start) * mag;

            if (mag > maxMag) {
                maxMag = mag;
                peakBinIndex = bin;
            }
        }

        // Normalize to [0,1] within bin
        peakPos[i] = float(peakBinIndex - start) / float(length);
        centroidPos[i] = (sumMag > 0.f) ? weightedSum / sumMag / float(length) : 0.f;

        // Smooth both
        rollingPeakPos[i] = (1.f - alpha) * rollingPeakPos[i] + alpha * peakPos[i];
        rollingCentroid[i] = (1.f - alpha) * rollingCentroid[i] + alpha * centroidPos[i];
    }



    for (int i = 0; i < 3; ++i) {
        peakDrift[i] = std::abs(peakPos[i] - rollingPeakPos[i]);
        centroidDrift[i] = std::abs(centroidPos[i] - rollingCentroid[i]);

		// Drift-based excitation
        if (GlobalSettings->Audio.peakDriftEnabled) {
            energies[i] *= 1.0f + peakDrift[i] * GlobalSettings->Audio.peakDriftStrength;
        }
        if (GlobalSettings->Audio.centroidDriftEnabled) {
            energies[i] *= 1.0f + centroidDrift[i] * GlobalSettings->Audio.centroidDriftStrength;
        }

        // Drift-based suppression
        if (GlobalSettings->Audio.suppressLowPeakDrift) {
            float suppressFactor = 1.0f - (1.0f - peakDrift[i]) * GlobalSettings->Audio.suppressPeakStrength;
            energies[i] *= suppressFactor;
        }

        if (GlobalSettings->Audio.suppressLowCentroidDrift) {
            float suppressFactor = 1.0f - (1.0f - centroidDrift[i]) * GlobalSettings->Audio.suppressCentroidStrength;
            energies[i] *= suppressFactor;
        }
    }

    // Apply gain
    energies[0] *= GlobalSettings->Audio.gainR;
    energies[1] *= GlobalSettings->Audio.gainG;
    energies[2] *= GlobalSettings->Audio.gainB;

    if (GlobalSettings->Audio.autoGainEqualizer) {
        const float decay = 0.05f;
        for (int i = 0; i < 3; ++i) {
            rollingMaxEnergy[i] = std::max(energies[i], (1.0f - decay) * rollingMaxEnergy[i]);
            float gain = 1.0f / std::max(rollingMaxEnergy[i], 0.01f);
            energies[i] *= gain;
        }
    }
    else {
        float currentMax = std::max({ energies[0], energies[1], energies[2], 0.0001f });
        for (int i = 0; i < 3; ++i)
            energies[i] /= currentMax;
    }

    // Final smoothing
    for (int i = 0; i < 3; ++i) {
        lastEnergies[i] = (1 - alpha) * lastEnergies[i] + alpha * energies[i];
    }

    audioVisualizer->update(lastEnergies, rollingPeakPos, rollingCentroid, peakDrift, centroidDrift);

    GlobalSettings->Audio.modR = lastEnergies[0];
    GlobalSettings->Audio.modG = lastEnergies[1];
    GlobalSettings->Audio.modB = lastEnergies[2];
}


void AudioManager::setFrequencies(const std::array<std::array<float, 2>, 3>& freqs) {
    frequencies = freqs;
}

void AudioManager::drawVisualizerImage() {
    audioVisualizer->drawGuiImage();
}