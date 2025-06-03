#pragma once

#include "Config.hpp"

#include <memory>
#include <vector>
#include <string>
#include <functional>

#include <imgui.h>

#include <SFML/Graphics.hpp>

class Setting {
protected:
    std::string name = "Setting";
	std::function<void()> onChange = []() {};

public:
    void setOnChange(std::function<void()> func);
    virtual void draw() = 0;
    virtual ~Setting() = default;
};

class FloatSetting : public Setting {
    float valueDisplay = 0;
    float valueBase = 0;
    float valueModified = 0;
    float min = 0.0f;
    float max = 1.0f;
	std::function<float(float)> inputFunc = [](float f) { return f; };

public:
    operator float() const;
    FloatSetting(std::string name, float defaultVal);
    void setInputFunc(std::function<float(float)> func);
    void setValue(float newValue, bool applyInputFunc = false);
	void setScaleSettings(float min, float max);
    void draw() override;
};

class IntSetting : public Setting {
    int value = 0;

public:
    operator int() const;
    IntSetting(std::string name, int defaultVal);
    void setValue(int newValue);
    void draw() override;
};

class ChoiceSetting : public Setting {
    std::vector<std::string> choices;
    int selectedIndex = 0;

public:
    ChoiceSetting(std::string name, std::vector<std::string> choices);
    operator std::string() const;
    operator int() const;
	void updateChoiceOptions(std::vector<std::string> newChoices);
    void setSelectedIndex(int index);
    int getSelectedIndex() const;
    void draw() override;
};

class CheckboxSetting : public Setting {
    bool value = false;

public:
    CheckboxSetting(std::string name, bool defaultVal = false);
    operator bool() const;
    void setValue(bool newValue);
    void draw() override;
};

struct SEARCH_SETTINGS {
	FloatSetting searchSize = FloatSetting("Search Size", 10.0f);
	FloatSetting searchAngle = FloatSetting("Search Angle", 30.0f);
	FloatSetting searchAngleOffset = FloatSetting("Search Angle Offset", 0.0f);
	ChoiceSetting searchMethodIndex = ChoiceSetting("Search Method", { "Triangle", "Point", "Line" });
	CheckboxSetting useCuda = CheckboxSetting("Use CUDA", false);
};

struct AGENT_COLOR_SETTINGS {
	FloatSetting filterR = FloatSetting("Filter R", 1.0f);
	FloatSetting filterG = FloatSetting("Filter G", 1.0f);
	FloatSetting filterB = FloatSetting("Filter B", 1.0f);

	CheckboxSetting hueRotationEnabled = CheckboxSetting("Hue Rotation", false);
	FloatSetting hueRotationAngle = FloatSetting("Hue Angle", 0.0f);
	CheckboxSetting hueOscillation = CheckboxSetting("Hue Oscillation", false);
	FloatSetting hueOscillationPeriod = FloatSetting("Hue Oscillation Period", 50.0f);
};

struct AGENT_SETTINGS {
	IntSetting agentCount = IntSetting("Agent Count", 10000);
	FloatSetting speed = FloatSetting("Speed", 1.0f);
	FloatSetting turnFactor = FloatSetting("Turn Factor", 1.0f);
	FloatSetting randFactor = FloatSetting("Random Factor", 1.0f);
	FloatSetting biasFactor = FloatSetting("Bias Factor", 1.5f);
	FloatSetting repulsion = FloatSetting("Repulsion", 0.0f);
	CheckboxSetting lockDir = CheckboxSetting("Lock Direction", false);
	IntSetting lockDirCount = IntSetting("Lock Dir Count", 8);
	FloatSetting memoryFactor = FloatSetting("Memory Factor", 0.0f);
	CheckboxSetting interpolateColorDrop = CheckboxSetting("Interpolate Color Drop", true);

	SEARCH_SETTINGS Search;
	AGENT_COLOR_SETTINGS Color;
};

struct OSCILLATION_SETTINGS {
	CheckboxSetting timeOscillationEnabled = CheckboxSetting("Time Oscillation", false);
	FloatSetting globalPeriodR = FloatSetting("Period R", 50.0f);
	FloatSetting globalPeriodG = FloatSetting("Period G", 70.0f);
	FloatSetting globalPeriodB = FloatSetting("Period B", 30.0f);

	CheckboxSetting distAlternate = CheckboxSetting("Dist Alternate", false);
	FloatSetting distR = FloatSetting("Dist R", 500.0f);
	FloatSetting distG = FloatSetting("Dist G", 1000.0f);
	FloatSetting distB = FloatSetting("Dist B", 2000.0f);

	FloatSetting distPeriodR = FloatSetting("Dist Period R", 317.0f);
	FloatSetting distPeriodG = FloatSetting("Dist Period G", 731.0f);
	FloatSetting distPeriodB = FloatSetting("Dist Period B", 571.0f);
};

struct AUDIO_SETTINGS {
	ChoiceSetting deviceIndex = ChoiceSetting("Audio Device", { "Default", "Mic", "Virtual" });
	CheckboxSetting audioAlternate = CheckboxSetting("Audio Alternate", false);
	CheckboxSetting autoGainEqualizer = CheckboxSetting("Auto Gain", false);

	FloatSetting gainR = FloatSetting("Gain R", 1.0f);
	FloatSetting gainG = FloatSetting("Gain G", 1.0f);
	FloatSetting gainB = FloatSetting("Gain B", 1.0f);

	// non-GUI fields
	float modR = 1.0f;
	float modG = 1.0f;
	float modB = 1.0f;

	CheckboxSetting peakDriftEnabled = CheckboxSetting("Peak Drift", false);
	FloatSetting peakDriftStrength = FloatSetting("Peak Drift Strength", 1.0f);

	CheckboxSetting centroidDriftEnabled = CheckboxSetting("Centroid Drift", false);
	FloatSetting centroidDriftStrength = FloatSetting("Centroid Drift Strength", 1.0f);

	CheckboxSetting suppressLowPeakDrift = CheckboxSetting("Suppress Low Peak", false);
	FloatSetting suppressPeakStrength = FloatSetting("Suppress Peak Strength", 1.0f);

	CheckboxSetting suppressLowCentroidDrift = CheckboxSetting("Suppress Low Centroid", false);
	FloatSetting suppressCentroidStrength = FloatSetting("Suppress Centroid Strength", 1.0f);
};

struct SHADER_SETTINGS {
	FloatSetting dimRate = FloatSetting("Dim Rate", 0.005f);
	FloatSetting disperseFactor = FloatSetting("Disperse", 0.3f);

	CheckboxSetting useColorMap = CheckboxSetting("Use Color Map", false);
	ChoiceSetting colorMapIndex = ChoiceSetting("Color Map", { "Classic", "Vibrant", "Dark" });
};

struct SETTINGS {
	IntSetting targetFPS = IntSetting("Target FPS", 144);
	IntSetting targetUPS = IntSetting("Target UPS", 999);

	AGENT_SETTINGS Agents;
	OSCILLATION_SETTINGS Oscillation;
	AUDIO_SETTINGS Audio;
	SHADER_SETTINGS Shader;
};

struct GLOBAL_SETTINGS {
	IntSetting targetFPS = IntSetting("Target FPS", 144);
	IntSetting targetUPS = IntSetting("Target UPS", 999);
	SHADER_SETTINGS Shader;
	AUDIO_SETTINGS Audio;
};

struct GROUP_SETTINGS {
	AGENT_SETTINGS Agents;
	OSCILLATION_SETTINGS Oscillation;
};