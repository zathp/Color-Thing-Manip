#pragma once

#include "Config.hpp"

#include <vector>
#include <array>
#include <functional>
#include <string>
#include <memory>

#include <SFML/Graphics.hpp>
#include <SFGUI/SFGUI.hpp>
#include <SFGUI/Widgets.hpp>
#include <SFGUI/Button.hpp>
#include <SFGUI/Label.hpp>
#include <SFGUI/Entry.hpp>
#include <SFGUI/Box.hpp>
#include <SFGUI/ScrolledWindow.hpp>
#include <SFGUI/Scale.hpp>
#include <SFGUI/ComboBox.hpp>

#include "GUIConfig.hpp"
#include "Modifier.hpp"

template<typename T> class SettingBase {

protected:

	T val;

	std::string name;

	sfg::Box::Ptr box_main;
	sfg::Box::Ptr box_container;
	sfg::Label::Ptr label;

	std::function<void()> onChange = []() {};

	std::function<T(T)> inputFunc = [](T value) { return value; };

	static std::string fString(const std::string& s) {
		size_t pos = s.find('.');
		if (pos == std::string::npos || pos + 4 > s.length()) return s;
		return s.substr(0, pos + 4);
	}

	static sfg::Label::Ptr horizontalSpacer(float width) {
		sfg::Label::Ptr box = sfg::Label::Create(" ");
		box->SetRequisition({ width, 0.f });
		return box;
	}

	static sfg::Label::Ptr verticalSpacer(float height) {
		sfg::Label::Ptr box = sfg::Label::Create(" ");
		box->SetRequisition({ 0.f, height });
		return box;
	}

	bool tryParseEntry(sfg::Entry::Ptr entry, T& target) {
		if (entry->GetText().isEmpty()
			|| entry->GetText() == "")
			return false;
		std::string text = entry->GetText().toAnsiString();
		try {
			size_t idx;
			T v = parse(text, &idx);
			if (idx == text.size()) {
				entry->SetClass("valid");
				target = v;
				return true;
			}
			else {
				entry->SetClass("invalid");
				return false;
			}
		}
		catch (const std::exception&) {
			return false;
		}
	}

	virtual T parse(const std::string& s, size_t* idx) {
		throw std::invalid_argument("invalid setting type");
	}

public:

	SettingBase(const std::string& name,
		std::function<void()> onChange = []() {}) : name(name), onChange(onChange) {

		box_main = sfg::Box::Create(sfg::Box::Orientation::VERTICAL);

		label = sfg::Label::Create(name);
		label->SetRequisition(guiOptions.labelSize);
		label->SetClass("setting");

		box_container = sfg::Box::Create(sfg::Box::Orientation::HORIZONTAL);

		box_main->Pack(label, false, false);
		box_main->Pack(box_container, true, false);
	}

	SettingBase(T initialVal, const std::string& name) : SettingBase(name, []() {}) {
		val = initialVal;
	}

	T get() const {
		return val;
	}

	void set(T newVal) {
		val = newVal;
		onChange();
	}

	void setInputFunction(std::function<T(T)> func) {
		inputFunc = func;
		val = inputFunc(val);
	}

	void setOnChange(std::function<void()> func) {
		onChange = func;
	}

	std::string getString() {
		try {
			return std::to_string(val);
		}
		catch (const std::exception&) {
			return "*";
		}
	}

	operator T() const {
		return val;
	}

	operator sfg::Widget::Ptr() {
		return box_main;
	}

	sfg::Box::Ptr getBase() {
		return box_main;
	}
};

template <typename T> class Setting : public SettingBase<T> {

};

template <> class Setting<int> : public SettingBase<int> {

public:

	Setting(int val, const std::string& name, const std::vector<std::string>& map,
		std::function<void()> onChange = []() {});

	Setting(int val, const std::string& name, std::function<void()> onChange = []() {});

	void setEntryText(int i) {
		if (input) {
			input->SetText(fString(std::to_string(i)));
		}
	}

	int parse(const std::string& s, size_t* idx) override;

	void updateComboBoxItems(const std::vector<std::string>& map);

private:

	sfg::Entry::Ptr input;
	sfg::ComboBox::Ptr combo;
};



template <> class Setting<float> : public SettingBase<float> {

public:

	Setting(float val, std::string name, std::function<void()> onChange = [] {});

	operator float() const {
		return val;
	}

	void setEntryText(float f) {
		if (input) {
			input->SetText(fString(std::to_string(f)));
		}
	}

	float parse(const std::string& s, size_t* idx) override;

	void setScaleSettings(float min, float max, float step);

	void tryUpdateValueFromEntry();

	void updateValueFromScale();

	void toggleModifierBox();

	static sfg::Widget::Ptr createModifierEntry(std::shared_ptr<Modifier> mod);

private:

	sfg::Entry::Ptr input;

	sfg::Scale::Ptr scale;

	sfg::Button::Ptr modButton, addModButton;

	sfg::Box::Ptr modBox, modListContainer;

	sfg::Frame::Ptr modListFrame;

	float modifiedVal = 0.0f;

	bool dragging = false;

	sf::Clock sliderUpdateClock;

	std::vector<std::shared_ptr<Modifier>> localMods;

	static constexpr float sliderUpdateInterval = 50.0f; // milliseconds
};

template <> class Setting<bool> : public SettingBase<bool> {

public:

	Setting(bool val, std::string name, std::function<void()> onChange);

	Setting(bool val, std::string name) : Setting<bool>(val, name, []() {}) {}

private:

	sfg::CheckButton::Ptr input;
};

struct SEARCH_SETTINGS {
	Setting<float> searchSize = Setting<float>(10.0f, "Search Size");
	Setting<float> searchAngle = Setting<float>(30.0f, "Search Angle");
	Setting<float> searchAngleOffset = Setting<float>(0.f, "Angle Offset");
	Setting<int>   searchMethodIndex = Setting<int>(0, "Search Method", std::vector<std::string>{});
	Setting<bool>  useCuda = Setting<bool>(false, "Use CUDA");
};

struct AGENT_COLOR_SETTINGS {
	Setting<float> filterR = Setting<float>(1.0f, "Filter R");
	Setting<float> filterG = Setting<float>(1.0f, "Filter G");
	Setting<float> filterB = Setting<float>(1.0f, "Filter B");

	Setting<bool> useColorMap = Setting<bool>(false, "Use Color Map");
	Setting<bool> visibleToAgents = Setting<bool>(false, "Visible to Agents");
	Setting<int>  colorMapIndex = Setting<int>(0, "Color Map", std::vector<std::string>{});

	// Hue rotation
	Setting<bool>  hueRotationEnabled = Setting<bool>(false, "Hue Rotation Enabled");
	Setting<float> hueRotationAngle = Setting<float>(0.0f, "Hue Rotation Angle");
	Setting<bool>  hueOscillation = Setting<bool>(false, "Hue Oscillation");
	Setting<float> hueOscillationPeriod = Setting<float>(50.0f, "Hue Oscillation Period");
};

struct AGENT_SETTINGS {
	Setting<int>   agentCount = Setting<int>(10000, "Agent Count");
	Setting<float> speed = Setting<float>(1.0f, "Speed");
	Setting<float> timeScale = Setting<float>(1.0f, "Time Scale");
	Setting<float> turnFactor = Setting<float>(1.0f, "Turn Factor");
	Setting<float> randFactor = Setting<float>(1.0f, "Random Factor");
	Setting<float> biasFactor = Setting<float>(1.0f, "Bias Factor");
	Setting<float> repulsion = Setting<float>(0.0f, "Repulsion");
	Setting<bool>  lockDir = Setting<bool>(false, "Lock Direction");
	Setting<int>   lockDirCount = Setting<int>(8, "Lock Direction Count");

	Setting<float> memoryFactor = Setting<float>(0.0f, "Memory Factor");
	Setting<bool> interpolateColorDrop = Setting<bool>(true, "Interpolate Color Drop");

	SEARCH_SETTINGS Search;
	AGENT_COLOR_SETTINGS Color;
};

struct OSCILLATION_SETTINGS {
	Setting<bool>  timeOscillationEnabled = Setting<bool>(false, "Time Oscillation Enabled");
	Setting<float> globalPeriodR = Setting<float>(50.0f, "Global Period R");
	Setting<float> globalPeriodG = Setting<float>(70.0f, "Global Period G");
	Setting<float> globalPeriodB = Setting<float>(30.0f, "Global Period B");

	Setting<bool>  distAlternate = Setting<bool>(false, "Distance Alternate");
	Setting<float> distR = Setting<float>(500.0f, "Distance R");
	Setting<float> distG = Setting<float>(1000.0f, "Distance G");
	Setting<float> distB = Setting<float>(2000.0f, "Distance B");
	Setting<float> distPeriodR = Setting<float>(317.0f, "Distance Period R");
	Setting<float> distPeriodG = Setting<float>(731.0f, "Distance Period G");
	Setting<float> distPeriodB = Setting<float>(571.0f, "Distance Period B");
};

struct AUDIO_SETTINGS {
	Setting<int>   deviceIndex = Setting<int>(0, "Device", std::vector<std::string> {});
	Setting<bool>  audioAlternate = Setting<bool>(false, "Enable");
	Setting<bool>  autoGainEqualizer = Setting<bool>(false, "Auto Gain Equalizer");

	Setting<float> gainR = Setting<float>(1.0f, "Gain R");
	Setting<float> gainG = Setting<float>(1.0f, "Gain G");
	Setting<float> gainB = Setting<float>(1.0f, "Gain B");

	Setting<float> modR = Setting<float>(1.0f, "Mod R");
	Setting<float> modG = Setting<float>(1.0f, "Mod G");
	Setting<float> modB = Setting<float>(1.0f, "Mod B");

	// Spectral drift modifiers
	Setting<bool>  peakDriftEnabled = Setting<bool>(false, "Peak Drift Enabled");
	Setting<float> peakDriftStrength = Setting<float>(1.0f, "Peak Drift Strength");

	Setting<bool>  centroidDriftEnabled = Setting<bool>(false, "Centroid Drift Enabled");
	Setting<float> centroidDriftStrength = Setting<float>(1.0f, "Centroid Drift Strength");

	Setting<bool>  suppressLowPeakDrift = Setting<bool>(false, "Suppress Low Peak Drift");
	Setting<float> suppressPeakStrength = Setting<float>(1.0f, "Suppress Peak Strength");

	Setting<bool>  suppressLowCentroidDrift = Setting<bool>(false, "Suppress Low Centroid Drift");
	Setting<float> suppressCentroidStrength = Setting<float>(1.0f, "Suppress Centroid Strength");
};

struct SHADER_SETTINGS {
	Setting<float> dimRate = Setting<float>(0.005f, "Dim Rate");
	Setting<float> disperseFactor = Setting<float>(0.3f, "Disperse Factor");
};

struct SETTINGS {

	Setting<int> targetFPS = Setting<int>(144, "Target FPS");
	Setting<int> targetUPS = Setting<int>(999, "Target UPS");

	AGENT_SETTINGS Agents;
	OSCILLATION_SETTINGS Oscillation;
	AUDIO_SETTINGS Audio;
	SHADER_SETTINGS Shader;
	std::unordered_map<std::string, std::shared_ptr<Modifier>> modifierPool;
};

struct GLOBAL_SETTINGS {

	Setting<int> targetFPS = Setting<int>(144, "Target FPS");
	Setting<int> targetUPS = Setting<int>(999, "Target UPS");
	SHADER_SETTINGS Shader;
	AUDIO_SETTINGS Audio;
};

struct GROUP_SETTINGS {

	AGENT_SETTINGS Agents;
	OSCILLATION_SETTINGS Oscillation;
};