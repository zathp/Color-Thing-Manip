// GuiSettings.cpp
#include "Parameters.hpp"

using namespace std;
using namespace sf;

// --- Setting Base ---
void Setting::setOnChange(std::function<void()> func) {
    onChange = func;
}

// --- FloatSetting ---
FloatSetting::FloatSetting(std::string name, float defaultVal) {
    this->name = name;
    valueBase = defaultVal;
    valueModified = defaultVal;
    valueDisplay = defaultVal;
}

void FloatSetting::setInputFunc(std::function<float(float)> func) {
    inputFunc = func;
    valueBase = inputFunc(valueBase);
    valueModified = valueBase;
}

void FloatSetting::setValue(float newValue, bool applyInputFunc) {
	if (applyInputFunc) {
		valueBase = inputFunc(newValue);
	}
	else {
		valueBase = newValue;
	}
}

void FloatSetting::setScaleSettings(float min, float max) {
	this->min = min;
	this->max = max;
}

FloatSetting::operator float() const {
    return valueModified;
}

void FloatSetting::draw() {
    ImGui::Text("%s", name.c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    bool changed = ImGui::SliderFloat(("##slider_" + name).c_str(), &valueDisplay, min, max);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    changed |= ImGui::InputFloat(("##input_" + name).c_str(), &valueDisplay);

    if (changed) {
        valueBase = valueDisplay;
        valueModified = inputFunc ? inputFunc(valueBase) : valueBase;
        if (onChange) onChange();
    }

    ImGui::Separator();
}

// --- IntSetting ---
IntSetting::IntSetting(std::string name, int defaultVal) {
    this->name = name;
    value = defaultVal;
}

IntSetting::operator int() const {
    return value;
}

void IntSetting::setValue(int newValue) {
    if (value != newValue) {
        value = newValue;
    }
}

void IntSetting::draw() {
    ImGui::Text("%s", name.c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    bool changed = ImGui::InputInt(("##input_" + name).c_str(), &value);
    if (changed && onChange) onChange();
    ImGui::Separator();
}

// --- ChoiceSetting ---
ChoiceSetting::ChoiceSetting(std::string name, std::vector<std::string> choices) : choices(choices) {
    this->name = name;
}

ChoiceSetting::operator std::string() const {
    return choices[selectedIndex];
}

ChoiceSetting::operator int() const {
    return selectedIndex;
}

void ChoiceSetting::updateChoiceOptions(std::vector<std::string> newChoices) {
	choices = std::move(newChoices);
	if (selectedIndex >= choices.size()) {
		selectedIndex = 0; // Reset to first choice if current index is out of bounds
	}
}

void ChoiceSetting::setSelectedIndex(int index) {
    if (index >= 0 && index < choices.size()) {
        selectedIndex = index;
        if (onChange) onChange();
    }
}

int ChoiceSetting::getSelectedIndex() const {
    return selectedIndex;
}

void ChoiceSetting::draw() {
    ImGui::Text("%s", name.c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);

    if (ImGui::BeginCombo(("##combo_" + name).c_str(), choices[selectedIndex].c_str())) {
        for (int i = 0; i < choices.size(); ++i) {
            bool isSelected = (i == selectedIndex);
            if (ImGui::Selectable(choices[i].c_str(), isSelected)) {
                selectedIndex = i;
                if (onChange) onChange();
            }
            if (isSelected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();
}

// --- CheckboxSetting ---
CheckboxSetting::CheckboxSetting(std::string name, bool defaultVal) {
    this->name = name;
    value = defaultVal;
}

CheckboxSetting::operator bool() const {
    return value;
}

void CheckboxSetting::setValue(bool newValue) {
    if (value != newValue) {
        value = newValue;
    }
}

void CheckboxSetting::draw() {
    bool changed = ImGui::Checkbox(name.c_str(), &value);
    if (changed && onChange) onChange();
}