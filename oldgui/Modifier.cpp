#pragma once

#include "Modifier.hpp"

#include <cmath>
#include <algorithm>

#include <SFGUI/Widget.hpp>
#include <SFGUI/Box.hpp>
#include <SFGUI/Button.hpp>
#include <SFGUI/Label.hpp>

float Modifier::evaluate(float input, const std::vector<float*>& featureMap, float time) {

	float modValue = 1.0f;
	int index = -1;

	float value = input;

	switch (type) {
	case Modifier::SourceType::Constant:
		modValue = constant;
		break;

	case Modifier::SourceType::Oscillator:
		modValue = std::sin(time * speed) * constant;
		break;

	case Modifier::SourceType::AudioEnergy:
	case Modifier::SourceType::AudioCentroidDrift:
	case Modifier::SourceType::AudioPeakDrift:
		index = Modifier::featureMapIndex(type);
		if (index != -1) {
			modValue = featureMap[index][audioChannel] * constant;
		}
		break;
	}

	switch (operation) {
	case Modifier::OperationType::Add:
		value += modValue;
		break;

	case Modifier::OperationType::Multiply:
		value *= modValue;
		break;

	case Modifier::OperationType::Divide:
		if (modValue != 0) {
			value /= modValue;
		}
		break;

	case Modifier::OperationType::Subtract:
		value -= modValue;
		break;
	}

	return value;
}

std::string Modifier::generateName() const {

	static const char* sourceNames[] = {
		"Energy", "Centroid", "Peak", "Oscillator", "Constant"
	};

	static const char* opSymbols[] = {
		"+", "×", "÷", "-"
	};

	static const char* channelNames[] = {
		"Bass", "Mid", "Treble"
	};

	std::string src = sourceNames[static_cast<int>(type)];

	std::string channel = (type == SourceType::AudioEnergy || type == SourceType::AudioCentroidDrift || type == SourceType::AudioPeakDrift)
		? " " + std::string(channelNames[std::clamp(audioChannel, 0, 2)])
		: "";

	std::string op = " " + std::string(opSymbols[static_cast<int>(operation)]);
	std::string value = " " + std::to_string(constant).substr(0, 4);

	return src + channel + op + value;
}