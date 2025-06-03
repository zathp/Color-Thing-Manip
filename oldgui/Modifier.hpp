#pragma once

#include <vector>
#include <string>

class Modifier {
public:
	enum class SourceType { AudioEnergy, AudioCentroidDrift, AudioPeakDrift, Oscillator, Constant };
	enum class OperationType { Add, Multiply, Divide, Subtract };

	std::string name = "New Mod";
	SourceType type = Modifier::SourceType::Constant;
	OperationType operation = Modifier::OperationType::Add;
	int audioChannel = 0; //used for audio features
	float constant = 1.0f; // For constant (used as a multipler to the audio features)
	float speed = 1.0f; // For oscillator

	Modifier() {

	}

	float evaluate(float input, const std::vector<float*>& featureMap, float time);

	static int featureMapIndex(Modifier::SourceType type) {
		switch (type) {
		case Modifier::SourceType::AudioEnergy:         return 0;
		case Modifier::SourceType::AudioCentroidDrift:  return 1;
		case Modifier::SourceType::AudioPeakDrift:      return 2;
		default:                                        return -1; // not a featureMap type
		}
	}

	std::string generateName() const;
};