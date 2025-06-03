#pragma once
#include "Config.hpp"

#include <SFML/Graphics.hpp>

#include <glm.hpp>

#include <vector>
#include <string>
#include <functional>
#include <array>

#include "Parameters.hpp"
#include "GUI-Textures.hpp"
#include "Agent.hpp"
#include "searchData.h"

#include "AudioManager.hpp"
#include "CudaManager.hpp"

class AgentManager {

	Point* ptData = nullptr;
	SolverSettings settingData;
	SearchOutput* outputData = nullptr;
	AgentData* agentData = nullptr;
	cudaGraphicsResource* cudaResource = nullptr;

	std::vector<Agent> agentList;

	static std::shared_ptr<GLOBAL_SETTINGS> GlobalSettings;
	std::shared_ptr<GROUP_SETTINGS> Settings;

	std::vector<std::function<void()>> searchMethods = {
		[&]() { searchMethod_triangle(); },
		[&]() { searchMethod_point(); },
		[&]() { searchMethod_line(); }
	};

	static std::vector<glm::mat3> colorMaps;

	CudaManager cudaManager;

	std::string name = "AgentManager";

public:
	
	std::vector<sf::Vector2i> searchArea;
	std::vector<std::string> searchMethodNames = {
		"Triangle",
		"Point",
		"Line"
	};
	static std::vector<std::string> colorMapNames;

	static SearchPreview searchPreview;

	AgentManager();
	~AgentManager();

	Agent* getAgent(int idx) { return &agentList[idx]; }

	void initializeAgents();

	void initializeCuda(GLint texId);

	void resetColors();

	std::string getName() const {
		return name;
	}

	void update_cpu(sf::Image&, float time);

	void update_cuda(float time); //for cuda

	static void setGlobalSettings(std::shared_ptr<GLOBAL_SETTINGS> Settings);

	void updateAgentCount();

	void updateSearchArea();

	void searchMethod_triangle();

    void searchMethod_point();

    void searchMethod_line();

	std::vector<sf::Vector2i> getSearchArea() {
		return searchArea;
	}

	std::shared_ptr<GROUP_SETTINGS> getSettings() {
		return Settings;
	}

	//glm::mat3 getColorMap() {
	//	return colorMaps[Settings->Agents.Color.colorMapIndex];
	//}

	static sf::Glsl::Mat3 getColorMapGlsl() {
		int idx = (int)GlobalSettings->Shader.colorMapIndex;
		glm::mat3 glmMat = colorMaps[idx];

		return sf::Glsl::Mat3({
			glmMat[0][0], glmMat[0][1], glmMat[0][2],
			glmMat[1][0], glmMat[1][1], glmMat[1][2],
			glmMat[2][0], glmMat[2][1], glmMat[2][2]
		});
	}
};