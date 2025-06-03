#include "AgentManager.hpp"

#include <execution>
#include <algorithm>

using namespace std;
using namespace sf;

shared_ptr<GLOBAL_SETTINGS> AgentManager::GlobalSettings = nullptr;
SearchPreview AgentManager::searchPreview = SearchPreview();

std::vector<std::string> AgentManager::colorMapNames = {
		"Identity",
		"Invert",
		"Sepia",
		"Cool-Warm",
		"Purple-Orange",
		"Grayscale"
};

std::vector<glm::mat3> AgentManager::colorMaps = {
	// Identity (no change)
	glm::mat3(1.0f),

	// Invert
	glm::mat3(
		-1,  0,  0,
		 0, -1,  0,
		 0,  0, -1
	),

	// Sepia
	glm::mat3(
		0.393f, 0.769f, 0.189f,
		0.349f, 0.686f, 0.168f,
		0.272f, 0.534f, 0.131f
	),

	// Cool to warm
	glm::mat3(
		0.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f
	),

	// Purple–Orange shift
	glm::mat3(
		1.0f, 0.5f, 0.0f,
		0.0f, 0.5f, 1.0f,
		0.5f, 0.0f, 1.0f
	),

	// Grayscale
	glm::mat3(
		0.33f, 0.33f, 0.33f,
		0.33f, 0.33f, 0.33f,
		0.33f, 0.33f, 0.33f
	)
};

AgentManager::AgentManager() {

	if (GlobalSettings == nullptr) {
		return;
	}

	Settings = make_shared<GROUP_SETTINGS>();

	Settings->Agents.agentCount.setOnChange([this] {

		if (this->Settings->Agents.agentCount < 1) {
			this->Settings->Agents.agentCount.setValue(1);
		}

		if (this->Settings->Agents.agentCount > MAX_AGENTS) {
			this->Settings->Agents.agentCount.setValue(MAX_AGENTS);
		}

		this->updateAgentCount();
	});

	Settings->Agents.repulsion.setScaleSettings(-1.0f, 1.0f);
	Settings->Agents.memoryFactor.setScaleSettings(-2.0f, 2.0f);

	Settings->Agents.Search.searchSize.setOnChange([this] {
		this->updateSearchArea();
	});

	Settings->Agents.Search.searchAngle.setInputFunc([](float f) { return _Pi * f / 180.0f; });
	Settings->Agents.Search.searchAngle.setScaleSettings(0.0f, 180.0f);
	Settings->Agents.Search.searchAngle.setOnChange([&] {
		this->updateSearchArea();
		AgentManager::searchPreview.update(Settings->Agents.Search);
	});

	Settings->Agents.Search.searchAngleOffset.setInputFunc([](float f) { return _Pi * f / 180.0f; });
	Settings->Agents.Search.searchAngleOffset.setScaleSettings(-180.0f, 180.0f);
	Settings->Agents.Search.searchAngleOffset.setOnChange([&] {
		this->updateSearchArea();
		searchPreview.update(Settings->Agents.Search);
	});

	Settings->Agents.Search.searchMethodIndex.updateChoiceOptions(this->searchMethodNames);
	Settings->Agents.Search.searchMethodIndex.setOnChange([&] {
		this->updateSearchArea();
		searchPreview.update(Settings->Agents.Search);
	});

	Settings->Agents.Color.hueRotationAngle.setInputFunc([](float f) { return f * 180.0f / _Pi; });

	initializeAgents();
	updateSearchArea();
}

AgentManager::~AgentManager() {
	delete[] outputData;
	delete[] ptData;
	delete[] agentData;
}

void AgentManager::initializeAgents() {
    for (int i = 0; i < Settings->Agents.agentCount; i++) {
        agentList.push_back(Agent(i));
    }
	Agent::setColorMap(colorMaps[GlobalSettings->Shader.colorMapIndex]);
}

void AgentManager::initializeCuda(GLint texId) {

	if (CudaManager::isCudaCapable()) {

		cudaManager.init(texId);

		//host data
		outputData = new SearchOutput[MAX_AGENTS];
		ptData = new Point[MAX_SEARCH_PIXELS];
		agentData = new AgentData[MAX_AGENTS];

		//collect agent positions
		float2* prevPos = new float2[MAX_AGENTS];
		for (int i = 0; i < Settings->Agents.agentCount; i++) {
			Vector2f pos = getAgent(i)->getPos();
			prevPos[i].x = pos.x;
			prevPos[i].y = pos.y;
		}
		cudaManager.updatePreviousPositions(prevPos);

		//load search settings
		settingData.ptCount = 0;

		settingData.searchAngle = Settings->Agents.Search.searchAngle;
		settingData.searchOffset = Settings->Agents.Search.searchAngleOffset;

		settingData.imWidth = WIDTH;
		settingData.imHeight = HEIGHT;

		settingData.agentCount = Settings->Agents.agentCount;
		settingData.interpolatePlacement = Settings->Agents.interpolateColorDrop;
	}
}

void AgentManager::update_cuda(float time) {

	settingData.ptCount = searchArea.size();
	settingData.searchAngle = Settings->Agents.Search.searchAngle;
	settingData.searchOffset = Settings->Agents.Search.searchAngleOffset;
	settingData.agentCount = Settings->Agents.agentCount;
	settingData.interpolatePlacement = Settings->Agents.interpolateColorDrop;

	//copy search coordinate data to GPU
	for (int i = 0; i < searchArea.size(); i++) {
		ptData[i].x = searchArea[i].x;
		ptData[i].y = searchArea[i].y;
	}

	cudaManager.updateSettings(settingData, searchArea);

	cudaManager.copyAgentData(agentData);

	cudaManager.runCudaSearch();

	cudaManager.fetchSearchResults(outputData);

	std::for_each(std::execution::par_unseq, agentList.begin(), agentList.end(),
		[&](Agent& agent) {

			int i = agent.getId();

			//load output data into agent
			agent.avgColors[0] =
				Vector3f(
					outputData[i].avgr[0],
					outputData[i].avgg[0],
					outputData[i].avgb[0]
				);
			agent.avgColors[1] =
				Vector3f(
					outputData[i].avgr[1],
					outputData[i].avgg[1],
					outputData[i].avgb[1]
				);
			agent.avgColors[2] =
				Vector3f(
					outputData[i].avgr[2],
					outputData[i].avgg[2],
					outputData[i].avgb[2]
				);

			agent.updateDir(Settings);

			agent.applyColorTransformations(Settings, time);

			agent.getFinalColor(agentData[i].color);

			agent.updatePos(Settings);

			//update input data
			agentData[i].x = agent.getPos().x;
			agentData[i].y = agent.getPos().y;
			agentData[i].dir = agent.getDir();
		}
	);

	cudaManager.runCudaPlacement();


}
void AgentManager::update_cpu(Image& im, float time) {
	std::for_each(std::execution::par_unseq, agentList.begin(), agentList.end(),
		[&](Agent& agent) {

			agent.searchImage(Settings, im, searchArea);

			agent.updateDir(Settings);

			agent.applyColorTransformations(Settings, time);

			agent.updatePos(Settings, im);
		}
	);
}

void AgentManager::updateAgentCount() {
	while (agentList.size() < Settings->Agents.agentCount) {
		agentList.push_back(Agent(agentList.size()));
	}
	while (agentList.size() > Settings->Agents.agentCount) {
		agentList.pop_back();
	}
}

void AgentManager::updateSearchArea() {
	searchArea.clear();

	int idx = static_cast<int>(Settings->Agents.Search.searchMethodIndex);

	if (idx < 0 || idx >= searchMethods.size()) {
		cout << "Invalid search method index: " << idx << endl;
		return;
	}

	(searchMethods[idx])();
}

void AgentManager::searchMethod_line() {

	//include all x vals from 0 - search size and add to normTriangle
	float searchSize = Settings->Agents.Search.searchSize;

	for (int x = 0; x <= searchSize; x++) {
		searchArea.push_back(Vector2i(x, 0));
		if (searchArea.size() > MAX_SEARCH_PIXELS)
			return;
	}
}

void AgentManager::searchMethod_triangle() {

	float searchSize = Settings->Agents.Search.searchSize;
	float searchAngle = Settings->Agents.Search.searchAngle;

	Vector2f pts[3] = {
	Vector2f(0, 0),
	Vector2f(
		searchSize * cos(searchAngle / 2),
		searchSize * sin(searchAngle / 2)
	),
	Vector2f(
		searchSize * cos(-1 * searchAngle / 2),
		searchSize * sin(-1 * searchAngle / 2)
	)
	};

	float xStart = 0;
	float xEnd = pts[1].x;

	if (xEnd < xStart)
		swap(xStart, xEnd);

	float m1 = pts[1].y / pts[1].x;
	float m2 = pts[2].y / pts[2].x;

	if (m2 < m1)
		swap(m1, m2);

	for (int x = floor(xStart); x <= ceil(xEnd); x++) {

		float yStart = m1 * x;
		float yEnd = m2 * x;

		for (int y = floor(yStart); y <= ceil(yEnd); y++) {
			searchArea.push_back(Vector2i(x, y));
			if (searchArea.size() > MAX_SEARCH_PIXELS)
				return;
		}
	}
}

void AgentManager::searchMethod_point() {
	searchArea.push_back(Vector2i(
		Settings->Agents.Search.searchSize,
		0
	));
}

void AgentManager::setGlobalSettings(std::shared_ptr<GLOBAL_SETTINGS> settings) {
	AgentManager::GlobalSettings = settings;
	Agent::setGlobalSettings(settings);
}

void AgentManager::resetColors() {
	for (int i = 0; i < agentList.size(); i++) {
		agentList[i].randomizeColorBase();
	}
}