#pragma once

#include "Config.hpp"

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Graphics.hpp>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "searchData.h"
#include "Parameters.hpp"

class CudaManager {
public:

    CudaManager() {};
    ~CudaManager();

    static bool checkCudaDevice();
	static bool isCudaCapable() {
		return cudaCapable;
	}

    void init(GLint textureId);

    void updateSettings(const SolverSettings settings, const std::vector<sf::Vector2i>& searchArea);
    void updatePreviousPositions(float2* prevPos);
    void copyAgentData(const AgentData* hostAgentData);
    void runCudaSearch();
    void runCudaPlacement();
    void fetchSearchResults(SearchOutput* outputData);
    void cleanup();

private:

    static bool cudaCapable;

    Point* dev_ptData = nullptr;
    SearchOutput* dev_outputData = nullptr;
    AgentData* dev_agentData = nullptr;
    float2* dev_prevPos = nullptr;
    struct cudaGraphicsResource* cudaResource = nullptr;
	SolverSettings settingData;

    GLint textureID;

    void checkCuda(cudaError_t status);
};