#include "CudaManager.hpp"

#include <iostream>

extern "C" void runSearch(int, SolverSettings, const AgentData*, cudaGraphicsResource*, Point*, SearchOutput*);
extern "C" void runPlace(int, SolverSettings, const AgentData*, float2*, cudaGraphicsResource*);

using namespace std;
using namespace sf;

bool CudaManager::cudaCapable = false;

bool CudaManager::checkCudaDevice() {

    //test if host is cuda capable
    int cudaDeviceCount = 0;

    if (cudaGetDeviceCount(&cudaDeviceCount) != cudaSuccess) {
        cudaCapable = false;
        return false;
    }

    if (cudaDeviceCount == 0) {
        cudaCapable = false;
        return false;
    }

    cudaCapable = true;
    return true;
}

void CudaManager::init(GLint textureId) {
	textureID = textureId;

    cudaSetDevice(0);
    checkCuda(cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    checkCuda(cudaMalloc((void**)&dev_ptData, sizeof(Point) * MAX_SEARCH_PIXELS));
    checkCuda(cudaMalloc((void**)&dev_outputData, sizeof(SearchOutput) * MAX_AGENTS));
    checkCuda(cudaMalloc((void**)&dev_agentData, sizeof(AgentData) * MAX_AGENTS));
    checkCuda(cudaMalloc((void**)&dev_prevPos, sizeof(float2) * MAX_AGENTS));
}

void CudaManager::updateSettings(const SolverSettings settings, const std::vector<Vector2i>& searchArea) {
	settingData = settings;
    std::vector<Point> pts(searchArea.size());
    for (size_t i = 0; i < searchArea.size(); ++i) {
        pts[i].x = searchArea[i].x;
        pts[i].y = searchArea[i].y;
    }
    checkCuda(cudaMemcpy(dev_ptData, pts.data(), sizeof(Point) * pts.size(), cudaMemcpyHostToDevice));
}

void CudaManager::updatePreviousPositions(float2* prevPos) {
    checkCuda(cudaMemcpy(dev_prevPos, prevPos, sizeof(float2) * settingData.agentCount, cudaMemcpyHostToDevice));
}

void CudaManager::copyAgentData(const AgentData* agentData) {
    checkCuda(cudaMemcpy(dev_agentData, agentData, sizeof(AgentData) * settingData.agentCount, cudaMemcpyHostToDevice));
}

void CudaManager::runCudaSearch() {
    runSearch(32, settingData, dev_agentData, cudaResource, dev_ptData, dev_outputData);
    checkCuda(cudaDeviceSynchronize());
}

void CudaManager::fetchSearchResults(SearchOutput* hostOutputData) {
    checkCuda(cudaMemcpy(hostOutputData, dev_outputData, sizeof(SearchOutput) * settingData.agentCount, cudaMemcpyDeviceToHost));
}

void CudaManager::runCudaPlacement() {
    runPlace(32, settingData, dev_agentData, dev_prevPos, cudaResource);
    checkCuda(cudaDeviceSynchronize());
}

void CudaManager::checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << "\n";
        cleanup();
        std::exit(-1);
    }
}

void CudaManager::cleanup() {
    cudaGraphicsUnregisterResource(cudaResource);
    cudaFree(dev_ptData);
    cudaFree(dev_outputData);
    cudaFree(dev_agentData);
    cudaFree(dev_prevPos);
}

CudaManager::~CudaManager() {
    if (cudaCapable) {
        cleanup();
        cudaDeviceReset();
    }
}
