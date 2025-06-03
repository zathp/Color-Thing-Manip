#include <cuda_runtime.h>
#include <surface_functions.h>
#include <device_launch_parameters.h>

#include "searchData.h"

__global__ void placeKernel(const SolverSettings settings,
    const AgentData* agentData,
    float2* prevPos,
    cudaSurfaceObject_t surfObj) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= settings.agentCount) return;

    float x = agentData[i].x;
    float y = agentData[i].y;

    float px = prevPos[i].x;
    float py = prevPos[i].y;

    uchar4 color = make_uchar4(
        __saturatef(agentData[i].color[0] / 255.0f) * 255.0f,
        __saturatef(agentData[i].color[1] / 255.0f) * 255.0f,
        __saturatef(agentData[i].color[2] / 255.0f) * 255.0f,
        255
    );

    // Always write final position
    int ix = ((int)floorf(x) + settings.imWidth) % settings.imWidth;
    int iy = ((int)floorf(y) + settings.imHeight) % settings.imHeight;
    surf2Dwrite(color, surfObj, ix * sizeof(uchar4), iy);

    if (settings.interpolatePlacement) {
        float dx = x - px;
        float dy = y - py;

        // unwrap X
        if (fabsf(dx) > settings.imWidth * 0.5f) {
            if (dx > 0) {
                px += settings.imWidth;  // unwrap across left edge
            }
            else {
                px -= settings.imWidth;  // unwrap across right edge
            }
            dx = x - px;
        }

        // unwrap Y
        if (fabsf(dy) > settings.imHeight * 0.5f) {
            if (dy > 0) {
                py += settings.imHeight;
            }
            else {
                py -= settings.imHeight;
            }
            dy = y - py;
        }

        float dist = sqrtf(dx * dx + dy * dy);

        int steps = max(1, (int)ceilf(dist));
        float stepX = dx / steps;
        float stepY = dy / steps;

        for (int s = 1; s < steps; ++s) {
            float fx = px + stepX * s;
            float fy = py + stepY * s;

            int ix_interp = ((int)floorf(fx) + settings.imWidth) % settings.imWidth;
            int iy_interp = ((int)floorf(fy) + settings.imHeight) % settings.imHeight;

            surf2Dwrite(color, surfObj, ix_interp * sizeof(uchar4), iy_interp);
        }
    }

    // Save current position to prevPos
    prevPos[i] = make_float2(x, y);
}

// Host-callable entry point
extern "C" void runPlace(int blockSize,
    const SolverSettings settings,
    const AgentData* dev_agentData,
    float2* dev_prevPos,
    cudaGraphicsResource* cudaResource) {

    // Map the OpenGL texture for CUDA access
    cudaArray_t textureArray;
    cudaGraphicsMapResources(1, &cudaResource);
    cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0);

    // Set up surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &resDesc);

    // Launch kernel
    int gridSize = (settings.agentCount + blockSize - 1) / blockSize;
    placeKernel << <gridSize, blockSize >> > (
        settings,
        dev_agentData,
		dev_prevPos,
        surfObj
    );
    cudaDeviceSynchronize();

    // Cleanup
    cudaDestroySurfaceObject(surfObj);
    cudaGraphicsUnmapResources(1, &cudaResource);
}

