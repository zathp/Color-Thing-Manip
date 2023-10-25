
#include <windows.h>
#include <GL/glew.h>

#include "searchData.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

__global__ void search(const SearchSettings settingData, const SearchInput* inputData,
	cudaTextureObject_t texObj, const Point* ptData, SearchOutput* outputData) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int agentCount = settingData.agentCount;

	if (i < agentCount && settingData.ptCount > 0) {

		float angles[3] = { 0,0,0 };
		angles[0] = inputData[i].dir + settingData.searchOffset + settingData.searchAngle;
		angles[1] = inputData[i].dir;
		angles[2] = inputData[i].dir - settingData.searchOffset - settingData.searchAngle;

		for (int dir = 0; dir < 3; dir++) {
			outputData[i].avgr[dir] = 0;
			outputData[i].avgg[dir] = 0;
			outputData[i].avgb[dir] = 0;
		}

		for (int pt = 0; pt < settingData.ptCount; pt++) {

			float ptx = (float)ptData[pt].x;
			float pty = (float)ptData[pt].y;

			for (int dir = 0; dir < 3; dir++) {

				float ptransx =
					(ptx * cosf(angles[dir]))
					- (pty * sinf(angles[dir]))
					;
				float ptransy =
					(ptx * sinf(angles[dir]))
					+ (pty * cosf(angles[dir]))
					;

				ptransx += inputData[i].posx;
				ptransy += inputData[i].posy;

				ptransx = (settingData.imWidth + (int)floor(ptransx)) % settingData.imWidth;
				ptransy = (settingData.imHeight + (int)floor(ptransy)) % settingData.imHeight;
				//float imPosx = ptransx / settingData.imWidth;
				//float imPosy = ptransy / settingData.imHeight;

				/*uchar4 pixelValue;
				surf2Dread(&pixelValue, surfObj, imPosx, imPosy);*/

				uchar4 pixelValue = tex2D<uchar4>(texObj, ptransx, ptransy);

				outputData[i].avgr[dir] += pixelValue.x;
				outputData[i].avgg[dir] += pixelValue.y;
				outputData[i].avgb[dir] += pixelValue.z;
			}
		}

		for (int dir = 0; dir < 3; dir++) {
			outputData[i].avgr[dir] /= settingData.ptCount;
			outputData[i].avgg[dir] /= settingData.ptCount;
			outputData[i].avgb[dir] /= settingData.ptCount;
		}
	}
}

extern "C" void run(int blockSize, const SearchSettings settingData, const SearchInput* inputData,
	cudaGraphicsResource* cudaResource, const Point* ptData, SearchOutput* outputData) {

	cudaArray_t textureArray;
	cudaGraphicsMapResources(1, &cudaResource);
	cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0);

	/*cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = textureArray;

	cudaSurfaceObject_t surfObj = 0;
	cudaCreateSurfaceObject(&surfObj, &resDesc);*/

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = textureArray;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	int agentCount = settingData.agentCount;

	int gridSize = (agentCount + blockSize - 1) / blockSize;

	search << <gridSize, blockSize>> > (settingData, inputData, texObj, ptData, outputData);

	cudaDeviceSynchronize();

	/*cudaDestroySurfaceObject(surfObj);*/
	cudaDestroyTextureObject(texObj);

	cudaGraphicsUnmapResources(1, &cudaResource);
}