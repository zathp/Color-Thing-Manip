#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "searchData.h"
#include <cmath>

__global__ void search(const SearchSettings settingData, const SearchInput* inputData,
	const Pixel* imData, const Point* ptData, SearchOutput* outputData) {

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

			float ptx = (float) ptData[pt].x;
			float pty = (float) ptData[pt].y;

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

				int imPosx = (settingData.imWidth + (int)floor(ptransx)) % settingData.imWidth;
				int imPosy = (settingData.imHeight + (int)floor(ptransy)) % settingData.imHeight;

				int pixPos = imPosx + settingData.imWidth * imPosy;

				outputData[i].avgr[dir] += imData[pixPos].r;
				outputData[i].avgg[dir] += imData[pixPos].g;
				outputData[i].avgb[dir] += imData[pixPos].b;
			}
		}

		for (int dir = 0; dir < 3; dir++) {
			outputData[i].avgr[dir] /= settingData.ptCount;
			outputData[i].avgg[dir] /= settingData.ptCount;
			outputData[i].avgb[dir] /= settingData.ptCount;
		}
	}
}

extern "C" void run(const SearchSettings settingData, const SearchInput* inputData,
	const Pixel* imData, const Point* ptData, SearchOutput* outputData, cudaStream_t &stream) {

	int agentCount = settingData.agentCount;

	int blockSize = 96;
	int gridSize = (agentCount + blockSize - 1) / blockSize;

	search << <gridSize, blockSize, 0, stream >> > (settingData, inputData, imData, ptData, outputData);
}
