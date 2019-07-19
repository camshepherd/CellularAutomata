#include <cuda_runtime.h>
#include "SimulatorGPU.hpp"
#include "IRulesArray.hpp"
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <crt/host_defines.h>

namespace CellularAutomata {

	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, const IRulesArray<T>* rules) {
		int y_dim = A[0];
		int x_dim = A[1];
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		for (int y = regions[tid * 4]; y < regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x < regions[tid * 4 + 3]; ++x) {
				B[x + y * x_dim] = rules->getNextState(A, y, x);
			}
		}
	}

	template <typename T>
	double SimulatorGPU<T>::stepForward(int steps) {
		this->timer.reset();
		T* currFrame = cellStore.back();
		this->blankFrame();
		T* newFrame = cellStore.back();
		int frameSize = x_dim * y_dim;
		checkCudaErrors(cudaMallocManaged(&currFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMallocManaged(&newFrame, sizeof(T) * frameSize));
		int* segments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, this->nBlocks * this->nThreads);
		checkCudaErrors(cudaMallocManaged(&segments, sizeof(int) * 4 * this->nThreads * this->nBlocks));
		const IRulesArray<T>* rules = &this->rules;
		checkCudaErrors(cudaMallocManaged(&rules, sizeof(rules)));
		//stepForwardRegion<int> <<<nBlocks, nThreads>>> (currFrame, newFrame, segments, rules);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaFree(newFrame));
		checkCudaErrors(cudaFree(currFrame));
		checkCudaErrors(cudaFree(segments));
		cudaDeviceReset();
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template class SimulatorGPU<int>;
}
