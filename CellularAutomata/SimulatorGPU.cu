#include <cuda_runtime.h>
#include "SimulatorGPU.hpp"
#include "IRulesArray.hpp"
#include "RulesArrayConway.hpp"
#include "RulesArrayBML.hpp"
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <cuda_runtime_api.h>

namespace CellularAutomata {

	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, const RulesArrayConway<T>* rules) {
		int y_dim = A[1];
		int x_dim = A[2];
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		for (int y = regions[tid * 4 + 2]; y < regions[tid * 4 + 3]; ++y) {
			for (int x = regions[tid * 4 + 4]; x < regions[tid * 4 + 5]; ++x) {
				//B[x + y * x_dim] = rules->l;
				B[x + y * x_dim] = rules->getNextState(A, y, x);
				if(B[x+y*x_dim] == 1)
				{
					printf("****Found one!******");
				}
				printf("%d",B[x + y * x_dim]);
			}
		}
	}

	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, const RulesArrayBML<T>* rules) {
		int y_dim = A[0];
		int x_dim = A[1];
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		for (int y = regions[tid * 4]; y < regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x < regions[tid * 4 + 3]; ++x) {
				//B[x + y * x_dim] = rules->l;
				B[x + y * x_dim] = rules->getNextState(A, y, x);
				printf("%d", B[x + y * x_dim]);
			}
		}
	}

	template <typename T>
	double SimulatorGPU<T>::stepForward(int steps) {
		this->timer.reset();
		T *currFrame, *newFrame;
		int frameSize = this->x_dim * this->y_dim;
		

		checkCudaErrors(cudaMallocManaged(&currFrame, sizeof(T) * frameSize + 2));
		checkCudaErrors(cudaMallocManaged(&newFrame, sizeof(T) * frameSize + 2));

		currFrame = this->cellStore.back();
		this->blankFrame();
		newFrame = this->cellStore.back();

		int* segments, *tempSegments;
		tempSegments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, this->nBlocks * this->nThreads);
		checkCudaErrors(cudaMallocManaged(&segments, sizeof(int) * 4 * tempSegments[0] + 3));
		segments = tempSegments;

		std::cout << std::endl;
		const RulesArrayConway<T>* con = dynamic_cast<const RulesArrayConway<T>*>(&(this->rules));
		const RulesArrayBML<T>* bml = dynamic_cast<const RulesArrayBML<T>*>(&(this->rules));
		if(con != nullptr)
		{
			checkCudaErrors(cudaMallocManaged(&con, sizeof(RulesArrayConway<T>)));
			con = dynamic_cast<const RulesArrayConway<T>*>(&(this->rules));
			stepForwardRegion<int> << <nBlocks, nThreads >> > (currFrame, newFrame, segments, con);
		}
		else if(bml != nullptr)
		{
			checkCudaErrors(cudaMallocManaged(&bml, sizeof(RulesArrayBML<T>)));
			bml = dynamic_cast<const RulesArrayBML<T>*>(&(this->rules));
			stepForwardRegion<int> << <nBlocks, nThreads >> > (currFrame, newFrame, segments, bml);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaFree(newFrame));
		checkCudaErrors(cudaFree(currFrame));
		checkCudaErrors(cudaFree(segments));
		//checkCudaErrors(cudaFree(con));
		//checkCudaErrors(cudaFree(bml));
		checkCudaErrors(cudaDeviceReset()),stderr;
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template class SimulatorGPU<int>;
}
