#include <cuda_runtime.h>
#include "SimulatorGPU.hpp"
#include "IRulesArray.hpp"
#include "RulesArrayConway.hpp"
#include "RulesArrayBML.hpp"
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <cuda_runtime_api.h>

#define Y_DIM context[0]
#define X_DIM context[1]
#define NUM_SEGMENTS context[2]


namespace CellularAutomata {

	template <typename T>
	CUDA_FUNCTION T RulesArrayConway<T>::getNextState(T* cells, int y, int x) const {
		int count = countNeighours(cells, y, x);
		printf("Count: %d\n", count);
		if (cells[y*this->x_dim + x]) {
			//alive
			if (count >= live_min && count <= live_max) {
				return 1;
			}
		}
		else {
			//dead
			if (count >= birth_min && count <= birth_max) {
				return 1;
			}
		}
		return 0;
	}


	template <typename T>
	CUDA_FUNCTION int RulesArrayConway<T>::countNeighours(T* cells, int y, int x) const {
		int count = 0;
		// assumed that the world will be a rectangle
		for (int _y = y - 1; _y <= y + 1; ++_y) {
			for (int _x = x - 1; _x <= x + 1; ++_x) {
				if (_y == y && _x == x) {
					continue;
				}
				else
				{
					int pos = (((_y + this->y_dim) % this->y_dim) * this->x_dim) + ((_x + this->x_dim) % this->x_dim);
					printf("Attempting to access position %d",pos) ;
					if (cells[(((_y + this->y_dim) % this->y_dim) * this->x_dim) + ((_x + this->x_dim) % this->x_dim)]) {
						count += 1;
					}
				}
				
			}
		}
		printf("Countt: %d", count);
		// TODO: REMOVE HARD CODING
		return count;
	}


	template class RulesArrayConway<int>;
	template class RulesArrayConway<bool>;



	template <typename T>
	SimulatorGPU<T>::~SimulatorGPU()
	{
		checkCudaErrors(cudaDeviceReset());
	}

	template <typename T>
	__global__ void constructConway(RulesArrayConway<T>* dest, int* args)
	{
		printf("\nydim: %d, xdIM: %d\n", args[0], args[1]);
		new (dest) RulesArrayConway<T>(args[0],args[1]);
	}

	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, int* context, RulesArrayConway<T>* rules) {
		printf("\nStarting kernel");
		printf("numSegments: %d", NUM_SEGMENTS);

		
		// context: y_dim, x_dim, numSegments
		const int tid = threadIdx.x + blockDim.x * blockIdx.x;

		//print out segments
		if (tid == 0) {
			printf("Printing A:\n");
			for (int u = 0; u < Y_DIM * X_DIM; ++u)
			{
				printf("\n%d: %d", u, A[u]);
			}
		}
		if(tid >= NUM_SEGMENTS)
		{
			printf("Returning %d", tid);
			// if there isn't the data for the thread to read, end
			return;
		}
		for (int y = regions[tid * 4]; y <= regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x <= regions[tid * 4 + 3]; ++x) {
				printf("STUFF");
				if(y == -1 && x == -1)
				{
					return;
				}
				//B[x + y * X_DIM] = rules->k;
				//rules->getNextState(A, y, x);
				printf("\nPrinting to %d, %d", y, x);
				//B[x + y * X_DIM] = 1;
				B[x + y * X_DIM] = rules->getNextState(A, y, x);
				if(B[x+y*X_DIM] == 1)
				{
					printf("****Found one!******");
					printf("%d", B[x + y * X_DIM]);
				}
			}
		}
	}

	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, int* context, RulesArrayBML<T>* rules) {
		int y_dim = A[0];
		int x_dim = A[1];
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		for (int y = regions[tid * 4]; y < regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x < regions[tid * 4 + 3]; ++x) {
				//B[x + y * x_dim] = rules->l;
				//B[x + y * x_dim] = rules->getNextState(A, y, x);
				printf("%d", B[x + y * x_dim]);
			}
		}
	}

	template <typename T>
	double SimulatorGPU<T>::stepForward(int steps) {
		const int orientation = 1;
		int numSegments = this->nBlocks * this->nThreads;
		printf("\nnBlocks: %d, nThreads: %d", this->nBlocks, this->nThreads);
		//int numSegments = std::min<int>(this->nBlocks * this->nThreads, orientation ? x_dim : y_dim);
		this->timer.reset();
		T *h_currFrame, *h_newFrame, *d_currFrame, *d_newFrame;
		int *h_segments, *d_segments;
		int *h_context = static_cast<int*>(malloc(sizeof(int) * 3));
		int *d_context;
		int frameSize = this->x_dim * this->y_dim;

		h_segments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, numSegments);

		


		h_context[0] = this->y_dim;
		h_context[1] = this->x_dim;
		h_context[2] = numSegments;

		// allocate host memory
		h_currFrame = this->cellStore.back();
		// no need to set h_currFrame explicitly after the first run

		// allocate the device memory
		checkCudaErrors(cudaMalloc(&d_currFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_newFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_segments, sizeof(int) * 4 * numSegments));
		checkCudaErrors(cudaMalloc(&d_context, sizeof(int) * 3));


		checkCudaErrors(cudaMemcpy(d_currFrame, h_currFrame, sizeof(T) * frameSize, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(d_context, h_context, sizeof(int) * 3, cudaMemcpyHostToDevice));
		
		printf("\nHOST\n");
		for (int u = 0; u < numSegments * 4; ++u)
		{
			printf("\n%d: %d", u, h_segments[u]);
		}
		
		checkCudaErrors(cudaMemcpy(d_segments, h_segments, sizeof(int) * 4 * numSegments, cudaMemcpyHostToDevice));

		RulesArrayConway<T>* h_con = dynamic_cast<RulesArrayConway<T>*>(&(this->rules));
		RulesArrayBML<T>* h_bml = dynamic_cast<RulesArrayBML<T>*>(&(this->rules));

		int* h_dimensions = static_cast<int*>(malloc(sizeof(int) * 2));
		h_dimensions[0] = this->x_dim;
		h_dimensions[1] = this->y_dim;
		printf("h_dimensions are x: %d, y: %d", h_dimensions[0], h_dimensions[1]);

		int* d_dimensions;
		checkCudaErrors(cudaMalloc(&d_dimensions, sizeof(int) * 2));
		cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 2, cudaMemcpyHostToDevice);

		if (h_con != nullptr)
		{
			printf("\nCreating Conway ruleset");
			printf("\nSize of conway ruleset: %llu\n", sizeof(RulesArrayConway<T>));
			RulesArrayConway<T>* d_con;
			checkCudaErrors(cudaMalloc(&d_con, sizeof(RulesArrayConway<T>) * 2));
			constructConway<T><<<1,1>>>(d_con, d_dimensions);
			printf("\nAllocated memory for ruleset");
			printf("Copied across ruleset");
			for (int step = 0; step < steps; ++step)
			{
				this->blankFrame();
				h_newFrame = this->cellStore.back();
				//no need to copy the new frame to the device, as every value will be replaced during the step process
				//checkCudaErrors(cudaMemcpy(d_newFrame, h_newFrame, sizeof(T) * frameSize, cudaMemcpyHostToDevice));
				stepForwardRegion<int> << <nBlocks, nThreads >> > (d_currFrame, d_newFrame, d_segments, d_context, d_con);
				// copy back the data 
				cudaMemcpy(h_newFrame, d_newFrame, sizeof(T) * frameSize, cudaMemcpyDeviceToHost);

				// swap the pointers, ready for next iteration
				T *temp = d_currFrame;
				d_currFrame = d_newFrame;
				d_newFrame = temp;
			}
			
		}

		checkCudaErrors(cudaFree(d_currFrame));
		checkCudaErrors(cudaFree(d_newFrame));
		checkCudaErrors(cudaFree(d_context));
		checkCudaErrors(cudaFree(d_segments));

		//checkCudaErrors(cudaDeviceReset());
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template class SimulatorGPU<int>;
}
