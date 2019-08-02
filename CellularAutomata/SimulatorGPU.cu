#include <cuda_runtime.h>
#include "SimulatorGPU.hpp"
#include "SimulatorGPUZoning.hpp"
#include "IRulesArray.hpp"
#include "RulesArrayConway.hpp"
#include "RulesArrayBML.hpp"
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <cuda_runtime_api.h>
#include "ZonerArrayPixels.hpp"
#include "IDeadZoneHandlerArray.hpp"

#define Y_DIM context[0]
#define X_DIM context[1]
#define NUM_SEGMENTS context[2]


namespace CellularAutomata {

	/** Construct an instance of RulesArrayConway on the device
		@param dest: Destination address for the ruleset. Must already be (cuda)Malloced
		@param args: 3-element array containing: [y_dim, x_dim, Ruleset type] of the frames to be simulated
	 */
	template <typename T>
	__global__ void constructRuleset(IRulesArray<T>* dest, int* args)
	{
		switch (args[2])
		{
		case 0:
			new (dest) RulesArrayConway<T>(args[0], args[1]);
			break;
		case 1:
			new (dest) RulesArrayBML<T>(args[0], args[1]);
			break;
		}
	}

	/** Construct an instance of ZonerArrayPixels on the device
		@param dest: Destination address for the ruleset. Must already be (cuda)Malloced
		@param args: 4-element array containing: [y_dim, x_dim, y_max, x_max] of the frames to be simulated
	 */
	template <typename T>
	__global__ void constructZoner(IDeadZoneHandlerArray<T>* dest, int *dims, int *maxDims, bool* A, bool* B)
	{
		new (dest) ZonerArrayPixels<T>(dims, maxDims, A, B);
	}

	template <typename T>
	SimulatorGPU<T>::~SimulatorGPU()
	{
		checkCudaErrors(cudaDeviceReset());
	}


	template <typename T>
	SimulatorGPUZoning<T>::~SimulatorGPUZoning()
	{
		checkCudaErrors(cudaFree(d_zoner));
		checkCudaErrors(cudaFree(d_zoner_a));
		checkCudaErrors(cudaFree(d_zoner_b));
		checkCudaErrors(cudaFree(d_zoner_dims));
		checkCudaErrors(cudaFree(d_zoner_maxDims));
	};



	template <typename T>
	bool SimulatorGPUZoning<T>::setDimensions(int y, int x)
	{
		this->y_dim = y;
		this->x_dim = x;
		int* dims = static_cast<int*> (malloc(sizeof(int) * 2));
		dims[0] = y;
		dims[1] = x;
		checkCudaErrors(cudaMemcpy(d_zoner_dims, dims, sizeof(int) * 2, cudaMemcpyHostToDevice));
		free(dims);
		refreshZoner << <1, 1 >> > (d_zoner);
		return true;
	}

	template <typename T>
	SimulatorGPU<T>::SimulatorGPU(int ydim, int xdim, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads) : SimulatorArray<T>(ydim, xdim, rules), segmenter(segmenter), nBlocks(nBlocks), nThreads(nThreads)
	{
		size_t size;
		checkCudaErrors(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
		printf("The size is: %llu", size);
		size = 2000000000;
		checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
		checkCudaErrors(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
		printf("And now it is: %llu", size);
		nSegments = this->y_dim * this->x_dim;
		printf("The dimensions of SimulatorGPU are: %d and %d\n", this->y_dim, this->x_dim);
	}

	template<typename T>
	SimulatorGPUZoning<T>::SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads, int y_max, int x_max) : SimulatorGPU<T>(y, x, rules, segmenter, nBlocks, nThreads), y_max(y_max), x_max(x_max)
	{
		size_t size;
		checkCudaErrors(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
		printf("The size is: %llu", size);
		size = 2000000000;
		checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
		checkCudaErrors(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
		printf("And now it is: %llu", size);
		printf("The size is: %llu", size);
		int *dims = static_cast<int*>(malloc(sizeof(int) * 2));
		dims[0] = y;
		dims[1] = x;
		int *maxDims = static_cast<int*>(malloc(sizeof(int) * 2));
		maxDims[2] = y_max;
		maxDims[3] = x_max;
		checkCudaErrors(cudaMalloc(&d_zoner_dims, sizeof(int) * 2));
		checkCudaErrors(cudaMemcpy(d_zoner_dims, dims, sizeof(int) * 2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc(&d_zoner_maxDims, sizeof(int) * 2));
		checkCudaErrors(cudaMemcpy(d_zoner_maxDims, maxDims, sizeof(int) * 2, cudaMemcpyHostToDevice));
		
		checkCudaErrors(cudaMalloc(&d_zoner_a, sizeof(bool) * y_max * x_max));
		checkCudaErrors(cudaMalloc(&d_zoner_b, sizeof(bool) * y_max * x_max));
		//free(dims);
		//free(maxDims);
		// Create the zoner
		constructZoner<T> << <1, 1 >> > (d_zoner, d_zoner_dims, d_zoner_maxDims, d_zoner_a, d_zoner_b);
		
	};


	

	/** Refresh the instance of ZonerArrayPixels on th device
		@param dest: Destination address for the ruleset. Must already be (cuda)Malloced
	 */
	template <typename T>
	__global__ void refreshZoner(IDeadZoneHandlerArray<T>* zoner)
	{
		zoner->refresh();
	}


	/** Step forward the given region, using the given ruleset
		@param A: The previous frame of the simulation
		@param B: The frame to be simulated from A
		@param regions: The boundaries for the regions that each thread is responsible for
		@param context: Information necessary to understand the given inputs, containing: [y_dim, x_dim, numSegments]
		@param rules: The ruleset to be used to step forward through the given simulation
	 */
	template <typename T>
	__global__ void stepForwardRegion(T* A, T* B, int* regions, int* context, IRulesArray<T>* rules) {
		// context: y_dim, x_dim, numSegments
		const int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if(tid >= NUM_SEGMENTS)
		{
			// if there isn't the data for the thread to read, end
			return;
		}
		for (int y = regions[tid * 4]; y <= regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x <= regions[tid * 4 + 3]; ++x) {
				if(y == -1 && x == -1)
				{
					return;
				}
				B[x + y * X_DIM] = rules->getNextState(A, y, x);
			}
		}
	}

	/** Step forward the given region, using the given ruleset, and keeping track of the active zones.
		@param A: The previous frame of the simulation
		@param B: The frame to be simulated from A
		@param regions: The boundaries for the regions that each thread is responsible for
		@param context: Information necessary to understand the given inputs, containing: [y_dim, x_dim, numSegments]
		@param rules: The ruleset to be used to step forward through the given simulation
	 */
	template <typename T>
	__global__ void stepForwardRegionZoning(T* A, T* B, int* regions, int* context, IRulesArray<T>* rules, IDeadZoneHandlerArray<T>* zoner) {
		// context: y_dim, x_dim, numSegments
		const int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid >= NUM_SEGMENTS)
		{
			// if there isn't the data for the thread to read, end
			return;
		}
		for (int y = regions[tid * 4]; y <= regions[tid * 4 + 1]; ++y) {
			for (int x = regions[tid * 4 + 2]; x <= regions[tid * 4 + 3]; ++x) {
				if (y == -1 && x == -1)
				{
					return;
				}
				if(zoner->isLive(y,x))
				{
					B[x + y * X_DIM] = rules->getNextState(A, y, x);
				}
			}
		}
		// update the zoner, ready for the next round
		zoner->updateDeadZones(A, B);
	}


	template <typename T>
	double SimulatorGPU<T>::stepForward(int steps) {
		printf("SIMULATOR DIMENSIONS: %d, %d", this->y_dim, this->x_dim);
		this->timer.reset();
		// declare the variables needed
		//int numSegments = this->nBlocks * this->nThreads;
		T *h_currFrame, *h_newFrame, *d_currFrame, *d_newFrame;
		int *h_segments, *d_segments;
		int *h_context,*d_context;
		int* h_dimensions, *d_dimensions;
		int frameSize = this->x_dim * this->y_dim;

		// define the host variables
		h_segments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, nSegments);
		h_context = static_cast<int*>(malloc(sizeof(int) * 3));
		h_context[0] = this->y_dim;
		h_context[1] = this->x_dim;
		h_context[2] = nSegments;

		h_currFrame = this->cellStore.back();
		h_dimensions = static_cast<int*>(malloc(sizeof(int) * 3));
		h_dimensions[0] = this->y_dim;
		h_dimensions[1] = this->x_dim;
		h_dimensions[2] = 0;
		// allocate the device memory
		checkCudaErrors(cudaMalloc(&d_currFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_newFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_segments, sizeof(int) * 4 * nSegments));
		checkCudaErrors(cudaMalloc(&d_context, sizeof(int) * 3));
		checkCudaErrors(cudaMalloc(&d_dimensions, sizeof(int) * 3));
		
		// copy over data to the device
		checkCudaErrors(cudaMemcpy(d_currFrame, h_currFrame, sizeof(T) * frameSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_context, h_context, sizeof(int) * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_segments, h_segments, sizeof(int) * 4 * nSegments, cudaMemcpyHostToDevice));
		
		// Get the ruleset 
		RulesArrayConway<T>* h_con = dynamic_cast<RulesArrayConway<T>*>(&(this->rules));
		RulesArrayBML<T>* h_bml = dynamic_cast<RulesArrayBML<T>*>(&(this->rules));
		IRulesArray<T>* d_rules;
		if (h_con != nullptr)
		{
			// Rules type is Conway
			h_dimensions[2] = 0;
			cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 3, cudaMemcpyHostToDevice);
			// Get the rules set up on the device
			checkCudaErrors(cudaMalloc(&d_rules, sizeof(RulesArrayConway<T>) * 2));
			constructRuleset<T><<<1,1>>>(d_rules, d_dimensions);	
		}
		else if((h_bml != nullptr))
		{
			// Rules type is BML
			h_dimensions[2] = 1;
			cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 3, cudaMemcpyHostToDevice);
			// Get the rules set up on the device
			checkCudaErrors(cudaMalloc(&d_rules, sizeof(RulesArrayBML<T>)));
			constructRuleset<T> << <1, 1 >> > (d_rules, d_dimensions);
		}

		for (int step = 0; step < steps; ++step)
		{
			this->blankFrame();
			h_newFrame = this->cellStore.back();
			//no need to copy the new frame to the device, as every cell's value will be assigned to during the step process
			stepForwardRegion<T> << <nBlocks, nThreads >> > (d_currFrame, d_newFrame, d_segments, d_context, d_rules);
			// copy back the data 
			cudaMemcpy(h_newFrame, d_newFrame, sizeof(T) * frameSize, cudaMemcpyDeviceToHost);

			// swap the pointers, ready for the next iteration
			T *temp = d_currFrame;
			d_currFrame = d_newFrame;
			d_newFrame = temp;
		}

		// Free up all the space used by the function call
		checkCudaErrors(cudaFree(d_currFrame));
		checkCudaErrors(cudaFree(d_newFrame));
		checkCudaErrors(cudaFree(d_context));
		checkCudaErrors(cudaFree(d_segments));
		checkCudaErrors(cudaFree(d_dimensions));
		checkCudaErrors(cudaFree(d_rules));
		//checkCudaErrors(cudaDeviceReset());
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;


		// free the memory used on the host
		free(h_segments);
		free(h_context);
		free(h_dimensions);


		return elapsed;
	}

	template <typename T>
	double SimulatorGPUZoning<T>::stepForward(int steps) {

		printf("ydim: %d, xdim: %d, segments: %d, blocks: %d, threads: %d\n", this->y_dim, this->x_dim, this->nSegments, this->nBlocks, this->nThreads);

		this->timer.reset();
		// declare the variables needed
		//int numSegments = this->nBlocks * this->nThreads;
		T *h_currFrame, *h_newFrame, *d_currFrame, *d_newFrame;
		int *h_segments, *d_segments;
		int *h_context, *d_context;
		int* h_dimensions, *d_dimensions;
		int frameSize = this->x_dim * this->y_dim;

		// define the host variables
		h_segments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, this->nSegments);
		h_context = static_cast<int*>(malloc(sizeof(int) * 3));
		h_context[0] = this->y_dim;
		h_context[1] = this->x_dim;
		h_context[2] = this->nSegments;

		h_currFrame = this->cellStore.back();
		h_dimensions = static_cast<int*>(malloc(sizeof(int) * 3));
		h_dimensions[0] = this->y_dim;
		h_dimensions[1] = this->x_dim;
		h_dimensions[2] = 0;

		// allocate the device memory
		checkCudaErrors(cudaMalloc(&d_currFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_newFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_segments, sizeof(int) * 4 * this->nSegments));
		checkCudaErrors(cudaMalloc(&d_context, sizeof(int) * 3));
		checkCudaErrors(cudaMalloc(&d_dimensions, sizeof(int) * 3));
		

		// copy over data to the device
		checkCudaErrors(cudaMemcpy(d_currFrame, h_currFrame, sizeof(T) * frameSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_context, h_context, sizeof(int) * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_segments, h_segments, sizeof(int) * 4 * this->nSegments, cudaMemcpyHostToDevice));		

		// Get the ruleset 
		RulesArrayConway<T>* h_con = dynamic_cast<RulesArrayConway<T>*>(&(this->rules));
		RulesArrayBML<T>* h_bml = dynamic_cast<RulesArrayBML<T>*>(&(this->rules));
		IRulesArray<T>* d_rules;
		if (h_con != nullptr)
		{
			// Rules type is Conway
			h_dimensions[2] = 0;
			cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 3, cudaMemcpyHostToDevice);
			// Get the rules set up on the device
			checkCudaErrors(cudaMalloc(&d_rules, sizeof(RulesArrayConway<T>)));
			constructRuleset<T> << <1, 1 >> > (d_rules, d_dimensions);

		}
		else if (h_bml != nullptr)
		{
			// Rules type is BML
			h_dimensions[2] = 1;
			checkCudaErrors(cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 3, cudaMemcpyHostToDevice));
			// Get the rules set up on the device
			checkCudaErrors(cudaMalloc(&d_rules, sizeof(RulesArrayBML<T>)));
			constructRuleset<T> << <1, 1 >> > (d_rules, d_dimensions);
		}
		else
		{
			std::cout << "******************" << std::endl;
			std::cout << "Reached the land that should never be reached!!" << std::endl;
		}
		for (int step = 0; step < steps; ++step)
		{
			this->blankFrame();
			h_newFrame = this->cellStore.back();
			//no need to copy the new frame to the device, as every cell's value will be assigned to during the step process
			stepForwardRegionZoning<T> << <this->nBlocks, this->nThreads >> > (d_currFrame, d_newFrame, d_segments, d_context, d_rules, d_zoner);
			// copy back the data 
			cudaMemcpy(h_newFrame, d_newFrame, sizeof(T) * frameSize, cudaMemcpyDeviceToHost);

			// swap the pointers, ready for the next iteration
			T *temp = d_currFrame;
			d_currFrame = d_newFrame;
			d_newFrame = temp;
		}

		// Free up all the space used by the function call
		checkCudaErrors(cudaFree(d_currFrame));
		checkCudaErrors(cudaFree(d_newFrame));
		checkCudaErrors(cudaFree(d_context));
		checkCudaErrors(cudaFree(d_segments));
		checkCudaErrors(cudaFree(d_dimensions));
		checkCudaErrors(cudaFree(d_rules));

		//checkCudaErrors(cudaDeviceReset());
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;

		// free the memory used on the host
		free(h_segments);
		free(h_context);
		free(h_dimensions);

		return elapsed;
	}

	template class SimulatorGPU<bool>;
	template class SimulatorGPU<char>;
	template class SimulatorGPU<int>;
	template class SimulatorGPU<long int>;
	template class SimulatorGPU<long long int>;

	template class SimulatorGPUZoning<bool>;
	template class SimulatorGPUZoning<char>;
	template class SimulatorGPUZoning<int>;
	template class SimulatorGPUZoning<long int>;
	template class SimulatorGPUZoning<long long int>;
}
