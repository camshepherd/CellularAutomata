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
	SimulatorGPU<T>::~SimulatorGPU()
	{
		checkCudaErrors(cudaDeviceReset());
	}

	/** Construct an instance of RulesArrayConway on the device
		@param dest: Destination address for the ruleset. Must already be (cuda)Malloced
		@param args: 2-element array containing: [y_dim, x_dim] of the frames to be simulated
	 */
	template <typename T>
	__global__ void constructConway(RulesArrayConway<T>* dest, int* args)
	{
		printf("\nydim: %d, xdIM: %d\n", args[0], args[1]);
		new (dest) RulesArrayConway<T>(args[0],args[1]);
	}

	/** Construct an instance of RulesArrayBML on the device
		@param dest: Destination address for the ruleset. Must already be (cuda)Malloced
		@param args: 2-element array containing: [y_dim, x_dim] of the frames to be simulated
	 */
	template <typename T>
	__global__ void constructBML(RulesArrayBML<T>* dest, int* args)
	{
		printf("\nydim: %d, xdIM: %d\n", args[0], args[1]);
		new (dest) RulesArrayBML<T>(args[0], args[1]);
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

	template <typename T>
	double SimulatorGPU<T>::stepForward(int steps) {
		this->timer.reset();
		// declare the variables needed
		int numSegments = this->nBlocks * this->nThreads;
		T *h_currFrame, *h_newFrame, *d_currFrame, *d_newFrame;
		int *h_segments, *d_segments;
		int *h_context,*d_context;
		int* h_dimensions, *d_dimensions;
		int frameSize = this->x_dim * this->y_dim;

		// define the host variables
		h_segments = this->segmenter.segmentToArray(this->y_dim, this->x_dim, numSegments);
		h_context = static_cast<int*>(malloc(sizeof(int) * 3));
		h_context[0] = this->y_dim;
		h_context[1] = this->x_dim;
		h_context[2] = numSegments;

		h_currFrame = this->cellStore.back();
		h_dimensions = static_cast<int*>(malloc(sizeof(int) * 2));
		h_dimensions[0] = this->x_dim;
		h_dimensions[1] = this->y_dim;
		// allocate the device memory
		checkCudaErrors(cudaMalloc(&d_currFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_newFrame, sizeof(T) * frameSize));
		checkCudaErrors(cudaMalloc(&d_segments, sizeof(int) * 4 * numSegments));
		checkCudaErrors(cudaMalloc(&d_context, sizeof(int) * 3));
		checkCudaErrors(cudaMalloc(&d_dimensions, sizeof(int) * 2));
		
		// copy over data to the device
		checkCudaErrors(cudaMemcpy(d_currFrame, h_currFrame, sizeof(T) * frameSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_context, h_context, sizeof(int) * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_segments, h_segments, sizeof(int) * 4 * numSegments, cudaMemcpyHostToDevice));
		cudaMemcpy(d_dimensions, h_dimensions, sizeof(int) * 2, cudaMemcpyHostToDevice);
		// Get the ruleset 
		RulesArrayConway<T>* h_con = dynamic_cast<RulesArrayConway<T>*>(&(this->rules));
		RulesArrayBML<T>* h_bml = dynamic_cast<RulesArrayBML<T>*>(&(this->rules));

		if (h_con != nullptr)
		{
			// Rules type is Conway
			// Get the rules set up on the device
			RulesArrayConway<T>* d_con;
			checkCudaErrors(cudaMalloc(&d_con, sizeof(RulesArrayConway<T>) * 2));
			constructConway<T><<<1,1>>>(d_con, d_dimensions);
			
			for (int step = 0; step < steps; ++step)
			{
				this->blankFrame();
				h_newFrame = this->cellStore.back();
				//no need to copy the new frame to the device, as every cell's value will be assigned to during the step process
				stepForwardRegion<int> << <nBlocks, nThreads >> > (d_currFrame, d_newFrame, d_segments, d_context, d_con);
				// copy back the data 
				cudaMemcpy(h_newFrame, d_newFrame, sizeof(T) * frameSize, cudaMemcpyDeviceToHost);

				// swap the pointers, ready for the next iteration
				T *temp = d_currFrame;
				d_currFrame = d_newFrame;
				d_newFrame = temp;
			}
			// Free up the space used by the ruleset
			checkCudaErrors(cudaFree(d_con));
		}
		else if((h_bml != nullptr))
		{
			// Rules type is Conway
			// Get the rules set up on the device
			RulesArrayBML<T>* d_bml;
			checkCudaErrors(cudaMalloc(&d_bml, sizeof(RulesArrayBML<T>) * 2));
			constructBML<T> << <1, 1 >> > (d_bml, d_dimensions);

			for (int step = 0; step < steps; ++step)
			{
				this->blankFrame();
				h_newFrame = this->cellStore.back();
				//no need to copy the new frame to the device, as every cell's value will be assigned to during the step process
				stepForwardRegion<int> << <nBlocks, nThreads >> > (d_currFrame, d_newFrame, d_segments, d_context, d_bml);
				// copy back the data 
				cudaMemcpy(h_newFrame, d_newFrame, sizeof(T) * frameSize, cudaMemcpyDeviceToHost);

				// swap the pointers, ready for the next iteration
				T *temp = d_currFrame;
				d_currFrame = d_newFrame;
				d_newFrame = temp;
			}
			// Free up the space used by the ruleset
			checkCudaErrors(cudaFree(d_bml));
		}

		// Free up all the space used by the function call
		checkCudaErrors(cudaFree(d_currFrame));
		checkCudaErrors(cudaFree(d_newFrame));
		checkCudaErrors(cudaFree(d_context));
		checkCudaErrors(cudaFree(d_segments));
		checkCudaErrors(cudaFree(d_dimensions));

		//checkCudaErrors(cudaDeviceReset());
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template class SimulatorGPU<int>;
}
