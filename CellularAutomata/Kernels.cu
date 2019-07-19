#include <cuda_runtime.h>
#include "IRulesArray.hpp"
#include <crt/host_defines.h>

namespace CellularAutomata {
	namespace GPU {
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
	}
}