#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <cuda_runtime_api.h>
#include "RulesArrayConway.hpp"

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
	CUDA_FUNCTION int RulesArrayConway<T>::countNeighours(const T* cells, int y, int x) const {
		int count = 0;
		// assumed that the world will be a rectangle
		for (int _y = y - 1; _y <= y + 1; ++_y) {
			for (int _x = x - 1; _x <= x + 1; ++_x) {
				if (_y == y && _x == x) {
					continue;
				}
				else if (cells[(((_y + this->y_dim) % this->y_dim) * this->x_dim) + ((_x + this->x_dim) % this->x_dim) + 2]) {
					count += 1;
				}
			}
		}
		printf("Countt: %d", count);
		// TODO: REMOVE HARD CODING
		return count;
	}


	template class RulesArrayConway<int>;
	template class RulesArrayConway<bool>;
}