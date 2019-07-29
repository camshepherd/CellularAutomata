#include <cuda_runtime_api.h>

namespace CellularAutomata {
	template <typename T>
	CUDA_FUNCTION RulesArrayConway<T>::RulesArrayConway() : live_min(2), live_max(3), birth_min(3), birth_max(3), cell_min(0), cell_max(1)
	{
		this->setFrameDimensions(this->y_dim, this->x_dim);
	}

	template <typename T>
	CUDA_FUNCTION RulesArrayConway<T>::RulesArrayConway(int y_dim, int x_dim) : RulesArrayConway()
	{
		this->setFrameDimensions(y_dim, x_dim);
	}

	template <typename T>
	CUDA_FUNCTION RulesArrayConway<T>::RulesArrayConway(int _live_min, int _live_max, int _birth_min, int _birth_max, int _cell_min, int _cell_max, int y_dim, int x_dim) : live_min(_live_min), live_max(_live_max), birth_min(_birth_min), birth_max(_birth_max), cell_min(_cell_min), cell_max(_cell_max)
	{
		this->setFrameDimensions(y_dim, x_dim);
	}

	template <typename T>
	CUDA_FUNCTION RulesArrayConway<T>::~RulesArrayConway()
	{
		printf("\N*********CONWAY IS DEAD***********\N");
	}

	template <typename T>
	CUDA_FUNCTION bool RulesArrayConway<T>::isValid(T cellState) const {
		if (cellState >= cell_min && cellState <= cell_max) {
			return true;
		}
		else {
			return false;
		}
	}

	


	template <typename T>
	CUDA_FUNCTION T RulesArrayConway<T>::getMaxValidState() const {
		return 1;
	}

	template <typename T>
	CUDA_FUNCTION T RulesArrayConway<T>::getNextState(T* cells, int y, int x) const {
		int count = countNeighours(cells, y, x);
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
				else if (cells[(((_y + this->y_dim) % this->y_dim) * this->x_dim) + ((_x + this->x_dim) % this->x_dim)]) {
					count += 1;
				}
			}
		}
		return count;
	}


	template class RulesArrayConway<int>;
	template class RulesArrayConway<bool>;
}
