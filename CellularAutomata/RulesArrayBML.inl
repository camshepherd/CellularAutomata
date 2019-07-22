#include <cuda_runtime_api.h>

namespace CellularAutomata {
	template <typename T>
	RulesArrayBML<T>::RulesArrayBML()
	{
		// make sure that the frame dimensions are valid
		this->setFrameDimensions(3, 3);
	}

	template <typename T>
	RulesArrayBML<T>::RulesArrayBML(int y_dim, int x_dim) : RulesArrayBML<T>::RulesArrayBML(){
		this->setFrameDimensions(y_dim, x_dim);
	}

	template <typename T>
	RulesArrayBML<T>::~RulesArrayBML()
	{
	}

	template <typename T>
	bool RulesArrayBML<T>::isValid(T cellState) const {
		switch (cellState) {
		case 0:
			return true;
			break;
		case 1:
			return true;
			break;
		case 2:
			return true;
			break;
		default:
			return false;
			break;
		}
	}

	template <typename T>
	__host__ __device__ T RulesArrayBML<T>::getNextState(T* cells, int y, int x) const {
		switch (cells[x + y*this->x_dim]) {
		case 0:
			if (cells[(y*this->x_dim) + ((x - 1 + this->x_dim) % this->x_dim)] == 1) {
				return 1;
			}
			else if (cells[(((y - 1 + this->y_dim) % this->y_dim) * this->x_dim) + x] == 2) {
				return 2;
			}
			else {
				return 0;
			}
			break;
		case 1:
			if (cells[(y * this->x_dim) + ((x + 1) % this->x_dim)] == 0) {
				if (cells[(((y - 1 + this->y_dim) % this->y_dim) * this->x_dim) + x] == 2) {
					return 2;
				}
				else {
					return 0;
				}
			}
			else {
				return 1;
			}
			break;
		case 2:
			if (cells[(((y + 1 + this->y_dim) % this->y_dim) * this->x_dim) + x] == 0) {
				return 0;
			}
			else {
				return 0;
			}
			break;
		default:
			throw std::runtime_error("An invalid state got into the cell store");
		};
	}

	template <typename T>
	T RulesArrayBML<T>::getMaxValidState() const {
		return 2;
	}

}
