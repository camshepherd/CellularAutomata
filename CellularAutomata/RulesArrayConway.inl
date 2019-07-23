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


}
