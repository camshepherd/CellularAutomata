#include <iostream>

namespace CellularAutomata {

	



	template <typename T>
	T SimulatorGPU<T>::getMaxValidState() {
		return this->rules.getMaxValidState();
	}

	template <typename T>
	bool SimulatorGPU<T>::setLaunchParams(int nBlocks, int nThreads, int nSegments) {
		this->nBlocks = nBlocks;
		this->nThreads = nThreads;
		this->nSegments = nSegments;
		return true;
	}

	template <typename T>
	bool SimulatorGPU<T>::setDimensions(int y , int x)
	{
		this->y_dim = y;
		this->x_dim = x;
		return true;
	}
}