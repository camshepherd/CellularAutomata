#include <iostream>

namespace CellularAutomata {
	template <typename T>
	SimulatorGPU<T>::SimulatorGPU(int y, int x, const IRulesArray<T>& rules, ISegmenter& segmenter) : SimulatorArray<T>(y, x, rules), segmenter(segmenter)
	{
		setLaunchParams(2, 32);
	}

	template <typename T>
	SimulatorGPU<T>::SimulatorGPU(int ydim, int xdim, const IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads) : SimulatorArray<T>(ydim, xdim, rules), segmenter(segmenter), nBlocks(nBlocks), nThreads(nThreads)
	{
	}

	template <typename T>
	SimulatorGPU<T>::~SimulatorGPU()
	{
	}

	template <typename T>
	T SimulatorGPU<T>::getMaxValidState() {
		return this->rules.getMaxValidState();
	}

	template <typename T>
	bool SimulatorGPU<T>::setLaunchParams(int nBlocks, int nThreads) {
		this->nBlocks = nBlocks;
		this->nThreads = nThreads;
		return true;
	}
}