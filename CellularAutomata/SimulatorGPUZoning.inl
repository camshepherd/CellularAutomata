namespace CellularAutomata {
	template <typename T>
	SimulatorGPUZoning<T>::SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter) : SimulatorGPU<T>(y, x, rules, segmenter)
	{

	};

	template<typename T>
	SimulatorGPUZoning<T>::SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, int nBlocks, int nThreads) : SimulatorGPU<T>(y,x,rules,segmenter,nBlocks,nThreads)
	{
		
	};

	template <typename T>
	SimulatorGPUZoning<T>::~SimulatorGPUZoning() {};


	template <typename T>
	bool SimulatorGPUZoning<T>::setParams(int* list)
	{
		this->nSegments = list[0];
		this->y_dim = list[1];
		this->x_dim = list[2];
		this->nBlocks = list[3];
		this->nThreads = list[4];
		return true;
	}

}