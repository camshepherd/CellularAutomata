namespace CellularAutomata {
	template <typename T>
	SimulatorGPUZoning<T>::SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, IDeadZoneHandlerArray<T>& zoner) : SimulatorGPU<T>(y, x, rules, segmenter), zoner(zoner)
	{

	};

	template<typename T>
	SimulatorGPUZoning<T>::SimulatorGPUZoning(int y, int x, IRulesArray<T>& rules, ISegmenter& segmenter, IDeadZoneHandlerArray<T>& zoner, int nBlocks, int nThreads) : SimulatorGPU<T>(y,x,rules,segmenter,nBlocks,nThreads), zoner(zoner)
	{
		
	};

	template <typename T>
	SimulatorGPUZoning<T>::~SimulatorGPUZoning() {};

}