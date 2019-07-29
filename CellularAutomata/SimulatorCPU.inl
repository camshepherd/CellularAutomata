#include <iostream>

namespace CellularAutomata {
	template <typename T>
	SimulatorCPU<T>::SimulatorCPU(int y, int x, IRules<T>& rules, ISegmenter& segmenter) : SimulatorVector<T>(y, x, rules), segmenter(segmenter)
	{
		nSegments = std::thread::hardware_concurrency();
	}

	template <typename T>
	SimulatorCPU<T>::~SimulatorCPU()
	{
	}

	template <typename T>
	bool SimulatorCPU<T>::stepForwardRegion(int y_min, int y_max, int x_min, int x_max) {
		if(y_min == -1 && x_min == -1)
		{
			return true;
		}
		for (int y = y_min; y <= y_max; ++y) {
			for (int x = x_min; x <= x_max; ++x) {
				this->setCell(y, x, this->rules.getNextState(*(this->cellStore.end() - 2), y, x));
			}
		}
		//std::cout << "Finished simulating" << std::endl;
		return true;
	}

	template <typename T>
	double SimulatorCPU<T>::stepForward(int steps) {
		this->timer.reset();
		//int num_threads = std::thread::hardware_concurrency();
		int num_threads = nSegments;
		std::vector<std::tuple<int, int, int, int>> segments = segmenter.segment(this->y_dim, this->x_dim, num_threads);
		std::vector<std::thread> threads{};
		
		for (int u = 0; u < steps; ++u) {
			this->blankFrame();
			for (int k = 0; k < num_threads; ++k) {
				int y_min, y_max, x_min, x_max;
				std::tie(y_min, y_max, x_min, x_max) = segments[k];
				threads.push_back(std::thread(&SimulatorCPU<T>::stepForwardRegion, this, y_min, y_max, x_min, x_max));
			}

			for (int ref = 0; ref < threads.size(); ++ref) {
				threads[ref].join();
			}
			threads.clear();
		}
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template <typename T>
	T SimulatorCPU<T>::getMaxValidState() {
		return this->rules.getMaxValidState();
	}

	template <typename T>
	bool SimulatorCPU<T>::setParams(int* list)
	{
		nSegments = list[0];
		return true;
	}

}