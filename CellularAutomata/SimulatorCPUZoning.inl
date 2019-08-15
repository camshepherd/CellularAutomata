namespace CellularAutomata {
	template <typename T>
	SimulatorCPUZoning<T>::SimulatorCPUZoning(int y, int x, IRules<T>& rules, ISegmenter& segmenter, IDeadZoneHandler<T>& zoner) : SimulatorCPU<T>(y, x, rules, segmenter), zoner(zoner)
	{

	};

	template <typename T>
	SimulatorCPUZoning<T>::~SimulatorCPUZoning() {};

	template <typename T>
	double SimulatorCPUZoning<T>::stepForward(int steps) {
		this->timer.reset();
		
		//int num_threads = std::thread::hardware_concurrency();
		int num_threads = this->nSegments;
		std::vector<std::tuple<int, int, int, int>> segments = this->segmenter.segment(SimulatorVector<T>::y_dim, SimulatorVector<T>::x_dim, num_threads);
		std::vector<std::thread> threads{};

		for (int u = 0; u < steps; ++u) {
			this->copyFrame();
			
			for (int k = 0; k < num_threads; ++k) {
				int y_min, y_max, x_min, x_max;
				std::tie(y_min, y_max, x_min, x_max) = segments[k];
				threads.push_back(std::thread(&SimulatorCPU<T>::stepForwardRegion, this, y_min, y_max, x_min, x_max));
			}

			for (int ref = 0; ref < threads.size(); ++ref) {
				threads[ref].join();
			}
			threads.clear();
			if (this->cellStore.size() >= 2) zoner.updateDeadZones(*(this->cellStore.end() - 2), *(this->cellStore.end() - 1));
		}
		
		double elapsed = this->timer.elapsed();
		this->elapsedTime += elapsed;
		return elapsed;
	}

	template <typename T>
	bool SimulatorCPUZoning<T>::stepForwardRegion(int y_min, int y_max, int x_min, int x_max) {
		for (int y = y_min; y <= y_max; ++y) {
			for (int x = x_min; x <= x_max; ++x) {
				if(y_min != -1 && x_min != -1) if (zoner.isLive(y, x)) {
					this->setCell(y, x, this->rules.getNextState(*(this->cellStore.end() - 2), y, x));
				}
			}
		}
		return true;
	}

	template <typename T>
	bool SimulatorCPUZoning<T>::setDimensions(int y, int x)
	{
		this->y_dim = y;
		this->x_dim = x;
		this->rebuildCellStore();
		this->zoner.setDimensions(this->y_dim, this->x_dim);
		return true;
	}

}