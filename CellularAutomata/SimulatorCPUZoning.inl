namespace CellularAutomata {
	template <typename T>
	SimulatorCPUZoning<T>::SimulatorCPUZoning(int y, int x, IRules<T>& rules, ISegmenter& segmenter, IDeadZoneHandler<T>& zoner) : SimulatorCPU(y, x, rules, segmenter), zoner(zoner)
	{

	};

	template <typename T>
	SimulatorCPUZoning<T>::~SimulatorCPUZoning() {};

	template <typename T>
	double SimulatorCPUZoning<T>::stepForward(int steps) {
		timer.reset();
		for (int u = 0; u < steps; ++u) {
			copyFrame();
			std::vector<std::thread> threads{};
			int num_threads = std::thread::hardware_concurrency();

			std::vector<std::tuple<int, int, int, int>> segments = segmenter.segment(y_dim, x_dim, num_threads);

			for (int k = 0; k < num_threads; ++k) {
				int y_min, y_max, x_min, x_max;
				std::tie(y_min, y_max, x_min, x_max) = segments[k];
				threads.push_back(std::thread(&SimulatorCPU<T>::stepForwardRegion, this, y_min, y_max, x_min, x_max));
			}

			for (int ref = 0; ref < threads.size(); ++ref) {
				threads[ref].join();
			}
		}
		zoner.updateDeadZones(*(cellStore.end() - 2), *(cellStore.end() - 1));
		double elapsed = timer.elapsed();
		elapsedTime += elapsed;
		return elapsed;
	}

	template <typename T>
	bool SimulatorCPUZoning<T>::stepForwardRegion(int y_min, int y_max, int x_min, int x_max) {
		for (int y = y_min; y <= y_max; ++y) {
			for (int x = x_min; x <= x_max; ++x) {
				if (zoner.isLive(y, x)) {
					setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
				}
			}
		}
		return true;
	}
}