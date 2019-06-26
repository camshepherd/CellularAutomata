#include "SimulatorCPU.h"
#include <iostream>


SimulatorCPU::SimulatorCPU(int y, int x, IRules& rules, ISegmenter& segmenter) : SimulatorVector(y, x, rules), segmenter(segmenter)
{
}


SimulatorCPU::~SimulatorCPU()
{
}


bool SimulatorCPU::stepForwardRegion(int y_min, int y_max, int x_min, int x_max) {
	for (int y = y_min; y <= y_max; ++y) {
		for (int x = x_min; x <= x_max; ++x) {
			setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
		}
	}
	//std::cout << "Finished simulating" << std::endl;
	return true;
}

double SimulatorCPU::stepForward(int steps) {
	timer.reset();
	for (int u = 0; u < steps; ++u) {
		blankFrame();
		std::vector<std::thread> threads{};
		int num_threads = std::thread::hardware_concurrency();

		std::vector<std::tuple<int, int, int, int>> segments = segmenter.segment(y_dim, x_dim, num_threads);

		for (int k = 0; k < num_threads; ++k) {
			int y_min, y_max, x_min, x_max;
			std::tie(y_min, y_max, x_min, x_max) = segments[k];
			threads.push_back(std::thread(&SimulatorCPU::stepForwardRegion, this, y_min, y_max, x_min, x_max));
		}

		for (int ref = 0; ref < threads.size(); ++ref) {
			threads[ref].join();
		}
	}
	double elapsed = timer.elapsed();
	elapsedTime += elapsed;
	return elapsed;
}
