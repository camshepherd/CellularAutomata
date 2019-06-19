#include "SimulatorCPU.h"
#include <iostream>


SimulatorCPU::SimulatorCPU(int y, int x, IRules& rules) : y_dim(y), x_dim(x), rules(rules)
{
	// Initialise the blank cellStore
	cellStore = std::vector<std::vector<std::vector<int>>>();
	clear();
}


SimulatorCPU::~SimulatorCPU()
{
}


bool SimulatorCPU::clear(bool addBlankFirstFrame) {
	cellStore.clear();
	// Add cells into the blank frame only if needed
	cellStore.push_back(std::vector<std::vector<int>>(addBlankFirstFrame ? y_dim : 0));
	for (auto& it : cellStore.back()) {
		it = std::vector<int>(addBlankFirstFrame ? x_dim : 0);
	}

	return true;
}

int SimulatorCPU::getNumFrames() const {
	if (!cellStore[0].empty()) {
		return cellStore.size();
	}
	else {
		return 0;
	}

}

bool SimulatorCPU::setCell(int y, int x, int new_val, int t) {
	t = (t == -1) ? getNumFrames() - 1 : t;
	if (t < 0 || t >= getNumFrames()) {
		throw std::runtime_error("The timestep t requested is outside of the valid range");
	}
	else if (!rules.isValid(new_val)) {
		throw std::runtime_error("The requested cell assignment is not a valid value");
	}
	else {
		cellStore[t][y][x] = new_val;
		return true;
	}
}

bool SimulatorCPU::blankFrame() {
	if (!cellStore[0].empty()) {
		cellStore.push_back(std::vector<std::vector<int>>(y_dim));
	}
	else {
		cellStore[0] = std::vector<std::vector<int>>(y_dim);
	}
	for (auto& it : cellStore.back()) {
		it = std::vector<int>(x_dim);
	}
	return true;
}

int SimulatorCPU::getCell(int y, int x, int t) const {
	t = (t == -1) ? getNumFrames() - 1 : t;
	if (y < 0 || y >= y_dim) {
		throw std::runtime_error("Y dimension out of range");
	}
	else if (x < 0 || x >= x_dim) {
		throw std::runtime_error("X dimension out of range");
	}
	else if (t < 0 || t >= getNumFrames()) {
		throw std::runtime_error("The timestep t requested is outside of the valid range");
	}
	else {
		return cellStore[t][y][x];
	}
}


bool SimulatorCPU::stepForwardTime(double seconds) {
	timer.reset();
	while (timer.elapsed() <= seconds) {
		stepForward();
	}
	return true;
}

bool SimulatorCPU::stepForwardRegion(int y_min, int y_max, int x_min, int x_max) {
	for (int y = y_min; y <= y_max; ++y) {
		for (int x = x_min; x <= x_max; ++x) {
			setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
		}
	}
	std::cout << "Finished simulating" << std::endl;
	return true;
}

bool SimulatorCPU::stepForward(int steps) {
	blankFrame();
	std::vector<std::thread> threads{};
	int num_threads = std::thread::hardware_concurrency();

	// CURRENTLY THE FRAME SPLITTING IS TOTALLY NON-FUNCTIONAL
	for (int k = 0; k < num_threads - 1; ++k) {
		threads.push_back(std::thread(&SimulatorCPU::stepForwardRegion, this, static_cast<int>(k * y_dim / num_threads), static_cast<int>((k + 1) * y_dim / num_threads - 1), static_cast<int>(k * x_dim / num_threads), static_cast<int>((k + 1) * x_dim / num_threads - 1)));
	}
	// Assign the final thread whatever portion is left
	threads.push_back(std::thread(&SimulatorCPU::stepForwardRegion, this, static_cast<int>((num_threads - 1) * y_dim / num_threads), y_dim - 1, static_cast<int>((num_threads - 1) * x_dim / num_threads), x_dim - 1));
	for (int ref = 0; ref < threads.size(); ++ref){
		threads[ref].join();
	}
	return true;
	getchar();
}

