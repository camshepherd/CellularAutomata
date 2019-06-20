#include "SimulatorSequential.h"

SimulatorSequential::SimulatorSequential(const int y, const int x, const IRules& _rules) : y_dim(y), x_dim(x), rules(_rules)
{
	// Initialise the blank cellStore
	cellStore = std::vector<std::vector<std::vector<int>>>();
	clear();
}


SimulatorSequential::~SimulatorSequential()
{
}

bool SimulatorSequential::clear(bool addBlankFirstFrame) {
	cellStore.clear();
	// Add cells into the blank frame only if needed
	cellStore.push_back(std::vector<std::vector<int>>(addBlankFirstFrame ? y_dim : 0));
	for (auto& it : cellStore.back()) {
		it = std::vector<int>(addBlankFirstFrame ? x_dim : 0);
	}
	
	return true;
}

int SimulatorSequential::getNumFrames() const {
	if (!cellStore[0].empty()) {
		return cellStore.size();
	}
	else {
		return 0;
	}
	
}

bool SimulatorSequential::setCell(int y, int x, int new_val, int t) {
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

bool SimulatorSequential::blankFrame() {
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

int SimulatorSequential::getCell(int y, int x, int t) const {
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

double SimulatorSequential::stepForward(int steps) {
	timer.reset();
	if (steps < 0) {
		throw std::runtime_error("The simulation cannnot work backwards");
	}
	else {
		for (int step = 0; step < steps; ++step) {
			blankFrame();
			for (int y = 0; y < y_dim; ++y) {
				for (int x = 0; x < x_dim; ++x) {
					// calculate the new cell value based on the old one
					setCell(y, x, rules.getNextState(*(cellStore.end() - 2), y, x));
				}
			}
		}
	}
	elapsedTime += timer.elapsed();
	return timer.elapsed();
}

int SimulatorSequential::getYDim() {
	return y_dim;
}

int SimulatorSequential::getXDim() {
	return x_dim;
}