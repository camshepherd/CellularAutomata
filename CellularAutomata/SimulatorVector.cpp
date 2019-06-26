#include "SimulatorVector.h"


SimulatorVector::SimulatorVector(const int y, const int x, const IRules& _rules) : y_dim(y), x_dim(x), rules(_rules)
{
	// Initialise the blank cellStore
	cellStore = std::vector<std::vector<std::vector<int>>>();
	clear();
}


SimulatorVector::~SimulatorVector()
{
}

bool SimulatorVector::clear(bool addBlankFirstFrame) {
	cellStore.clear();
	// Add cells into the blank frame only if needed
	cellStore.push_back(std::vector<std::vector<int>>(addBlankFirstFrame ? y_dim : 0));
	for (auto& it : cellStore.back()) {
		it = std::vector<int>(addBlankFirstFrame ? x_dim : 0);
	}

	return true;
}

int SimulatorVector::getNumFrames() const {
	if (!cellStore[0].empty()) {
		return cellStore.size();
	}
	else {
		return 0;
	}

}

bool SimulatorVector::setCell(int y, int x, int new_val, int t) {
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

bool SimulatorVector::blankFrame() {
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

int SimulatorVector::getCell(int y, int x, int t) const {
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


int SimulatorVector::getYDim() {
	return y_dim;
}

int SimulatorVector::getXDim() {
	return x_dim;
}

bool SimulatorVector::copyFrame() {
	cellStore.push_back(std::vector<std::vector<int>>(cellStore.back().begin(), cellStore.back().end()));
	return true;
}