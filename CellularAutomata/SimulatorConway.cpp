#include "SimulatorConway.h"



SimulatorConway::SimulatorConway(int y, int x) : y_dim(y), x_dim(x)
{
	// Initialise the blank cellStore
	clear();
}


SimulatorConway::~SimulatorConway()
{
}

bool SimulatorConway::clear(bool addBlankFirstFrame = true) {
	try {
		// check if the cellStore is initialised or not
		cellStore.empty(); 
	}
	catch (...) {
		cellStore = std::vector<std::vector<std::vector<int>>>();
	}
	cellStore.clear();
	// Add cells into the blank frame only if needed
	cellStore.push_back(std::vector<std::vector<int>>(addBlankFirstFrame ? y_dim : 0));
	cellStore.back().push_back(std::vector<int>(addBlankFirstFrame ? x_dim : 0));
}

int SimulatorConway::getNumFrames() {
	return cellStore.size();
}

bool SimulatorConway::setCell(int y, int x, int new_val, int t = -1) {
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
	else if (new_val < cell_min || new_val > cell_max) {
		throw std::runtime_error("The requested cell value is outside of the valid range");
	}
	else {
		cellStore[t][y][x] = new_val;
		return true;
	}
}

bool SimulatorConway::blankFrame() {
	cellStore.push_back(std::vector<std::vector<int>>(y_dim));
	cellStore.back().push_back(std::vector<int>(x_dim));
}

int SimulatorConway::getCell(int y, int x, int t = -1) {
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

int SimulatorConway::countNeighbours(int y, int x, int t=-1) {
	int t = (t == -1) ? getNumFrames() - 1 : t;
	int count = 0;
	if (y < 0 || y >= y_dim) {
		throw std::runtime_error("Y dimension out of range");
	}
	else if (x < 0 || x >= x_dim) {
		throw std::runtime_error("X dimension out of range");
	}
	int xmin, xmax, ymin, ymax;
	xmin = (x == 0) ? x : x - 1;
	xmax = (x == x_dim - 1) ? x : x + 1;
	ymin = (y == 0) ? y : y - 1;
	ymax = (y == x_dim - 1) ? y : x + 1;
	for (int y = ymin; y <= ymax; y++) {
		for (int x = xmin; x <= xmax; x++) {
			if (getCell(y, x)) {
				++count;
			}
		}
	}
	return count;
}

int SimulatorConway::updateCell(int y, int x, int t=-1) {
	t = (t == -1) ? getNumFrames() - 1 : t;
	int cell_count = countNeighbours(y, x, t-1);
	if (getCell(y,x,t-1)) {
		// is alive
		if (cell_count < live_min || cell_count > live_max) {
			setCell(y, x, 0);
			return 0;
		}
	}
	else {
		// is dead
		if (cell_count >= live_min || cell_count <= live_max) {
			// comes alive
			setCell(y, x, 1);
			return 1;
		}
	}
}

bool SimulatorConway::stepForward(int steps = 1) {
	if (steps < 0) {
		throw std::runtime_error("The simulation cannnot work backwards");
	}
	else {
		for (int step = 0; step < steps; ++steps) {
			blankFrame();
			for (int y = 0; y < y_dim; ++y) {
				for (int x = 0; x < x_dim; ++x) {
					updateCell(y, x);
				}
			}
		}
	}
	return true;
}

bool SimulatorConway::stepForward(float seconds) {
	timer.reset();
	while (timer.elapsed <= seconds) {
		stepForward();
	}
	
}