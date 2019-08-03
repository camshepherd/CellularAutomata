namespace CellularAutomata {
	template <typename T>
	SimulatorVector<T>::SimulatorVector(const int y, const int x, const IRules<T>& _rules) : y_dim(y), x_dim(x), rules(_rules)
	{
		// Initialise the blank cellStore
		cellStore = std::vector<std::vector<std::vector<T>>>();
		clear();
	}

	template <typename T>
	SimulatorVector<T>::~SimulatorVector()
	{
	}

	template <typename T>
	bool SimulatorVector<T>::clear(bool addBlankFirstFrame) {
		cellStore.clear();
		// Add cells into the blank frame only if needed
		cellStore.push_back(std::vector<std::vector<T>>(addBlankFirstFrame ? y_dim : 0));
		for (auto& it : cellStore.back()) {
			it = std::vector<T>(addBlankFirstFrame ? x_dim : 0);
		}

		return true;
	}

	template <typename T>
	int SimulatorVector<T>::getNumFrames() const {
		if (!cellStore[0].empty()) {
			return cellStore.size();
		}
		else {
			return 0;
		}

	}

	template <typename T>
	bool SimulatorVector<T>::setCell(int y, int x, T new_val, int t) {
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

	template <typename T>
	bool SimulatorVector<T>::blankFrame() {
		if (!cellStore[0].empty()) {
			cellStore.push_back(std::vector<std::vector<T>>(y_dim));
		}
		else {
			cellStore[0] = std::vector<std::vector<T>>(y_dim);
		}
		for (auto& it : cellStore.back()) {
			it = std::vector<T>(x_dim);
		}
		return true;
	}

	template <typename T>
	T SimulatorVector<T>::getCell(int y, int x, int t) const {
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

	template <typename T>
	int SimulatorVector<T>::getYDim() {
		return y_dim;
	}

	template <typename T>
	int SimulatorVector<T>::getXDim() {
		return x_dim;
	}

	template <typename T>
	bool SimulatorVector<T>::copyFrame() {
		cellStore.push_back(std::vector<std::vector<T>>(cellStore.back().begin(), cellStore.back().end()));
		return true;
	}

	template <typename T>
	void SimulatorVector<T>::printFrame(int frameNumber) {
		if (frameNumber == -1) {
			frameNumber = this->getNumFrames() - 1;
		}
		std::cout << "______________ START" << std::endl;
		for (int y = 0; y < this->y_dim; ++y) {
			std::cout << "|\n";
			for (int x = 0; x < this->x_dim; ++x) {
				std::cout << this->getCell(y, x, frameNumber);
			}
		}
		std::cout << "\n______________ END" << std::endl;
	}
	template <typename T>
	void SimulatorVector<T>::rebuildCellStore()
	{
		cellStore = std::vector<std::vector<std::vector<T>>>();
		clear();
	}

}