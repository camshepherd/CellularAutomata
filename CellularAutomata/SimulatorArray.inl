namespace CellularAutomata {
	template <typename T>
	SimulatorArray<T>::SimulatorArray(const int y, const int x, const IRulesArray<T>& _rules) : y_dim(y), x_dim(x), rules(_rules)
	{
		// Initialise the blank cellStore
		cellStore = std::vector<T*>();
		clear();
	}

	template <typename T>
	SimulatorArray<T>::~SimulatorArray()
	{
	}

	template <typename T>
	bool SimulatorArray<T>::clear(bool addBlankFirstFrame) {
		cellStore.clear();
		if (addBlankFirstFrame) {
			blankFrame();
		}
		return true;
	}

	template <typename T>
	int SimulatorArray<T>::getNumFrames() const {
		return cellStore.size();
	}

	template <typename T>
	bool SimulatorArray<T>::setCell(int y, int x, T new_val, int t) {
		t = (t == -1) ? getNumFrames() - 1 : t;
		if (t < 0 || t >= getNumFrames()) {
			throw std::runtime_error("The timestep t requested is outside of the valid range");
		}
		else if (!rules.isValid(new_val)) {
			throw std::runtime_error("The requested cell assignment is not a valid value");
		}
		else {
			cellStore[t][x + y*x_dim] = new_val;
			return true;
		}
	}

	template <typename T>
	bool SimulatorArray<T>::blankFrame() {
		cellStore.push_back(static_cast<T*>(malloc(sizeof(T) * y_dim * x_dim)));
		for (int k = 0; k < y_dim * x_dim; ++k) {
			cellStore.back()[k] = 0;
		}
		return true;
	}

	template <typename T>
	T SimulatorArray<T>::getCell(int y, int x, int t) const {
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
			return cellStore[t][x + y * x_dim];
		}
	}

	template <typename T>
	int SimulatorArray<T>::getYDim() {
		return y_dim;
	}

	template <typename T>
	int SimulatorArray<T>::getXDim() {
		return x_dim;
	}

	template <typename T>
	bool SimulatorArray<T>::copyFrame() {
		int numFrames = getNumFrames();
		cellStore.push_back(static_cast<T*>(malloc(sizeof(T) * y_dim * x_dim)));
		for (int k = 0; k < y_dim * x_dim; ++k) {
			cellStore[numFrames][k] = cellStore[numFrames - 1][k];
		}
		return true;
	}
}