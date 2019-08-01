namespace CellularAutomata {
	template <typename T>
	ZonerPixels<T>::ZonerPixels(int y, int x) : ydim(y), xdim(x)
	{
		cellActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, 1));
		rawActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, false));
	}

	template <typename T>
	ZonerPixels<T>::~ZonerPixels()
	{
	}

	template <typename T>
	bool ZonerPixels<T>::updateDeadZones(std::vector<std::vector<T>> frame1, std::vector<std::vector<T>> frame2) {
		// get all cells that are different between the cells
		// mark all differing cell locations, and their neighbours, as being active

		

		for (int y = 0; y < ydim; ++y) {
			for (int x = 0; x < xdim; ++x) {
				rawActivities[y][x] = frame1[y][x] != frame2[y][x];
				cellActivities[y][x] = false;
			}
		}

		for (int y = 0; y < ydim; ++y) {
			for (int x = 0; x < xdim; ++x) {
				if (rawActivities[y][x]) {
					for (int ypos = y - 1; ypos <= y + 1; ++ypos) {
						for (int xpos = x - 1; xpos <= x + 1; ++xpos) {
							cellActivities[(ypos + ydim) % ydim][(xpos + xdim) % xdim] = true;
						}
					}
				}
			}
		}
		return true;
	};

	template <typename T>
	bool ZonerPixels<T>::isLive(int y, int x) {
		return cellActivities[y][x];
	};

	template <typename T>
	std::vector<std::vector<bool>> ZonerPixels<T>::getCellActivities() {
		return cellActivities;
	}

	template <typename T>
	bool ZonerPixels<T>::setDimensions(int y, int x)
	{
		// only need to rebuild the vectors if the dimensions have changed
		if(y != ydim || x != xdim)
		{
			std::cout << "Updating ZonerPixels dimensions to " << y << " and " << x << std::endl;
			this->ydim = y;
			this->xdim = x;
			//delete cellActivities;
			cellActivities = std::vector<std::vector<bool>>(ydim, std::vector<bool>(xdim, 1));
			rawActivities = std::vector<std::vector<bool>>(ydim, std::vector<bool>(xdim, 1));
		}
		return true;
	}
}