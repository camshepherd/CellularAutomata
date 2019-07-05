template <typename T>
ZonerPixels<T>::ZonerPixels(int y, int x) : ydim(y),xdim(x)
{
	cellActivities = std::vector<std::vector<bool>>(y, std::vector<bool>(x, 1));
}

template <typename T>
ZonerPixels<T>::~ZonerPixels()
{
}

template <typename T>
bool ZonerPixels<T>::updateDeadZones(std::vector<std::vector<T>> frame1, std::vector<std::vector<T>> frame2) {
	// get all cells that are different between the cells
	// mark all differing cell locations, and their neighbours, as being active

	std::vector<std::vector<bool>> rawActivities(ydim, std::vector<bool>(xdim, false));
	
	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			rawActivities[y][x] = frame1[y][x] != frame2[y][x];
			cellActivities[y][x] = false;
		}
	}

	for (int y = 0; y < ydim; ++y) {
		for (int x = 0; x < xdim; ++x) {
			if (rawActivities[y][x] == true) {
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