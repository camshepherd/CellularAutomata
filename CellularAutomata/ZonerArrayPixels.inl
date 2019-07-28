namespace CellularAutomata {
	template <typename T>
	CUDA_FUNCTION ZonerArrayPixels<T>::ZonerArrayPixels(int y, int x) : ydim(y), xdim(x)
	{
		cellActivities = static_cast<bool*>(malloc(sizeof(bool) * ydim * xdim));
		rawActivities = static_cast<bool*>(malloc(sizeof(bool) * ydim * xdim));
		for(int t = 0; t < ydim * xdim; ++t)
		{
			cellActivities[t] = true, rawActivities[t] = false;
		}
	}

	template <typename T>
	CUDA_FUNCTION ZonerArrayPixels<T>::~ZonerArrayPixels()
	{
	}

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::updateDeadZones(T* frame1, T* frame2) {
		// get all cells that are different between the cells
		// mark all differing cell locations, and their neighbours, as being active

	
		for(int t = 0; t < ydim * xdim; ++t)
		{
			rawActivities[t] = false;
		}

		for (int y = 0; y < ydim; ++y) {
			for (int x = 0; x < xdim; ++x) {
				rawActivities[y*xdim + x] = frame1[y*xdim +x] != frame2[y*xdim+x];
				cellActivities[y*xdim+x] = false;
			}
		}

		for (int y = 0; y < ydim; ++y) {
			for (int x = 0; x < xdim; ++x) {
				if (rawActivities[y*xdim+x] == true) {
					for (int ypos = y - 1; ypos <= y + 1; ++ypos) {
						for (int xpos = x - 1; xpos <= x + 1; ++xpos) {
							cellActivities[(((ypos + ydim) % ydim) * xdim) + ((xpos + xdim) % xdim)] = true;
						}
					}
				}
			}
		}
		return true;
	};

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::isLive(int y, int x) {
		return cellActivities[y*xdim + x];
	};

	template <typename T>
	CUDA_FUNCTION bool* ZonerArrayPixels<T>::getCellActivities() {
		return cellActivities;
	}
}