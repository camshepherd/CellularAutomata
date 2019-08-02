namespace CellularAutomata {
	template <typename T>
	CUDA_FUNCTION ZonerArrayPixels<T>::ZonerArrayPixels(int *dims, int *maxDims, bool* A, bool* B) : dims(dims), maxDims(maxDims), cellActivities(A), rawActivities(B)
	{
		printf("ZONER ydim: %d, xdim: %d\n", dims[0], dims[1]);
		/*cellActivities = static_cast<bool*>(malloc(sizeof(bool) * ydim * xdim));
		if(cellActivities == NULL)
		{
			printf("IS NULL!\n");
		}*/
		/*rawActivities = static_cast<bool*>(malloc(sizeof(bool) * ydim * xdim));
		if(rawActivities == NULL)
		{
			printf("IS NULL\n");
		}*/
		for(int t = 0; t < dims[0] * dims[1]; ++t)
		{
			// printf("Assigning to %d\n", t);
			cellActivities[t] = true;
			rawActivities[t] = false;
		}
		if (rawActivities == NULL)
		{
			printf("IS NULL\n");
		}
		if (cellActivities == NULL)
		{
			printf("IS NULL!\n");
		}
		printf("\n*****COMPLETED CONSTRUCTOR\n");
	}

	template <typename T>
	CUDA_FUNCTION ZonerArrayPixels<T>::~ZonerArrayPixels()
	{
	}

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::updateDeadZones(T* frame1, T* frame2) {
		// get all cells that are different between the cells
		// mark all differing cell locations, and their neighbours, as being active

		int frameY = dims[0], frameX = dims[1];
		for(int t = 0; t < frameY * frameX; ++t)
		{
			rawActivities[t] = false;
		}

		for (int y = 0; y < frameY; ++y) {
			for (int x = 0; x < frameX; ++x) {
				rawActivities[y*frameX + x] = frame1[y*frameX +x] != frame2[y*frameX+x];
				cellActivities[y*frameX+x] = false;
			}
		}

		for (int y = 0; y < frameX; ++y) {
			for (int x = 0; x < frameX; ++x) {
				if (rawActivities[y*frameX+x] == true) {
					for (int ypos = y - 1; ypos <= y + 1; ++ypos) {
						for (int xpos = x - 1; xpos <= x + 1; ++xpos) {
							cellActivities[(((ypos + frameY) % frameY) * frameX) + ((xpos + frameX) % frameX)] = true;
						}
					}
				}
			}
		}
		return true;
	};

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::isLive(int y, int x) {
		return cellActivities[y*dims[1] + x];
	};

	template <typename T>
	CUDA_FUNCTION bool* ZonerArrayPixels<T>::getCellActivities() {
		return cellActivities;
	}

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::setDimensions(int* dims)
	{
		this->dims = dims;
		return true;
	}

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::refresh()
	{
		for (int t = 0; t < dims[0] * dims[1]; ++t)
		{
			// printf("Assigning to %d\n", t);
			cellActivities[t] = true;
			rawActivities[t] = false;
		}
		return true;
	}
}