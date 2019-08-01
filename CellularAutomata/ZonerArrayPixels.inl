namespace CellularAutomata {
	template <typename T>
	CUDA_FUNCTION ZonerArrayPixels<T>::ZonerArrayPixels(int y, int x, bool* A, bool* B) : ydim(y), xdim(x), cellActivities(A), rawActivities(B)
	{
		printf("ZONER ydim: %d, xdim: %d\n", ydim, xdim);
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
		for(int t = 0; t < ydim * xdim; ++t)
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
	CUDA_FUNCTION bool ZonerArrayPixels<T>::updateDeadZones(T* frame1, T* frame2, int* dimensions) {
		// get all cells that are different between the cells
		// mark all differing cell locations, and their neighbours, as being active

		int frameX = dimensions[0], frameY = dimensions[1];
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

	template <typename T>
	CUDA_FUNCTION bool ZonerArrayPixels<T>::setDimensions(int y, int x)
	{
		// only reallocate if there is not enough memory available
		/*if(y*x < ydim * xdim)
		{
			this->ydim = y;
			this->xdim = x;
			bool* temp;
			temp = static_cast<bool*>(realloc(cellActivities, sizeof(bool) * ydim * xdim));
			if(temp != nullptr)
			{
				cellActivities = temp;
			}
			else
			{
				printf("AAAARRRRRRRRHHHHJJJJJJJJ");
			}
			temp = static_cast<bool*>(realloc(rawActivities, sizeof(bool) * ydim * xdim));
			if(temp != nullptr)
			{
				rawActivities = temp;
			}
			else
			{
				printf("WE'RE ALL GOING TO DIE!!!!");
			}
		}*/
		ydim = y;
		xdim = x;
		return true;
	}
}