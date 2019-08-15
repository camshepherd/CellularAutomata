#pragma once
#include "ISegmenter.hpp"

namespace CellularAutomata {
	/** Segmenter which will split up the region keeping rows and columns intact and contiguous
	*/
	class SegmenterStrips :
		public ISegmenter
	{
	protected:
		bool orientation;

	public:
		/** Constructor 1. Create the segmenter to split on the specified orientation
		@param orientation: Whether rows(0) or columns(1) are split. Defaults to 0:
		*/
		SegmenterStrips(bool orientation = 0);

		/** Destructor 1. Default destructor
		*/
		~SegmenterStrips();

		/** Generate segments to cover the entire region
		@param y_dim: The size of the region in the y-axis
		@param x_dim: The size of the region in the x-axis
		@param numSegments: The number of segments to split the region up in to
		*/
		virtual std::vector<std::tuple<int, int, int, int>> segment(int y_dim_, int x_dim_, int numSegments) const override;

		/** Generate segments to cover the entire region, with the output as a C-style array. Each set of boundaries are stored contiguously, 
		in format: y_min, y_max, x_min, x_max. the first two positions in the array store the y and x dimensions respectively
		@param y_dim: The size of the region in the y-axis
		@param x_dim: The size of the region in the x-axis
		@param numSegments: The number of segments to split the region up in to
		*/
		virtual int* segmentToArray(int y_dim_, int x_dim_, int numSegments) const override;

		/** Set the orientation of the segments that will be created
		 */
		virtual bool setOrientation(bool newOrientation);
	};
}