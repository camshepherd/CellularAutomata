#pragma once

#include <vector>
#include <tuple>

/** Interface defining functionality for any class to generate split up a rectangular region into consituent, non-overlapping parts
*/
class ISegmenter
{
public:
	/** Get non-overlapping segments that cover the given surface area
	@param y_dim: The height of the region in the y direction
	@param x_dim: The width of the region in the x direction
	@param numSegments: The number of segments to divide the region up into
	*/
	virtual std::vector<std::tuple<int, int, int, int>> segment(int y_dim_, int x_dim_, int numSegments) const = 0;
};

