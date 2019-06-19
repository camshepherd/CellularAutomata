#pragma once
#include "ISegmenter.h"

//
// ISegmenter implementation to generate segments that use an entire row or column
// of the cell store
class SegmenterStrips :
	public ISegmenter
{
private:
	int orientation;

public:
	SegmenterStrips(int orientation = 0);
	~SegmenterStrips();

	virtual std::vector<std::tuple<int, int, int, int>> segment(int y_dim_, int x_dim_, int numSegments) const override;
};

