#pragma once
#include "ISegmenter.h"

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

