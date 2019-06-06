#ifndef ISIMULATOR_H
#define ISIMULATOR_H

#include "Stopwatch.h"
//ISimulator is an interface defining the behaviour of the different 
// types of simulator. It defines basic functionality that is entirely 
// separated from the underlying implementation details

class ISimulator {
protected:
	Stopwatch timer;
public:
	ISimulator() {};

	virtual bool stepForward(int steps = 1) = 0;
	virtual bool stepForward(double seconds) = 0;
	virtual bool blankFrame() = 0;

	virtual int getCell(int y, int x, int t = -1) = 0;
	
	virtual bool setCell(int y, int x, int new_val, int t = -1) = 0;

	virtual int getNumFrames() = 0;
	virtual bool clear(bool addBlankFirstFrame=true) = 0;

	
};

#endif