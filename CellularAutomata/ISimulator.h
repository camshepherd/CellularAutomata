#ifndef ISIMULATOR_H
#define ISIMULATOR_H

#include "Stopwatch.h"

// Interface defining standard behaviour of simulators, able to generically simulate 
// synchronous cellular automata

class ISimulator {
protected:
	Stopwatch timer;
public:
	ISimulator() {};

	virtual bool stepForward(int steps = 1) = 0;
	virtual bool stepForwardTime(double seconds) = 0;
	virtual bool blankFrame() = 0;

	virtual int getCell(int y, int x, int t = -1) const = 0;
	
	virtual bool setCell(int y, int x, int new_val, int t = -1) = 0;

	virtual int getNumFrames() const= 0;
	virtual bool clear(bool addBlankFirstFrame=true) = 0;
};

#endif