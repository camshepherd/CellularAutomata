#ifndef ISIMULATOR_H
#define ISIMULATOR_H

#include "Stopwatch.h"
#include <string>
#include <fstream>

// Interface defining standard behaviour of simulators, able to generically simulate 
// synchronous cellular automata

class ISimulator {
protected:
	Stopwatch timer;
	double elapsedTime=0;
public:
	ISimulator() { elapsedTime = 0; };

	virtual double stepForward(int steps = 1) = 0;
	double stepForwardTime(double seconds);
	virtual bool blankFrame() = 0;
	virtual bool copyFrame() = 0;

	virtual int getCell(int y, int x, int t = -1) const = 0;
	virtual int getNumFrames() const = 0;
	virtual int getYDim() = 0;
	virtual int getXDim() = 0;
	virtual int getMaxValidState() = 0;

	virtual bool setCell(int y, int x, int new_val, int t = -1) = 0;

	
	virtual bool clear(bool addBlankFirstFrame=true) = 0;
	
	bool writeData(std::string filename);
};

#endif