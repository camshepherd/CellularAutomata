#ifndef ITESTER_H
#define ITESTER_H

#include "ISimulator.h"

// ITester is a simple interface to define the exposed functionality 
// required by the corresponding testing class for ISimulators
class ITester {
private:
	ISimulator& subject;
public:
	ITester(ISimulator& _subject) : subject(_subject) {};
	virtual bool test() = 0;
};

#endif