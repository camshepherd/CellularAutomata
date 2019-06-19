#include "SimulatorSequential.h"
#include "SimulatorCPU.h"

#include "RulesConway.h"
#include "RulesBML.h"
#include <iostream>


int main() {
	std::cout << "The system compiles!" << std::endl;
	
	RulesConway rules{};
	SimulatorCPU cpu{5, 3, rules};

	cpu.stepForward(4);
	getchar();
}