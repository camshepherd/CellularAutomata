// Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <time.h>

using namespace std;
using namespace CellularAutomata;

bool running = true;
int simType = 0;

bool *A, *B;
int dims[2] = { 1000,1000 };
int maxDims[2] = { 10000,10000 };

ISimulator<bool>* simBool;
ISimulator<short>* simShort;
ISimulator<int>* simInt;
ISimulator<long>* simLong;
ISimulator<long long>* simLongLong;

IRules<bool>* rulesBool;
IRules<short>* rulesShort;
IRules<int>* rulesInt;
IRules<long>* rulesLong;
IRules<long long>* rulesLongLong;

IRulesArray<bool>* rulesArrayBool;
IRulesArray<short>* rulesArrayShort;
IRulesArray<int>* rulesArrayInt;
IRulesArray<long>* rulesArrayLong;
IRulesArray<long long>* rulesArrayLongLong;


ISegmenter* segmenter;

IDeadZoneHandler<bool>* zonerBool;
IDeadZoneHandler<short>* zonerShort;
IDeadZoneHandler<int>* zonerInt;
IDeadZoneHandler<long>* zonerLong;
IDeadZoneHandler<long long>* zonerLongLong;



void printHelp(int angriness) {
	cout << "\n\n|______________________" << endl;
	cout << std::uppercase << "Help Page/Manual: " << endl;
	cout << "Note that all functionality is very primitive and will assume that you know what you're doing and don't make mistakes" << endl;
	cout << "To get more information about a command, type [command] /?" << endl;
	cout << "print [frameNum] [numToPrint]" << endl;
	cout << "build simulatorType ruleSet datatype ydim xdim: build simulator with given parameters" << endl;
	cout << "dimensions ydim xdim: change dimensions of simulator" << endl;
	cout << "step [steps=1]: step forward through the simulation by the given number of frames" << endl;
	cout << "clear: empties the current simulator" << endl;
	cout << "set y x val: set position [y,x] to value val" << endl;
	cout << "exit: exit the application in a way that doesn't involve crashing, hopefully" << endl;
	cout << "|__________________________ END\n\n";
}

template <typename T>
void printFrames(ISimulator<T> *sim, int frameStart, int frameCount) {
	for (int t = 0; t < frameCount; ++t) {
		if (frameStart != -1) {
			sim->printFrame(frameStart + t);
		}
		else {
			sim->printFrame();
		}
	}
}


void handleInput(string line) {
	if (line.length() == 0) {
		return;
	}
	vector<string> words{};
	stringstream stream(line);
	for (string word; stream >> word;) {
		words.push_back(word);
	}
	if (words[0] == "help") {
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nThis prints out a summary of the commands and their core functionalities\n\n" << endl;
		}
		else {
			if (words.size() > 1 && words[1] == "1") {
				printHelp(1);
			}
			else {
				printHelp(0);
			}
		}
	}
	else if (words[0] == "build") {
		int ydim, xdim;
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\n Build the simulator.\n\nSimulator types:\nseq\nseqzon\ncpuhor\ncpuver\ncpuhorzon\ncpuverzon\ngpuhor\ngpuver\n\nRulesets:\ncon/gol/conway\nbml\n\nALl arguments must be space-separated\n\n" << endl;
		}
		else {
			try {
				delete rulesBool;
				delete rulesShort;
				delete rulesInt;
				delete rulesLong;
				delete rulesLongLong;

				delete simBool;
				delete simShort;
				delete simInt;
				delete simLong;
				delete simLongLong;

				delete rulesArrayBool;
				delete rulesArrayShort;
				delete rulesArrayInt;
				delete rulesArrayLong;
				delete rulesArrayLongLong;

				delete segmenter;
				// build simulatorType ruleSet [datatype=int] [ydim xdim nSegments nBlocks nThreads ymax xmax]
				if (words[1] == "seq") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						simType = 0;
						simBool = new SimulatorSequential<bool>{ ydim, xdim, *rulesBool };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						simType = 1;
						simShort = new SimulatorSequential<short>{ ydim, xdim, *rulesShort };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						simType = 2;
						simInt = new SimulatorSequential<int>{ ydim, xdim, *rulesInt };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						simType = 3;
						simLong = new SimulatorSequential<long>{ ydim, xdim, *rulesLong };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						simType = 4;
						simLongLong = new SimulatorSequential<long long>{ ydim, xdim, *rulesLongLong };
						cout << "Created\n";
					}
				}
				else if (words[1] == "seqzon") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						zonerBool = new ZonerPixels<bool>{ ydim,xdim };
						simType = 0;
						simBool = new SimulatorSequentialZoning<bool>{ ydim, xdim, *rulesBool,*zonerBool };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						zonerShort = new ZonerPixels<short>{ ydim,xdim };
						simType = 1;
						simShort = new SimulatorSequentialZoning<short>{ ydim, xdim, *rulesShort,*zonerShort };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						zonerInt = new ZonerPixels<int>{ ydim,xdim };
						simType = 2;
						simInt = new SimulatorSequentialZoning<int>{ ydim, xdim, *rulesInt,*zonerInt };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						zonerLong = new ZonerPixels<long>{ ydim,xdim };
						simType = 3;
						simLong = new SimulatorSequentialZoning<long>{ ydim, xdim, *rulesLong,*zonerLong };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						zonerLongLong = new ZonerPixels<long long>{ ydim,xdim };
						simType = 4;
						simLongLong = new SimulatorSequentialZoning<long long>{ ydim, xdim, *rulesLongLong,*zonerLongLong };
						cout << "Created\n";
					}
				}
				else if (words[1] == "cpuhor") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only availalable for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 0;
						simBool = new SimulatorCPU<bool>{ ydim, xdim, *rulesBool,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 1;
						simShort = new SimulatorCPU<short>{ ydim, xdim, *rulesShort,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 2;
						simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 3;
						simLong = new SimulatorCPU<long>{ ydim, xdim, *rulesLong,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 4;
						simLongLong = new SimulatorCPU<long long>{ ydim, xdim, *rulesLongLong,*segmenter };
						cout << "Created\n";
					}
				}
				else if (words[1] == "cpuver") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 0;
						simBool = new SimulatorCPU<bool>{ ydim, xdim, *rulesBool,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 1;
						simShort = new SimulatorCPU<short>{ ydim, xdim, *rulesShort,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 2;
						simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 3;
						simLong = new SimulatorCPU<long>{ ydim, xdim, *rulesLong,*segmenter };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 4;
						simLongLong = new SimulatorCPU<long long>{ ydim, xdim, *rulesLongLong,*segmenter };
						cout << "Created\n";
					}
				}
				else if (words[1] == "cpuhorzon") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						zonerBool = new ZonerPixels<bool>{ ydim,xdim };
						simType = 0;
						simBool = new SimulatorCPUZoning<bool>{ ydim, xdim, *rulesBool,*segmenter,*zonerBool };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						zonerShort = new ZonerPixels<short>{ ydim,xdim };
						simType = 1;
						simShort = new SimulatorCPUZoning<short>{ ydim, xdim, *rulesShort,*segmenter,*zonerShort };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						zonerInt = new ZonerPixels<int>{ ydim,xdim };
						simType = 2;
						simInt = new SimulatorCPUZoning<int>{ ydim, xdim, *rulesInt,*segmenter,*zonerInt };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						zonerLong = new ZonerPixels<long>{ ydim,xdim };
						simType = 3;
						simLong = new SimulatorCPUZoning<long>{ ydim, xdim, *rulesLong,*segmenter,*zonerLong };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						zonerLongLong = new ZonerPixels<long long>{ ydim,xdim };
						simType = 4;
						simLongLong = new SimulatorCPUZoning<long long>{ ydim, xdim, *rulesLongLong,*segmenter,*zonerLongLong };
						cout << "Created\n";
					}
				}
				else if (words[1] == "cpuverzon") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesBool = new RulesConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						zonerBool = new ZonerPixels<bool>{ ydim,xdim };
						simType = 0;
						simBool = new SimulatorCPUZoning<bool>{ ydim, xdim, *rulesBool,*segmenter,*zonerBool };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesShort = new RulesConway<short>{};
						}
						else {
							rulesShort = new RulesBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						zonerShort = new ZonerPixels<short>{ ydim,xdim };
						simType = 1;
						simShort = new SimulatorCPUZoning<short>{ ydim, xdim, *rulesShort,*segmenter,*zonerShort };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesInt = new RulesConway<int>{};
						}
						else {
							rulesInt = new RulesBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						zonerInt = new ZonerPixels<int>{ ydim,xdim };
						simType = 2;
						simInt = new SimulatorCPUZoning<int>{ ydim, xdim, *rulesInt,*segmenter,*zonerInt };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLong = new RulesConway<long>{};
						}
						else {
							rulesLong = new RulesBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						zonerLong = new ZonerPixels<long>{ ydim,xdim };
						simType = 3;
						simLong = new SimulatorCPUZoning<long>{ ydim, xdim, *rulesLong,*segmenter,*zonerLong };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesLongLong = new RulesConway<long long>{};
						}
						else {
							rulesLongLong = new RulesBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						zonerLongLong = new ZonerPixels<long long>{ ydim,xdim };
						simType = 4;
						simLongLong = new SimulatorCPUZoning<long long>{ ydim, xdim, *rulesLongLong,*segmenter,*zonerLongLong };
						cout << "Created\n";
					}
				}
				else if (words[1] == "gpuhor") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayBool = new RulesArrayConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 0;
						simBool = new SimulatorGPU<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayShort = new RulesArrayConway<short>{};
						}
						else {
							rulesArrayShort = new RulesArrayBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 1;
						simShort = new SimulatorGPU<short>{ ydim, xdim, *rulesArrayShort,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayInt = new RulesArrayConway<int>{};
						}
						else {
							rulesArrayInt = new RulesArrayBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 2;
						simInt = new SimulatorGPU<int>{ ydim, xdim, *rulesArrayInt,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLong = new RulesArrayConway<long>{};
						}
						else {
							rulesArrayLong = new RulesArrayBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 3;
						simLong = new SimulatorGPU<long>{ ydim, xdim, *rulesArrayLong,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLongLong = new RulesArrayConway<long long>{};
						}
						else {
							rulesArrayLongLong = new RulesArrayBML<long long >{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 4;
						simLongLong = new SimulatorGPU<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,128,256 };
						cout << "Created\n";
					}
				}
				else if (words[1] == "gpuver") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayBool = new RulesArrayConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 0;
						simBool = new SimulatorGPU<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayShort = new RulesArrayConway<short>{};
						}
						else {
							rulesArrayShort = new RulesArrayBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 1;
						simShort = new SimulatorGPU<short>{ ydim, xdim, *rulesArrayShort,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayInt = new RulesArrayConway<int>{};
						}
						else {
							rulesArrayInt = new RulesArrayBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 2;
						simInt = new SimulatorGPU<int>{ ydim, xdim, *rulesArrayInt,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLong = new RulesArrayConway<long>{};
						}
						else {
							rulesArrayLong = new RulesArrayBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 3;
						simLong = new SimulatorGPU<long>{ ydim, xdim, *rulesArrayLong,*segmenter,128,256 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLongLong = new RulesArrayConway<long long>{};
						}
						else {
							rulesArrayLongLong = new RulesArrayBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 1 };

						simType = 4;
						simLongLong = new SimulatorGPU<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,128,256 };
						cout << "Created\n";
					}
				}
				else if (words[1] == "gpuhorzon") {
					if (words.size() > 3 && words[3] == "bool") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayBool = new RulesArrayConway<bool>{};
						}
						else {
							cout << "Only available for Game of Life" << endl;
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 0;
						simBool = new SimulatorGPUZoning<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,2,32,3000,3000 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "short") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayShort = new RulesArrayConway<short>{};
						}
						else {
							rulesArrayShort = new RulesArrayBML<short>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 1;
						simShort = new SimulatorGPUZoning<short>{ ydim, xdim, *rulesArrayShort,*segmenter,2,32,3000,3000 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "int") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayInt = new RulesArrayConway<int>{};
						}
						else {
							rulesArrayInt = new RulesArrayBML<int>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 2;
						simInt = new SimulatorGPUZoning<int>{ ydim, xdim, *rulesArrayInt,*segmenter,2,32,3000,3000 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "long") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLong = new RulesArrayConway<long>{};
						}
						else {
							rulesArrayLong = new RulesArrayBML<long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 3;
						simLong = new SimulatorGPUZoning<long>{ ydim, xdim, *rulesArrayLong,*segmenter,2,32,3000,3000 };
						cout << "Created\n";
					}
					else if (words.size() > 3 && words[3] == "longlong") {
						if (words[2] == "conway" || words[2] == "gol" || words[2] == "con") {
							rulesArrayLongLong = new RulesArrayConway<long long>{};
						}
						else {
							rulesArrayLongLong = new RulesArrayBML<long long>{};
						}
						if (words.size() > 5) {
							ydim = stoi(words[4]);
							xdim = stoi(words[5]);
						}
						segmenter = new SegmenterStrips{ 0 };

						simType = 4;
						simLongLong = new SimulatorGPUZoning<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,2,32,3000,3000 };
						cout << "Created\n";
					}
				}
			}
			catch (std::exception e) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "print") {
		// print [frameNum] [numToPrint]
		int frameStart = -1, frameCount = 1;
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nPrint out, starting from frameNum, numToPrint frames one after the other. \nUnless otherwise specified, only one frame will be printed, and unless otherwise specified the start frame is the highest-indexed\n\n" << endl;
		}
		else {
			try {
				if (words.size() > 1) {
					frameStart = stoi(words[1]);
				}
				if (words.size() > 2) {
					frameCount = stoi(words[2]);
				}
				if (simType == 0) {
					printFrames<bool>(simBool, frameStart, frameCount);
				}
				else if (simType == 1) {
					printFrames<short>(simShort, frameStart, frameCount);
				}
				else if (simType == 2)
				{
					printFrames<int>(simInt, frameStart, frameCount);
				}
				else if (simType == 3) {
					printFrames<long>(simLong, frameStart, frameCount);
				}
				else if (simType == 4) {
					printFrames<long long>(simLongLong, frameStart, frameCount);
				}
			}
			catch (exception e) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "step") {
		//step[steps = 1]
		int steps = 1;
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nSimulate successive frames of the active simulation\n\n" << endl;
		}
		else {
			try {
				if (words.size() > 1) {
					steps = stoi(words[1]);
				}
				if (simType == 0) {
					cout << "Took " << simBool->stepForward(steps) << " seconds\n";
				}
				else if (simType == 1) {
					cout << "Took " << simShort->stepForward(steps) << " seconds\n";
				}
				else if (simType == 2) {
					cout << "Took " << simInt->stepForward(steps) << " seconds\n";
				}
				else if (simType == 3) {
					cout << "Took " << simLong->stepForward(steps) << " seconds\n";
				}
				else if (simType == 4) {
					cout << "Took " << simLongLong->stepForward(steps) << " seconds\n";
				}
			}
			catch (...) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "clear") {
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nEmpty the simulated frames of the active simulation\n\n" << endl;
		}
		else {
			try {
				if (simType == 0) {
					simBool->clear();
				}
				else if (simType == 1) {
					simShort->clear();
				}
				else if (simType == 2) {
					simInt->clear();
				}
				else if (simType == 3) {
					simLong->clear();
				}
				else if (simType == 4) {
					simLongLong->clear();
				}
				cout << "\nDone\n";
			}
			catch (exception e) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "dims") {
		// dimensions ydim xdim
		int ydim = stoi(words[1]);
		int xdim = stoi(words[2]);
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nChange the dimensions of the active simulation, necessarily losing all simulated frames in the process\n\n" << endl;
		}
		else {
			try {
				if (simType == 0) {
					simBool->setDimensions(ydim, xdim);
				}
				else if (simType == 1) {
					simShort->setDimensions(ydim, xdim);
				}
				else if (simType == 2)
				{
					simInt->setDimensions(ydim, xdim);
				}
				else if (simType == 3) {
					simLong->setDimensions(ydim, xdim);
				}
				else if (simType == 4) {
					simLongLong->setDimensions(ydim, xdim);
				}
			}
			catch (exception e) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "set") {
		int y = stoi(words[1]);
		int x = stoi(words[2]);
		int val = stoi(words[3]);
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nSet the state of the specified cell in the active simulation\n\n" << endl;
		}
		else {
			try {
				if (simType == 0) {
					simBool->setCell(y, x, val);
				}
				else if (simType == 1) {
					simShort->setCell(y, x, val);
				}
				else if (simType == 2) {
					simInt->setCell(y, x, val);
				}
				else if (simType == 3) {
					simLong->setCell(y, x, val);
				}
				else if (simType == 4) {
					simLongLong->setCell(y, x, val);
				}
			}
			catch (exception e) {
				cout << "That didn't work" << endl;
			}
		}
	}
	else if (words[0] == "exit") {
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nIt exits the program. What else?...\n\n" << endl;
		}
		else {
			cout << "\nI'll be back...\n" << endl;
			running = false;
		}
	}
	else {
		cout << "No command recognised" << endl;
	}
}




int main()
{
	srand(time(nullptr)); // use current time as seed for random generator
	cout << "Welcome to the Cellular Automata Simulator Program!" << endl;
	cout << "type 'help' to see the available commands" << endl;
	string line;
	while (running) {
		getline(cin, line);
		handleInput(line);
	}
	cout << "\n\n|_______ENDING PROGRAM________|\n";
	return 0;
}
