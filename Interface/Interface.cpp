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
ISimulator<int>* simInt;
ISimulator<long>* simLong;
ISimulator<long long>* simLongLong;

IRules<bool>* rulesBool;
IRules<int>* rulesInt;
IRules<long>* rulesLong;
IRules<long long>* rulesLongLong;

IRulesArray<bool>* rulesArrayBool;
IRulesArray<int>* rulesArrayInt;
IRulesArray<long>* rulesArrayLong;
IRulesArray<long long>* rulesArrayLongLong;


ISegmenter* segmenter;

IDeadZoneHandler<bool>* zonerBool;
IDeadZoneHandler<int>* zonerInt;
IDeadZoneHandler<long>* zonerLong;
IDeadZoneHandler<long long>* zonerLongLong;


void printInsult() {
	int num = std::rand() / ((RAND_MAX + 1u) / 9);
	switch (num) {
	case 0:
		cout << "Your mother was a hamster and your father smelt of elderberries!" << endl;
		break;
	case 1:
		cout << "I fart in your general direction!" << endl;
		break;
	case 2:
		cout << "You great supine protoplasmic invertebrate jelly!" << endl;
		break;
	case 3:
		cout << "I don't want to talk to you no more, you empty headed animal food trough wiper!" << endl;		
		break;
	case 4:
		cout << "I am drunk today, and tomorrow I shall be sober but you will still be ugly!" << endl;
		break;
	case 5:
		cout << "English Pig Dog!" << endl;
		break;
	case 6:
		cout << "Go and boil your bottom you son of a silly person!" << endl;
		break;
	case 7:
		cout << "I blow my nose at you!" << endl;
		break;
	case 8:
		cout << "Go away or I shall taunt you a second time!" << endl;
		break;
	}
}


void printHelp(int angriness) {
	cout << "\n\n|______________________" << endl;
	cout << std::uppercase << "Help Page/Manual: " << endl;
	cout << "Note that all functionality is very primitive and will assume that you know what you're doing" << endl;
	cout << "print [frameNum] [numToPrint]" << endl;
	cout << "build simulatorType ruleSet [datatype=int] [ydim xdim nSegments nBlocks nThreads ymax xmax]: build simulator with given parameters, not all are needed for all implementations" << endl;
	cout << "dimensions ydim xdim: change dimensions of simulator" << endl;
	cout << "step [steps=1]: step forward through the simulation by the given number of frames" << endl;
	cout << "clear: empties the current simulator" << endl;
	cout << "set y x val: set position [y,x] to value val" << endl;
	cout << "exit: exit the application in a way that doesn't involve crashing" << endl;
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
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nWell if you need some help with the help, maybe try an insult instead...\n\nyou deserve it!" << endl;
			}
		}
		else if (words.size() > 1 && words[1] == "1") {
			printHelp(1);
		}
		else {
			printHelp(0);
		}
	}
	else if (words[0] == "insult") {
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\nIt prints an insult.\n\nWhat would you expect it to do?\n" << endl;
			}
		}
		else {
			printInsult();
		}
	}
	else if (words[0] == "build") {
		int ydim, xdim;
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\n Build the given simulation\n\n Simulator inputs: \nseq\nseqzon\ncpuhor\ncpuver\ncpuzonhor\ncpuzonver\ngpuhor\ngpuver\ngpuzonhor\ngpuzonver\n\n Data types: \nbool(BML only)\nshort\nint\nlong\nlonglong\n\nRulesets: \n\nconway/con/gol\nbml\n\n Arguments should all be space separated and are case-sensitive" << endl;
			}
		}
		else {
			// clean up the stored data
			if (rulesBool != null) {
				delete(rulesBool);
			}
			// build simulatorType ruleSet [datatype=int] [ydim xdim nSegments nBlocks nThreads ymax xmax]
			if (words.size() > 1 && words[1] == "seq") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorSequential<bool>{ ydim, xdim, *rulesBool };
						cout << "Created\n";
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}

				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
						rulesInt = new RulesConway<int>{};
					}
					else {
						rulesInt = new RulesBML<int>{};
					}
					if (words.size() > 5) {
						ydim = stoi(words[4]);
						xdim = stoi(words[5]);
					}
					simType = 1;
					try {
						simInt = new SimulatorSequential<int>{ ydim, xdim, *rulesInt };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
						rulesLong = new RulesConway<long>{};
					}
					else {
						rulesLong = new RulesBML<long>{};
					}
					if (words.size() > 5) {
						ydim = stoi(words[4]);
						xdim = stoi(words[5]);
					}
					simType = 2;
					try {
						simLong = new SimulatorSequential<long>{ ydim, xdim, *rulesLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
						rulesLongLong = new RulesConway<long long>{};
					}
					else {
						rulesLongLong = new RulesBML<long long>{};
					}
					if (words.size() > 5) {
						ydim = stoi(words[4]);
						xdim = stoi(words[5]);
					}
					simType = 3;
					try {

						simLongLong = new SimulatorSequential<long long>{ ydim, xdim, *rulesLongLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "seqzon") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {

						simBool = new SimulatorSequentialZoning<bool>{ ydim, xdim, *rulesBool,*zonerBool };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 1;
					try {

						simInt = new SimulatorSequentialZoning<int>{ ydim, xdim, *rulesInt,*zonerInt };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 2;
					try {
						simLong = new SimulatorSequentialZoning<long>{ ydim, xdim, *rulesLong,*zonerLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 3;
					try {
						simLongLong = new SimulatorSequentialZoning<long long>{ ydim, xdim, *rulesLongLong,*zonerLongLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "cpuhor") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorCPU<bool>{ ydim, xdim, *rulesBool,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 1;
					try {
						simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 2;
					try {
						simLong = new SimulatorCPU<long>{ ydim, xdim, *rulesLong,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 3;
					try {
						simLongLong = new SimulatorCPU<long long>{ ydim, xdim, *rulesLongLong,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "cpuver") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorCPU<bool>{ ydim, xdim, *rulesBool,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 1;
					try {
						simInt = new SimulatorCPU<int>{ ydim, xdim, *rulesInt,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 2;
					try {
						simLong = new SimulatorCPU<long>{ ydim, xdim, *rulesLong,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 3;
					try {
						simLongLong = new SimulatorCPU<long long>{ ydim, xdim, *rulesLongLong,*segmenter };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "cpuhorzon") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorCPUZoning<bool>{ ydim, xdim, *rulesBool,*segmenter,*zonerBool };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 1;
					try {
						simInt = new SimulatorCPUZoning<int>{ ydim, xdim, *rulesInt,*segmenter,*zonerInt };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 2;
					try {
						simLong = new SimulatorCPUZoning<long>{ ydim, xdim, *rulesLong,*segmenter,*zonerLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 3;
					try {
						simLongLong = new SimulatorCPUZoning<long long>{ ydim, xdim, *rulesLongLong,*segmenter,*zonerLongLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "cpuverzon") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorCPUZoning<bool>{ ydim, xdim, *rulesBool,*segmenter,*zonerBool };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 1;
					try {
						simInt = new SimulatorCPUZoning<int>{ ydim, xdim, *rulesInt,*segmenter,*zonerInt };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 2;
					try {
						simLong = new SimulatorCPUZoning<long>{ ydim, xdim, *rulesLong,*segmenter,*zonerLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					simType = 3;
					try {
						simLongLong = new SimulatorCPUZoning<long long>{ ydim, xdim, *rulesLongLong,*segmenter,*zonerLongLong };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "gpuhor") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorGPU<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 1;
					try {
						simInt = new SimulatorGPU<int>{ ydim, xdim, *rulesArrayInt,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 2;
					try {
						simLong = new SimulatorGPU<long>{ ydim, xdim, *rulesArrayLong,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 3;
					try {
						simLongLong = new SimulatorGPU<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "gpuver") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorGPU<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 1;
					try {
						simInt = new SimulatorGPU<int>{ ydim, xdim, *rulesArrayInt,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 2;
					try {
						simLong = new SimulatorGPU<long>{ ydim, xdim, *rulesArrayLong,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 3;
					try {
						simLongLong = new SimulatorGPU<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,2,32 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
			}
			else if (words[1] == "gpuhorzon") {
				if (words.size() > 3 && words[3] == "bool") {
					if (words[2] == "conway" || words[2] == "gol") {
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
					try {
						simBool = new SimulatorGPUZoning<bool>{ ydim, xdim, *rulesArrayBool,*segmenter,2,32,3000,3000 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "int") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 1;
					try {
						simInt = new SimulatorGPUZoning<int>{ ydim, xdim, *rulesArrayInt,*segmenter,2,32,3000,3000 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "long") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 2;
					try {
						simLong = new SimulatorGPUZoning<long>{ ydim, xdim, *rulesArrayLong,*segmenter,2,32,3000,3000 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else if (words.size() > 3 && words[3] == "longlong") {
					if (words[2] == "conway" || words[2] == "gol") {
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

					simType = 3;
					try {
						simLongLong = new SimulatorGPUZoning<long long>{ ydim, xdim, *rulesArrayLongLong,*segmenter,2,32,3000,3000 };
					}
					catch (std::exception e) {
						cout << "That didn't work" << endl;
					}
					cout << "Created\n";
				}
				else {
					cout << "That doesn't work" << endl;
				}
			}
		}
	}
	else if (words[0] == "print") {
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nPrint out frameCount frames starting from (and including) frameStart one after the other.\n The MOST RECENTLY CREATED simulator will be accessed. Unless otherwise specified only a single frame will be printed, and unless otherwise specified the final frame (most recently simulated) will be printed" << endl;
			}
		}
		// print [frameNum] [numToPrint]
		int frameStart = -1, frameCount = 1;
		if (words.size() > 1) {
			try {
				frameStart = stoi(words[1]);
			}
			catch (std::exception e) {
				cout << "that wasn't a number" << endl;
			}
		}
		if (words.size() > 2) {
			try {
				frameCount = stoi(words[2]);
			}
			catch (std::exception e) {
				cout << "that wasn't a number" << endl;
			}
		}
		if (simType == 0) {
			try {
				printFrames<bool>(simBool, frameStart, frameCount);
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
		else if (simType == 1) {
			try {
				printFrames<int>(simInt, frameStart, frameCount);
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
		else if (simType == 2) {
			try {
				printFrames<long>(simLong, frameStart, frameCount);
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
		else if (simType == 3) {
			try {
				printFrames<long long>(simLongLong, frameStart, frameCount);
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
		else {
			cout << "Something has gone very wrong with the state of this program! This should not have been able to happen" << endl;
		}
	}
	else if (words[0] == "step") {
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nStep forward the current simulator (most recently created) by the number of frames specified. If not specified then the simulator will only step forward a single frame\n\n" << endl;
			}
		}
		else {
			try {
				//step[steps = 1]
				int steps = 1;
				if (words.size() > 1) {
					steps = stoi(words[1]);
				}
				if (simType == 0) {
					simBool->stepForward(steps);
				}
				else if (simType == 1) {
					simInt->stepForward(steps);
				}
				else if (simType == 2) {
					simLong->stepForward(steps);
				}
				else if (simType == 3) {
					simLongLong->stepForward(steps);
				}
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
	}
	else if (words[0] == "clear") {
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nEmpty the current simulator" << endl;
			}
		}
		else {

			try {
				if (simType == 0) {
					simBool->clear();
				}
				else if (simType == 1) {
					simInt->clear();
				}
				else if (simType == 2) {
					simLong->clear();
				}
				else if (simType == 3) {
					simLongLong->clear();
				}
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
	}
	else if (words[0] == "dims") {

		// dimensions ydim xdim
		int ydim = stoi(words[1]);
		int xdim = stoi(words[2]);
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nChange the dimensions of the active simulator to match the given ones: y x\n\n";
			}
		}
		else {
			try {
				if (simType == 0) {
					simBool->setDimensions(ydim, xdim);
				}
				else if (simType == 1) {
					simInt->setDimensions(ydim, xdim);
				}
				else if (simType == 2) {
					simLong->setDimensions(ydim, xdim);
				}
				else if (simType == 3) {
					simLongLong->setDimensions(ydim, xdim);
				}
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
	}
	else if (words[0] == "set") {
		int y = stoi(words[1]);
		int x = stoi(words[2]);
		int val = stoi(words[3]);
		if (words.size() > 1) {
			if (words[1] == "/?") {
				cout << "\n\nSet the specified cell of the simulation to the given value " << endl;
			}
		}
		else {
			try {
				if (simType == 0) {
					simBool->setCell(y, x, val);
				}
				else if (simType == 1) {
					simInt->setCell(y, x, val);
				}
				else if (simType == 2) {
					simLong->setCell(y, x, val);
				}
				else if (simType == 3) {
					simLongLong->setCell(y, x, val);
				}
			}
			catch (std::exception e) {
				cout << "Something went wrong" << endl;
			}
		}
	}
	else if (words[0] == "exit") {
		if (words.size() > 1 && words[1] == "/?") {
			cout << "\n\nIt exits the program..." << endl;
		}
		else {
			cout << "\nI'll be back...\n" << endl;
			running = false;
		}
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
