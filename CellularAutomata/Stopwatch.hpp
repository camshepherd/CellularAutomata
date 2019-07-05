#pragma once
#include <chrono>

/** Very simple stopwatch, based on std::chrono, to enable timing of operations. Taken from https://www.learncpp.com/cpp-tutorial/8-16-timing-your-code/
*/
class Stopwatch
{
protected:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_t> start_time;

public:
	Stopwatch() : start_time(clock_t::now())
	{
	}

	void reset()
	{
		start_time = clock_t::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_t>(clock_t::now() - start_time).count();
	}
};