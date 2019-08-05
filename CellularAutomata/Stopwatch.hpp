#pragma once
#include <chrono>

namespace CellularAutomata {
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
		/** Constructor 1. Default constructor; start the clock
		*/
		Stopwatch() : start_time(clock_t::now())
		{
		}

		/** Reset the stopwatch, starting the clock from 0 again
		*/
		void reset()
		{
			start_time = clock_t::now();
		}

		/** Get tiem elapsed since the clock was started in seconds
		*/
		double elapsed() const
		{
			return std::chrono::duration_cast<second_t>(clock_t::now() - start_time).count();
		}
	};
}