#ifndef PERF_COUNTERS_HPP
#define PERF_COUNTERS_HPP

#include <ctime>
#include <string>

/**
 * Hardware performance counters (cycles, instructions, cache misses) read through Linux
 * perf_event_open, so a benchmark can report why it is fast or slow and not only how fast.
 *
 * Counting is inherited by threads created after construction, so construct this before a
 * Threadpool to include its workers. Threads that already exist (for example OpenBLAS's
 * workers, created when the library loads) are not counted.
 *
 * Events that the CPU, kernel or hypervisor does not provide are skipped. The L2, L3 and DRAM
 * events use AMD encodings and are opened only on AMD Zen 3 or newer.
 *
 * It also counts the process's page faults between start() and stop(). That count comes from the
 * kernel's own bookkeeping, not from a hardware counter, so it covers every thread and is always
 * available.
 */
class PerfCounters {
public:
    /**
     * Counted events:
     * - Cycles: core clock cycles while running.
     * - Instructions: retired instructions.
     * - L1DMisses: L1 data cache read misses.
     * - L2Misses: requests that missed L2 (on AMD this includes instruction fetches).
     * - L3Fills: L1D fills, demand or prefetch, served by the L3.
     * - DramFills: L1D fills, demand or prefetch, served by DRAM.
     */
    enum Event { Cycles, Instructions, L1DMisses, L2Misses, L3Fills, DramFills, NumEvents };

    /**
     * Opens every supported event, stopped. Never throws; unsupported events are skipped.
     */
    PerfCounters();

    /**
     * Closes all events.
     */
    ~PerfCounters();

    PerfCounters(const PerfCounters&) = delete;
    PerfCounters& operator=(const PerfCounters&) = delete;

    /**
     * Zeroes and starts every open event.
     */
    void start();

    /**
     * Stops every open event and saves its count for value().
     */
    void stop();

    /**
     * @param event The event to check.
     * @return True if the event could be opened on this machine.
     */
    bool isOpen(Event event) const;

    /**
     * @param event The event to read.
     * @return Its count between the last start() and stop(), summed over all counted threads
     *         and scaled up if the kernel had to time-share hardware counters. Zero if the
     *         event is not open.
     */
    double value(Event event) const;

    /**
     * @return CPU seconds used by the whole process between the last start() and stop(), summed
     *         over threads. Dividing Cycles by this gives the average clock of the cores while
     *         they ran.
     */
    double cpuSeconds() const;

    /**
     * @return Page faults taken by the whole process between the last start() and stop(): pages
     *         touched for the first time, which the kernel has to map in. Time spent on them is
     *         kernel time, which the Cycles event does not count but cpuSeconds() does.
     */
    double pageFaults() const;

    /**
     * @return The names of the open events, or why none could be opened. For logging.
     */
    std::string describe() const;

private:
    /**
     * One perf_event_open file descriptor per event, or -1 if it could not be opened.
     */
    int fds[NumEvents];

    /**
     * Counts saved by the last stop().
     */
    double counts[NumEvents] = {};

    /**
     * errno from the first event that failed to open, or 0.
     */
    int firstError = 0;

    /**
     * Process CPU time at the last start().
     */
    std::clock_t cpuStart = 0;

    /**
     * Process CPU seconds between the last start() and stop().
     */
    double cpuElapsed = 0.0;

    /**
     * The process's page-fault count at the last start().
     */
    long faultsStart = 0;

    /**
     * Page faults between the last start() and stop().
     */
    double faultsElapsed = 0.0;
};

#endif
