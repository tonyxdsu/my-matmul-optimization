#include "perf_counters.hpp"

#include <cpuid.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

namespace {

const char* const kEventNames[PerfCounters::NumEvents] = {
    "cycles", "instructions", "L1D-misses", "L2-misses", "L3-fills", "DRAM-fills"};

// True on AMD family 19h (Zen 3/4) or newer. There, the generic "cache-misses" event counts
// L2 misses (event 0x64), and raw event 0x44 counts L1D fills by where the data came from.
bool isAmdZen3OrNewer() {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return false;
    const bool amd = ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163;   // "AuthenticAMD"
    if (!amd || !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const unsigned family = ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff);
    return family >= 0x19;
}

int openEvent(std::uint32_t type, std::uint64_t config) {
    perf_event_attr attr;
    std::memset(&attr, 0, sizeof attr);
    attr.size = sizeof attr;
    attr.type = type;
    attr.config = config;
    attr.disabled = 1;
    attr.inherit = 1;          // also count threads created later, e.g. pool workers
    attr.exclude_kernel = 1;   // user space only, which the default perf_event_paranoid=2 allows
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
}

// Page faults of the whole process so far, minor (no disk read) plus major.
long processPageFaults() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_minflt + usage.ru_majflt;
}

}  // namespace

PerfCounters::PerfCounters() {
    const std::uint64_t l1dReadMiss = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    const bool zen = isAmdZen3OrNewer();
    auto open = [this](std::uint32_t type, std::uint64_t config) {
        const int fd = openEvent(type, config);
        if (fd < 0 && firstError == 0) firstError = errno;
        return fd;
    };

    fds[Cycles] = open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    fds[Instructions] = open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    fds[L1DMisses] = open(PERF_TYPE_HW_CACHE, l1dReadMiss);
    fds[L2Misses] = zen ? open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES) : -1;
    fds[L3Fills] = zen ? open(PERF_TYPE_RAW, 0x0244) : -1;     // ls_any_fills_from_sys.local_ccx
    fds[DramFills] = zen ? open(PERF_TYPE_RAW, 0x0844) : -1;   // ls_any_fills_from_sys.dram_io_near
}

PerfCounters::~PerfCounters() {
    for (int fd : fds) {
        if (fd >= 0) close(fd);
    }
}

void PerfCounters::start() {
    for (int fd : fds) {
        if (fd < 0) continue;
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    faultsStart = processPageFaults();
    cpuStart = std::clock();
}

void PerfCounters::stop() {
    cpuElapsed = static_cast<double>(std::clock() - cpuStart) / CLOCKS_PER_SEC;
    faultsElapsed = static_cast<double>(processPageFaults() - faultsStart);
    for (int i = 0; i < NumEvents; i++) {
        counts[i] = 0.0;
        if (fds[i] < 0) continue;
        ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);

        std::uint64_t data[3];   // count, time enabled, time running
        if (read(fds[i], data, sizeof data) != static_cast<ssize_t>(sizeof data) || data[2] == 0) continue;
        // With more events than hardware counters, the kernel rotates them and each one runs
        // only part of the time. Scale the count up to the whole measured interval.
        counts[i] = static_cast<double>(data[0]) * static_cast<double>(data[1]) / static_cast<double>(data[2]);
    }
}

bool PerfCounters::isOpen(Event event) const {
    return fds[event] >= 0;
}

double PerfCounters::value(Event event) const {
    return counts[event];
}

double PerfCounters::cpuSeconds() const {
    return cpuElapsed;
}

double PerfCounters::pageFaults() const {
    return faultsElapsed;
}

std::string PerfCounters::describe() const {
    std::string names;
    for (int i = 0; i < NumEvents; i++) {
        if (fds[i] < 0) continue;
        if (!names.empty()) names += ' ';
        names += kEventNames[i];
    }
    if (names.empty()) return std::string("none (perf_event_open: ") + std::strerror(firstError) + ")";
    return names;
}
