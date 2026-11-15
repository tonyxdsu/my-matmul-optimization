Draft of the takeaways and summary of exercises.cpp

2. Sum of Squares vs dot.
- Zen 5 Core:
    - 2 512-bit FMA units, 4 cycle latency. Max is retiring 2 FMA per cycle = 2 * 16 FLOPs = 32 FLOPs per cycle
    - 2 512-bit loads per cycle
    - 2 512-bit add units, 2 cycle latency. Max is retiring 2 adds per cycle = max 2 * 8 FLOPs = 16 FLOP per cycle
- Need multiple accumulators to keep up to 8 FMA in flight for max 32 FLOP per cycle per Zen 5 core.
- sum sqrs 
    - needed 8 accumulators
    - 32 max FLOPs theoretical
- dot
    - needed 4 accumulators
    - 16 max FLOPs due to needing 2 loads per FMA, therefore we can only have 1 FMA per cycle
- #pragma GCC unroll 16 for accArr[]
    - force loop unrolling so GCC keeps accArr[] in registers, instead of stack which would result in storing back to stack per iteration (this means storing back to L1 cache most likely right? though probably core's store buffer and reread from there next iteration, where store buffer is as slow as L1 cache)
- linear chain of adds of final results
    - dependency chain that we didn't bother solving
    - started 1 add per 2 cycles (2 cycle latency) with 1 add in flight at 4 FLOPs per cycle
    - theoretical maximum would have been retiring 2 adds per cycle at 16 FLOPs per cycle with 4 in flight (probably through manually coding an explicit reduction tree knowing the number of accumulators beforehand; probably can't reach 16 FLOPs with reduction tree with relatively low number of accumulators)
    