Strata keeps always-used weights and hot experts in VRAM, with expert weights in host memory available to CPU computation and pinned transfers.
The GPU publishes routed IDs and activations into mapped pinned memory; the CPU spins on a fenced doorbell instead of synchronizing the driver at each layer.
The host divides every selected expert among cached GPU work, a bounded PCIe DMA share, and CPU work; the GPU computes cached experts while transfers and CPU work overlap.
CPU results return through mapped memory; the GPU combines every contribution and advances inside one captured graph per verify-window size (per stage on a split).
The adaptive cache ranks decayed conversation usage against existing residents; adapt-every controls selection frequency and adapt-swaps bounds each round.
Nonblocking swaps invalidate an evicted slot before overwriting it, publish a replacement only after its copy lands, and never overwrite slots still read by an in-flight window.
Adapt-overlap is a distinct single-GPU scheme: layer-ordered arrivals delay only that layer’s plan; async split/pipeline swaps instead respect each stage’s in-flight windows.
Two-card pipelining guesses full acceptance plus the draft bonus, starts only with sufficient confidence, and restores front-stage recurrent state when exact token/length matching fails.
At this review, poly had Q4_1 cache eligibility, async arrivals and opt-in partial-stage graphs, but all graph misses ran on CPU; this change adds a bounded DMA tier, while a full pinned expert arena and stage-0 draft placement remain absent.
Next changes must preserve sampling and compute every expert; the documented tail-skipping policy is outside this task’s exactness rules.
