# How Strata works - read this before touching code. This is the design every change follows.
(From docs/paper/Strata-Paper.pdf, HOW_IT_WORKS.md, DETAILS.md, MULTI_GPU.md, RDNA2.md. Coordinator's summary.)

## The idea
A big MoE runs from system RAM with a VRAM expert cache. A token uses 8 of 256 experts per layer (Ornith) or 10 of
512 (Flash-Next), so only a few % of expert bytes move per token. GPU, CPU, RAM and PCIe all work AT THE SAME TIME.
Speed comes from never letting the GPU wait on the host, and from caching the experts the conversation uses.

## Memory tiers
- VRAM: every dense weight (attention/GDN mixers, routers, shared experts, norms, output head, MTP draft block), the
  KV window (--kv int8/q8_0, --kv-resident 32768: the most-read cells on the card, the rest streams from pinned RAM),
  and an EXPERT CACHE that fills all VRAM left. More VRAM = more experts = less CPU work.
- RAM: ALL experts, in one page-locked (pinned) arena, mapped so the GPU can read any expert over PCIe and the CPU can
  compute any expert in place. Nothing is copied per token except the misses the plan sends over PCIe.

## One layer, three workers at once (the core loop)
1. The GPU runs the mixer and the router and writes the chosen expert ids into a small block of PINNED HOST memory,
   the "doorbell". No cudaEventSynchronize, no stream sync, no relaunch.
2. The host thread SPINS on the doorbell (not on the driver). The moment ids land it splits the experts:
   - cached in VRAM -> the GPU computes them, right away, inside the graph
   - a share of the misses (--pcie-frac, set from measured link and CPU speed: share = L / (L + 1.85 R)) -> copied
     to the GPU by the COPY ENGINE (DMA issued when the plan is made), computed when they land
   - the rest -> the CPU pool computes them IN PLACE from the pinned arena (AVX-512/AVX2 kernels)
3. The CPU writes its results into MAPPED memory; a GPU kernel waits on a flag for them, adds everything (router
   weights applied once, on the GPU), and the next layer starts.
4. The WHOLE pass over all layers is ONE CAPTURED GRAPH per window size. The host never waits on a synchronizing
   driver call; GPU-side spin kernels wait for the CPU's flags. GPU and CPU each take about half the round and meet
   at every layer (paper section 5: balanced).

## Verify windows (speculation with the model's own MTP block)
- The MTP block drafts up to 3 tokens; a draft enters the window only while the block is >= 0.5-0.7 confident
  (--spec-min-p). One pass over all layers verifies the window: dense weights are read ONCE per window, so 2.4-3.2
  tokens per pass cost about one token's dense reads. Sampled requests are verified by rejection sampling: exact.
- Copied text: prompt lookup drafts up to 8 tokens from earlier context (--spec 8 --mtp-max-t 3, STRATA_PL_LOOKUP_STREAK).

## The adaptive expert cache (follows the conversation)
- Start: fill VRAM from an expert PROFILE, (layer, expert) pairs ranked by routing frequency on typical prompts
  (make it from the workload: SWE/coding for Ornith). Size slots in BYTES per layer, not one slot size (mixed formats:
  Ornith Q4_0 layers 0-5 have Q4_1 down).
- While answering: every --adapt-every rounds, up to --adapt-swaps experts the conversation keeps routing to (decayed
  counts, --adapt-decay 0.97) replace the least-used slots of their layer. Copies run on a copy stream while the GPU
  works; the evicted expert is dropped at once and the new one is admitted only when its copy has LANDED (no window
  waits). STRATA_ADAPT_OVERLAP: a layer's plan waits only for that layer's copies. --adapt-async 1: swaps on a helper
  thread. STRATA_EXCHANGE_ROTATE: hand buffers over instead of copying.
- So a new topic ("gorilla story") runs a few hundred tokens slower while its experts stream from RAM to VRAM, then
  runs at cache speed. Mixed topics ("gorilla website") mostly hit the cached code experts.
- Do NOT shrink the cache to make room for swapping: measured, fewer cached experts = slower (52 -> 50.7 -> 43 tok/s).

## Two cards
- Layer split: card 0 runs layers [0,K), card 1 [K,48) plus head and MTP; each card caches ITS layers' experts; one
  hand-off per window through pinned RAM (12 KB/token for Ornith). The cards take turns unless:
- --pipeline-windows 2: card 0 starts the NEXT window while card 1 verifies this one. The guess: this window is
  accepted WHOLE and its bonus token is the one the draft block predicts; started only when the draft's probability
  says it will likely hold; a wrong guess restores card 0's recurrent (GDN) state from a copy taken while it ran.
  Draft block on card 0 (STRATA_MTP_STAGE0) so the chain does not fight card 1's window. +12-21% on these V620s.

## Measured on THIS box (2x Radeon PRO V620 gfx1030, card 1 PCIe x4, Ryzen 9 9950X, 92 GB DDR5)
- Main engine, Swift Q4_0 (Flash-Next, 68 GiB of experts, does NOT fit VRAM): 103-114 tok/s sampled, layer split 26,
  --pipeline-windows 2 --spec 8 --mtp-max-t 3 --spec-min-p 0.7 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
  --pool-workers 4 --pcie-frac 0 --kv int8 --kv-resident 65536 --trim-stage-weights, STRATA_EXCHANGE_ROTATE=1,
  STRATA_ADAPT_LAG=2, plus the gfx103x defaults. That is the bar: Ornith Q4_0 (67 GB) must run like it.
- strata-poly, Ornith Q4_0, 262K: 52 tok/s. Kernel time 11.3 ms/token, the cards never overlap, ~850 launches/token
  (~4.5 us gap each on RDNA2), router wait 11 ms/token (a host cudaEventSynchronize every layer), CPU experts
  4.4 ms/token, VRAM hits capped at 87.5% (Q4_1 layers 0-5 never cached; fixed in ed61aaaf, unmeasured).

## What strata-poly lacks against that design, in order of cost
1. The host syncs on the GPU every layer (router wait). Strata: doorbell + spin kernels, the whole pass one graph.
2. Graphs per layer, not one per window per card.
3. The cards take turns; no working pipeline guess (52 windows, 3 held).
4. Profile learned from a few palindrome prompts; Strata's comes from the workload (SWE for Ornith).
5. Swaps that do not land in step with windows (ADAPT_OVERLAP / async / exchange rotation).

## Rules
No ceremony. No test suites, no parity matrices, no A/B protocols, no docs, no readbacks. Read, build, hand over the
run line in RUN.txt, keep cutting. Never drop or skip experts by default, never change sampling. The coordinator runs
the GPUs and measures.
