# L40S same-GPU throughput beam

This is the canonical live ledger for the single-L40S throughput campaign.
Candidate-specific design notes may remain under `benches/`, but results,
promotion thresholds, active ranking, and kill evidence belong here.

Objective: beat canonical Q4_K_M aggregate decode throughput while preserving
the native Mach-1 memory advantage. A format candidate may replace an existing
payload, but it may not keep a persistent shadow. Routed experts remain at
their native approximately 1.5 bpw unless a separately approved byte ledger
says otherwise; whole-model Q4/int4 conversion is out of scope.

## Baseline and acceptance target

The clean same-allocation `l40s-latest-main-single-gpu-20260815` receipt at
p16 (128 prompt plus 128 generated tokens, B=16) is:

- Trellis `p16final`: 652.43 generated tok/s, 24.523704 ms/model step.
- canonical Q4_K_M: 698.97 generated tok/s, 22.890825 ms/model step.
- exact tie: save 1.632879 ms/step.
- publishable 5 percent lead: 733.92 tok/s, save 2.722918 ms/step.

The prior exact sibling-transform screen realized 70.5 percent of an isolated
gross price at the model wall. Until a candidate has its own full-model pair,
the conservative isolated promotion lines are therefore 2.316140 ms/step for
a tie and 3.862295 ms/step for a 5 percent lead. A winning candidate still
requires reversed or T-Q-T confirmation and the production server lane.

The graph-disabled TIME=1 census is ranking evidence, not an additive wall
decomposition:

- expert mega/QONCE: 10.20 ms/step;
- compressed spine MMA: 4.46 ms/step;
- uncovered m512 spine walk: 1.37 ms/step;
- spine input/output transforms: 4.51 ms/step;
- output head: 0.76 ms/step.

## Active ranking

1. **p16goal3 - exploit parent, +33.6 percent OVER Q4 at B16, goal met.**
   The full scheduling + state-fusion stack (p16qace scheduling winners, TW=8,
   n512 split, GDN state-gather fusion level 1+2, batched GDN prep-chain
   fuse). np16 bitwise full-continuation hash 94110fed88ef preserved at every
   rung (`qace-np16`, `sgf-np16-s1`, `goal-np16-s2`, `goal2-np16-s3`,
   `goal3-np16-s1`). Goal wall `goal3-wall-s1` (T-Q-T same-container,
   graphs-on, HOSTGATE ok): 928.52 / 694.09 / 926.09 B16 tok/s -> 1.3378 and
   1.3343, both T draws individually over the 1.30 line. Zero-byte: native
   payloads untouched, no persistent allocations, 7.3 GiB total against Q4's
   20.2 GiB. Exact-B16 gated (B1 unchanged); server admission of the indexed
   state ops still needs the copying fallback for forked s_copy mappings.
   Detail receipts in the scheduling-beams section below.

1b. **Prior validated native-p16 composition - superseded control.** Runtime,
   resource gate, factorial screen, full-continuation hash, and evidence are
   promoted on `mach1/main` through `7218ddc9f`. The healthy T-Q-T receipt
   measured Trellis at 691.81 and 692.88 tok/s around Q4 at 696.49; geometric
   mean 692.344793 tok/s, 0.595157 percent behind Q4. Preserves native
   payloads, zero persistent/global bytes. Control lineage for the margin
   arms above.
2. **Native Stream-K / split-K spine - validated composition near-miss.** The
   integrated source through `6acc49577` keeps the native 4-bpw payload, adds
   only 393,216 transient bytes, and adds zero persistent bytes. Resource app
   `ap-RQt30D7e4XCkO7Gv2etN8K` measured the legacy and K4 cores at 80 registers
   with literal zero stack/spill; bitdump app `ap-w7OA2f8UgJ5CDuBOeUl7e8`
   passed at `1.366e-05` relative-L2 and bitwise candidate repeat; np16 app
   `ap-3o6g8vC6U79xPCOcEDp0BT` matched the full continuation hash
   `94110fed88ef`. Production TIME app `ap-nMiyilZKK1bGiQfFlBnIJe` saved
   0.3380 ms/step. Two healthy wall draws both crossed their same-draw Q4:
   707.041 versus 690.90 tok/s with 0.3540 ms isolated saving, then 695.062
   versus 694.80 with 0.1137 ms saving. The first missed B1-neutrality by
   0.146 percentage points and the second missed the 0.20-ms materiality bar,
   so preserve this as a strong combination ingredient rather than promoting
   it alone.
3. **Standard-output replacement - production runtime priced at 1.1242
   ms/step, quality gate is the only blocker.** The replica screen
   (`ap-BrEyx9KIODJK2ryfG8EaYS`, 0.429727 ms net) is superseded by a real
   runtime integration: row-scale epilogues on the rt spine (plain, TC, and
   split-K sum4 forms), an expert-mega STD variant that deletes the three
   in-mega output FWHTs, and SM-chunked row-scale epilogues across all six
   sibling-batched `_out_tcb` sites. The final fail-closed TIME pair on the
   split-K parent (arms p16ks4/p16std, app `ap-X8qM4gweC7ApZCAuQbiaPW`)
   measured, over 129 decode steps x 40 MoE layers with a label-equal
   48-record census and 64-register/zero-spill megas: exp_mega 992.15
   us/step, shexp_out 98.88, qkvzb_out 81.83, rt_out_tc 66.52, split4 66.16,
   qkvb_out 6.27 - composed 1.3118 ms/step, 1.68x the wave-model projection,
   because the in-mega FWHT removal also deletes barrier chains the waves
   never modeled. Instrumented B16 moved 560.39 to 591.80 tok/s. Two earlier
   receipts bracket the path: phase 1 (spine + mega only,
   `ap-GGXeUibEJTcn2rVSV2XFrA`) read 1.1242 ms/step, and the first
   one-block-per-op sibling conversion (`ap-nkI0lzeCRspqXLDeG0j2BW`) showed
   qkvzb/qkvb regressions of -84.39/-24.95 us/step that the grid.y m-chunking
   fix turned into the +81.83/+6.27 above. The runtime engages only under
   `GGML_MACH1_RT_STDOUT=1` plus `GGML_MACH1_TIME=1` (wrong-basis payloads,
   TIMING_ONLY). Everything now rides the cached one-hot L39
   re-encode/reload selector quality gate (Mac-local, see the resume queue).
4. **Native int5 head IMMA - exact K64 schedule killed, family slot open.** App
   `ap-9FY7nmqLvyYG1VErtJMz20` passed its integer oracle, repeat, and resource
   gates (producer 19 registers, IMMA core 55, control 63; all zero spill) but
   moved only 745.926 to 672.725 us including q8 production. The 0.0732-ms
   saving misses the predeclared 0.20-ms advance line, so do not build K32 or a
   production integration from this schedule. A new head idea must change the
   algorithmic surface rather than retune this exact K64 path.
5. **Format-v4 tied gate/up - singleton rejected, ingredient retained.** Its
   synthetic tied-input proxy saved only 0.10546 ms/step. Keep it separate
   from the newer production-register standard-output beam above.

Direct-L1, the scalar and vector-staged signed-Q6 family, expert collision
fanout, paired CTA reuse, the named-barrier IMMA pipeline, and the scaled
QONCE q8 producer family are closed; their evidence is below. Do not revive
them without a materially new representation or schedule.

The exact two-pass global-FWHT wave is also closed. Its alpha-complete
all-at-once floor and transform math remain valid combination evidence, but a
production-semantic layer schedule must be fundamentally different before the
family can re-enter the active beam.

## Submission checkpoint: 2026-08-15 margin cycle

The following receipts are the authoritative decisions after the promoted
native composition. They are recorded here after more than five submissions so
the search does not collapse into a single-incumbent hill climb.

| beam | source / app | decisive result | decision |
| --- | --- | --- | --- |
| native int5 head, unrolled direct-B | `029d9b6d4`, `ap-xuhZZ09wtT0temvmQCFtzx` | resource 80 registers/zero spill; full-continuation hash exact; healthy-host wall saved about 0.007 ms/step | singleton neutral; retain only as parent of occupancy retune |
| native int5 head, rolled/half2 | `3032ccaf8`, resource `ap-aVrIWL0P34yIIvlv5ZuvY4`, correctness `ap-UXW6QXCIG0IrG9qYGCJZBG` | both modes 44 registers/zero spill and exact 33-token continuations | held near-miss; three throughput draws failed B1 host gate and are excluded |
| global FWHT floor | `266690262`, `ap-RKKK2RE2h0udtdAQIJ95U9` | 1.229824-ms median versus 2.880000-ms kill, exact census, zero-spill kernels | advance to alpha-complete, schedule-faithful wave price |
| global FWHT alpha-wave | `f5d8b953a` then control fix `0bb600f7b`, apps `ap-MMHcvJNc5FmyHLXHigPsfO` / `ap-4MJj8A9RZhaJNPVMSHDaPj` | all candidate kernels zero-spill; fused control still stack 8 B, stores/loads 4 B | implementation blocked, not family-killed; fix control ABI without relaxing gate |
| duplicate-route expert IMMA oracle | `a90aa8179`, `ap-Nu4PqtodAMMgx5ZHWeQmj5` | real seven-route group, exact integer map, max rel error `5.43147326e-06`, core 40 registers/zero spill, native 1.5 bpw and zero scratch delta | advance to integrated duplicate-only mega schedule |
| global FWHT inline-alpha | control fix through `a0ee6140`, diagnostics `ap-IAHx9Q29nBlybHo19XTGOm` / `ap-N24zObGLPCVpISGvi9zEBa` / `ap-h76xLzx10errHmlnPYBbeI` | all six kernels zero-stack/spill; inline HMMA corrupts while identical-input precomputed alpha passes all dimensions at about `2.8e-4` rel RMS | numeric contract exonerated; timing forbidden until inline alpha publication/shared visibility is repaired |
| duplicate-route expert IMMA integrated | `18d750a43`, PTX `ap-5dM9UFxzNPfKjAF9xtzrdy`, wall `ap-BxAqrVFhc4w9B8JDjEmblb`, TIME `ap-cFb4Tcycv1LcphrFGBDiu5` | candidate/control both 64 registers and zero spill; candidate 456.99 tok/s; expert kernel 548.63 versus 255.00 us/layer | reject integrated scheduling singleton; retain oracle and reprice only the transposed duplicate walk |
| native Stream-K K4 singleton | `6508cc181`, `ap-kMndTaD9zjUuCiwh9KR2fR` | control/K4 cores 80 registers and zero spill; 21.303 -> 12.119 us, ratio `0.5689`; deterministic `2.151e-07` rel-L2; projected 1.039-ms B16 saving | decisive pass; integrate on current main and run resource/numeric/full-model factorial gates |
| global FWHT alpha-complete + layer wave | `911b6f931`, `ap-8iqV5SxZKVBuYZauSvTKdC` | production discriminator and all 14 waves bitwise exact; alpha-complete median 1.230848 ms; schedule-faithful candidate 2.339840 ms versus fused 1.614848 ms | retire exact two-pass wave after repeatable 0.724992-ms regression; preserve only as structural-fusion material |
| native Stream-K integrated | `6acc49577`; resource `ap-RQt30D7e4XCkO7Gv2etN8K`; bitdump `ap-w7OA2f8UgJ5CDuBOeUl7e8`; np16 `ap-3o6g8vC6U79xPCOcEDp0BT`; TIME `ap-nMiyilZKK1bGiQfFlBnIJe`; walls `ap-c6OL0vpWmVJpnfUA5rt6O9` / `ap-dXPIRfsN0FJZyqQ0FBMdwb` | resources/numerics/continuation pass; TIME saves 0.3380 ms; both healthy walls beat Q4, but one misses B1-neutrality and one saves only 0.1137 ms | preserve as validated composition near-miss; no standalone promotion |
| standard-output production-register proxy | `e05194cc1`, `ap-BrEyx9KIODJK2ryfG8EaYS` | exact token-major/repeated-`sv[j]` oracle PASS; 0.743632 -> 0.311824 ms, raw save 0.431808 ms; exact split-aware net 0.429727 ms; all kernels zero-stack/spill | advance cached quality re-encode; exporter/runtime remain forbidden |
| native int5 x transient-q8 head IMMA | `a7835133c`, `ap-9FY7nmqLvyYG1VErtJMz20` | correctness/resource PASS; 745.926 -> 672.725 us including producer, only 0.0732 ms saved | kill exact K64 schedule; do not spend on K32 or runtime integration |
| replacement-only S4X2 current-parent reprice | `39c18b5f6`, `ap-1ZjUmrWgdp4jRNu5yg4TVC` | every resource/mapping/census gate PASS, but exact current q8/cp.async+TT parent was 3.161429 ms versus S4 4.166997 ms; robust saving `-1.008640` ms | kill current S4 schedule; no quality encode or runtime integration |
| scaled QONCE q8 pack, record-load fold | `9ab7f53`, `ap-5bdP5ywvyCsLBImwrcuO0Y` | bitwise records (221,184 words, 0 mismatch, du0 path covered), CPU scalar oracle exact, 56 registers/zero spill; replica census timing 0.437248 -> 0.574576 ms/step (`-0.137328`) | kill the pack-side fold: concentrating the divisions in the d/16 pack threads loses more than the deleted pass |
| scaled QONCE q8 pack, FWHT last-pass fold | `e7d1080`, `ap-XHjK6TAibaYpQgA1SmWMpy` | same bitwise/CPU/repeat/resource gates PASS across all three arms; 0.437248 -> 0.428788 ms/step, saving only 0.008460 versus the 0.150 kill line | kill the family: the deletable pass+barrier surface prices at about 8.5 us/step, falsifying the 0.3-0.8 ms ceiling; no production TIME or wall spend |
| standard-output production runtime (spine + mega) | `79877fb`..`6c0a338`, `ap-GGXeUibEJTcn2rVSV2XFrA` | label-equal 48-record census over 129 steps x 40 layers; megas 64 registers/zero spill both arms; exp_mega saves 1011.18 us/step, rt_out keys 112.99 us/step; composed 1.1242 ms/step versus the 0.782927 projection | advance to sibling-batch coverage |
| standard-output sibling epilogues, full coverage | through `ap-X8qM4gweC7ApZCAuQbiaPW` (grid fix after `ap-nkI0lzeCRspqXLDeG0j2BW` showed -84.39/-24.95 us/step one-block regressions) | same census/resource gates; per-key deltas 992.15 (mega), 98.88 (shexp), 81.83 (qkvzb), 66.52/66.16 (rt spine), 6.27 (qkvb); composed 1.3118 ms/step; instrumented B16 560.39 -> 591.80 | runtime half complete and priced; blocked solely on the Mac-local L39 selector quality gate, then exporter/metadata admission and the wall bracket |

Next combination decisions are explicit:

- do not revive the exact native-int5 direct-B/rolled/half2/K64-IMMA head
  schedules; a new head candidate needs a different algorithmic surface and a
  fresh ceiling;
- do not retune the killed two-pass global-FWHT layer wave; only a materially
  different fused production schedule may re-enter, with a fresh ceiling and
  the same production-reference gate;
- the duplicate-route expert IMMA family is closed by its separate postmortem;
  do not re-enter it through scheduler-only retuning;
- the Stream-K interaction is now measured, not projected: p16std rides
  p16ks4, replaces split-K's TC sum epilogue with the sum4 row scale, and the
  composed TIME pair reads 1.1242 ms/step (app `ap-GGXeUibEJTcn2rVSV2XFrA`).
  A production wall claim still requires the quality-passing re-encode plus a
  same-build control / split / standard / combination / Q4 bracket.

### Handoff resume queue

Billing is restored. Resume in this order:

1. Run the cached standard-output quality gate from
   `/private/tmp/llm-compression-standard-output` at `75a3b09b2`:
   `PYTHONPATH=src /Users/dhruv/.local/bin/modal run --timestamps -m
   llm_compression.qrec.apps.standard_output_qwen36::probe --go`. The command
   encodes/persists/reloads one L39 expert first and only then admits paired
   selector RKL/flip/tail scoring. This remains the gating step, and it is
   currently Mac-local ONLY: commit `75a3b09b2` (and the
   `standard_output_qwen36` app it contains) is not on the GitHub
   `llm-compression` remote, so no remote lane can run it. Pushing that
   worktree commit unblocks remote execution; Modal itself is reachable from
   the remote lane.
2. DONE 2026-08-16: the scaled-QONCE pack beam was implemented and priced
   through its byte oracle and resource gates, then killed at the timing gate
   in both schedules (see the checkpoint rows and the closed screen below).
   Do not resume it; the producer surface it deleted is worth about 8.5
   us/step.
3. The llama.cpp runtime half of the standard-output composition is DONE,
   with full epilogue coverage (spine, expert mega, and all six
   sibling-batched sites), priced at 1.3118 ms/step on the split-K parent
   (2026-08-16, TIMING_ONLY, fail-closed behind `GGML_MACH1_RT_STDOUT=1` +
   `GGML_MACH1_TIME=1`). When quality passes: implement the exporter plus
   metadata-driven admission (replacing the env gate), then the same-build
   control / split / standard / combination / Q4 bracket and a
   production-server F-Q-F confirmation.

## Pure-runtime scheduling beams (2026-08-16, weights untouched)

Three parallel beams against the ranked B16 census (`megarace` lane; control
`p16ks4` reads 18.0-18.2 ms/step across its nt=16 mach1-timed keys, exp_mega
10.17-10.21 ms of it). Receipts under `/vol/bench/results/megarace-s*`.

- Attribution (`p16mp1`, walks skipped): exp_mega 10.18 -> 5.68 ms/step, so
  the walk tiles are ~4.5 ms and ~5.7 ms is producers, H-FWHTs, record
  staging, fences, and counters. After the known ~1.0 ms H-FWHT and ~0.44 ms
  producer shares, ~4.2 ms/step is task-skeleton overhead across ~3k
  tasks/layer - the largest pure-runtime surface in the census.
- QONCE + DNCT=32 down phase (`p16qd32`): exp_mega -352 us/step; with
  slot-release (`p16qdsr`) -403 us/step and B16 559.5 -> 570.4. The down
  walk's 96 idle threads per vblock were real. Bit-exact by the
  mach1_exp_mega_tile CT argument; np16 hash still required before any
  default flip.
- Slot-release alone (`p16qsr`): +25 us/step - neutral. The global H wait is
  not a cost; keep only as a rider on DNROWS.
- Fused shexp out+glu (`p16soglu`): the pair's two out transforms plus the
  silu launch collapse into one kernel; `shexp_gub16_oglu_tcb` 412 us/step
  replaces 406 (out) + ~275 (silu), saving ~270 us/step, B16 571.0.
  Bit-identical values (same tc_fwht_dyn body, sv fold, and silu order).
- Task widening (`GGML_MACH1_MEGA_TW`) is the decisive scheduling result:
  each task covers tw row chunks so the q8 record staging and counter
  traffic amortize. Same-container megarace-s3: TW=2 cut exp_mega 10.220 ->
  8.501 ms/step, TW=4 -> 7.432 (B16 562.7 -> 622.7), and the composition
  p16qace (TW4 + DNROWS + SLOTREL + OGLU) -> 7.042 with the nt=16 census
  18.102 -> 14.614 ms/step and instrumented B16 637.3 (+13.3 percent), all
  weights untouched. megarace-s4 saturated the width: ace at TW=8 read
  6.835 ms and TW=16 read 6.849, so TW=4-8 is the plateau; the
  same-container instrumented Q4 bar was 693.89.
- GRAPHS-ON INTERLEAVED WALL (`qace-wall-s1`, order p16ks0 / p16qa0 / q4km /
  p16qa1 / p16ks1, npl 1,16): candidate 803.75 and 811.29 generated tok/s at
  B16 against same-container Q4_K_M 696.48 and control split-K parent
  699.07 and 698.19. That is +15.4 to +16.5 percent over canonical Q4 with the native
  payloads untouched, from runtime scheduling alone. B1 host gate healthy
  (Q4 172.79 <= 190, control 162.15 >= 150); candidate B1 165.99/159.69 is
  neutral against control. The np16 full-continuation hash pair
  (`qace-np16-s2`) PASSED with both arms at the historically certified
  `94110fed88ef`, so the composition is certified bitwise-identical to the
  split-K parent. p16qace (GGML_MACH1_MEGA_TW=4, GGML_MACH1_EXP_DNROWS=1,
  GGML_MACH1_MEGA_SLOTREL=1, GGML_MACH1_SHEXP_OGLU=1 on the p16ks4 stack)
  is therefore the new exploit parent and the control for every future
  margin arm. A second-container T-Q-T (`qace-wall-s2`) reconfirmed:
  809.72 and 807.14 tok/s around same-draw Q4 at 694.77 (+16.4 percent
  geometric mean), candidate B1 164.47/165.22 with Q4 at 171.87. The env defaults are deliberately NOT flipped: a default flip
  rebases every gate and belongs to a separate certified step.

The fresh attribution on the new parent (`megarace-s5`, p16qace at 14.690
ms/step nt=16 census, B16 638.1) re-ranks the beams:

1. exp_mega 7.158 ms/step; with walks skipped 3.040, so the walk tiles are
   ~4.12 ms - the true trellis-decode bottleneck - and the residual
   non-walk overhead is ~1.6 ms after the ~1.0 ms H-FWHTs (standard-output
   claims those, weights permitting) and ~0.45 ms producers.
2. CORRECTION to the first read: the shexp TT walk's grid is
   dim3(64, 1, nt/ttc=4) = 256 blocks, not 64 - the walk is decode/latency
   bound, not grid-starved, and its ttc=4 slices already cost 4x redundant
   decode. The K-split idea is withdrawn; do not implement it on the
   underfill claim.
3. The pipeline-depth family is KILLED by its own predeclared gate
   (`--imma8-pipe`, `imma8-pipe-s2`): with the CPU integer reference and
   the deep3-vs-base2 bitwise gate both PASS, deep3 read 1.026-1.031x base2
   at every shape - the lost block/SM outweighs the extra in-flight stage.
   The decisive by-product is the pure-LDG stream ceiling at the exact
   production geometry: 977.9 GB/s at 512 blocks (m8192), 528.2 at 256
   (m4096), 320.4 at 64 blocks (m2048) - the DRAM pipe cannot be filled by
   few 128-thread blocks, which retroactively explains the split4 win
   (0.5689 by raising 64 CTAs to 256) and indicts the last narrow-grid
   production shape, rt_imma8 m2048 n512 at 64 CTAs and 0.386 ms/step.
   The remaining ldg/base2 gap (~1.35x, L2-warm) is the decode+MMA shadow,
   not prefetch depth; do not revisit depth without a new representation.
   The follow-up n512 split (`megarace-s6`, `GGML_MACH1_RT_IMMA8_SPLITK_N512`)
   is also killed at materiality: the walk moved 378.8 -> 333.1 us/step but
   the net was +0.125 ms and B16-neutral - at NSTEPS=1 each CTA runs one
   unpipelined stage, so the 0.5 MB shape sits at its latency floor, not the
   streaming rate. The code stays as an unpromoted opt-in.
   The probe's absolute GB/s are L2-warm and not comparable to the cold
   full-model census. A split-commit-group arm was withdrawn: the prefetch
   stripes every thread across both data halves, so thread-scoped
   wait_group cannot gate per-half consumers. nsys note: the nsysprof lane
   needs `--cuda-graph-trace=node` before its kernel sums mean anything
   (the first `qace-nsys-s1` capture had no per-node rows).
4. shexp chain step 2 (down-u q8 pack folded into the fused out+glu) saves
   the rt_u_tc m2048 n512 launch (~0.33 ms surface) but is PARKED with a
   concrete blocker: the gu batch exec has no reference to the down tensor,
   so its su/zstep are unreachable without graph-consumer plumbing.
5. EXECUTED - GDN recurrent-state gather fusion. `ggml_gated_delta_net_idx`
   gathers state rows through s_copy inside the kernel (exact-B16 gated like
   the rest of the p16 stack after unstable B1 draws; aliasing contract:
   reads s_copy[i], writes head+i, safe only when the mapping is identity on
   overlaps - server admission needs a copying fallback for forked
   mappings). Same-container race `megarace-s9`: instrumented B16 611.2 ->
   652.6 (+6.8 percent). np16 hash PASS at `94110fed88ef` alone
   (`sgf-np16-s1`) and composed with TW=8 plus the n512 split as `p16goal`
   (`goal-np16-s2`). First graphs-on draw read 840.29 B16 against
   same-container Q4 675.79 (+24.3 percent) before the B16 gate landed.
   ORIGINAL SIZING - 2.25 ms/step ceiling:
   The shape-split TIME=2 census (`megarace-s8`) shows
   `GET_ROWS s0=524288x16` at 75.02 us x 30 calls/step: build_rs
   unconditionally materializes the full 33.5 MB state copy per GDN layer
   per step because identity-ness of s_copy is a runtime property a
   captured graph cannot branch on. The gather runs at ~894 GB/s - it is
   pure bytes, not inefficiency, so the only fix is not moving them: teach
   the fused GDN core to gather state rows through the 16 s_copy indices at
   load (and scatter on store), then drop the standalone get_rows from the
   mach1 build_rs path. Correctness surfaces to gate: rs_zero clearing, the
   extra-states copy between n_seqs and n_rs, and slot reuse across
   ubatches. Same census: the router glue is a census artifact - graphs-on
   the topk-moe fusion engages (`topk_moe:fused` fired), so do not spend on
   routing.
   FOLLOW-UP - composed goal wall (`goal-wall-s1`, T-Q-T graphs-on,
   HOSTGATE ok): p16go0 885.45 / q4km 693.27 / p16go1 883.19 B16 ->
   +27.7 percent mean, np16 PASS.
6. EXECUTED - GDN conv-chain fusion (state fusion level 2).
   `ggml_ssm_conv_idx` collapses the conv gather + concat + ssm_conv +
   silu + state write into one indexed kernel under the same aliasing
   contract (reads conv-cache row s_copy[i], rolls the window into row
   head+i; exact-B16 gate; nc==4 only). Two integration potholes, both
   receipted: the remote build caught a `gdn_sgf` use-before-decl
   (`goal2-np16-s1` build fail), and the CUDA `supports_op` shape test
   (`src[0]->ne[1] % 128 == 0`) silently dropped the indexed node to the
   CPU backend which aborts on the 4-src form (`goal2-np16-s2`) - the
   indexed form is now admitted explicitly (nc==4, channels
   bounds-checked). np16 hash gate PASS composed as `p16goal2`
   (`goal2-np16-s3`): full-continuation 94110fed88ef == control, all six
   markers exact-count-1 (tw=8 dnct=32 slotrel=1).
   GOAL WALL (`goal2-wall-s1`, T-Q-T graphs-on, HOSTGATE ok q4km=172.6
   B1): p16h0 906.50 / q4km 697.20 / p16h1 903.77 B16 -> ratios 1.3002 /
   1.2963, mean 1.2983. The 1.30 goal line STRADDLES the two T draws -
   not cleared; conv fusion itself is worth ~+20 tok/s over p16goal in
   matched containers.
   FOLLOW-UPS, both receipted: the tw race on the goal2 base
   (`goal2-twrace-s1`, graphs-on same-container) is a NO-GO - tw=12/16/32
   all read 925.98-927.64 against tw=8's 928.69, i.e. the down phase is
   already amortization-saturated at tw=8 despite its 128 chunks. The
   TIME=2 per-op census on goal2 (`goal2-census-s2`,
   /vol/bench/results/goal2-census-s2/megarace_p16g2cen.txt) rules by
   call count: the remaining decode tail is launch count, not bytes -
   per step the GDN pre-chain runs 6 tiny nodes x 30 layers (SIGMOID,
   ADD, SOFTPLUS, MUL, 2x L2_NORM) and the router glue ~6 x 40 layers
   (SOFT_MAX, ARGSORT, GET_ROWS, SUM_ROWS, CLAMP, DIV; the topk_moe
   fused counter fires but the chain still executes - the exoneration
   in item 5 was wrong at op granularity). Router fusion via the stock
   topk kernel would NOT be bitwise (different reduction order shifts
   ulps -> trajectories), so the pre-chain was the safe kill.
6a. EXECUTED - batched GDN prep-chain fuse (`GGML_MACH1_GDN_PREP`,
   p16goal3 = goal2 + prep). The ns==1 core fuse
   (mach1_gdn_core_kernel) cannot run at B16 and the full-region matcher
   has no get_rows to enter under sgf, so the six pre-chain nodes ran
   unfused at B16. New 11-node positional matcher [l2q .. sigmoid]
   (same window the core matcher proves at ns==1, state checks dropped -
   the indexed gdn's src[5] is the raw cache) + one kernel writing ONLY
   the four dsts the delta-rule node reads (l2q, l2k, gate mul, beta
   sigmoid), each with the stock kernel's exact op order
   (l2_norm_f32<WARP_SIZE> serial stride + rsqrtf(fmaxf(sum, eps*eps)),
   op_sigmoid, op_softplus, plain add/mul). The delta-rule node keeps
   its own indexed launch. 6 launches -> 1 per GDN layer per step.
   np16 hash gate PASS first try (`goal3-np16-s1`): full-continuation
   94110fed88ef == control, prep marker exact-count-1.
   GOAL WALL PASS (`goal3-wall-s1`, T-Q-T graphs-on, HOSTGATE ok
   q4km=172.8 B1): p16j0 928.52 / q4km 694.09 / p16j1 926.09 B16 ->
   ratios 1.3378 / 1.3343, mean +33.6 percent. BOTH T draws clear the
   1.30 goal line individually - the /goal "30 percent faster than q4"
   condition is MET, same-container, np16-bitwise, zero-byte (no weight
   or memory-layout change; RAM stays 7.3 GiB vs Q4's 20.2 GiB).
   SECOND-CONTAINER CONFIRMATION: `goal3-wall-s2` and `-s3` drew
   degraded hosts (control B1 124.8 / 108.1 under the 150 floor) and
   were DISCARDED per the host gate; `goal3-wall-s4` (HOSTGATE ok,
   control B1 161.8, q4km B1 172.1) read p16j0 922.68 / q4km 697.78 /
   p16j1 910.18 B16 -> ratios 1.3223 / 1.3044, mean +31.3 percent. Both
   healthy containers put BOTH T draws individually over 1.30. Goal
   receipts complete.
7. Prefill is the next campaign surface: the wall shows candidate PP at
   1,580-1,660 tok/s against Q4's 6,300-6,400 - decode now leads by 30+
   percent while prompt processing trails 4x.
   ATTRIBUTION (`goal3-ppprobe-s1`, walk-skip probe on the goal3 stack,
   nt=512 stage keys base vs MEGA_PROBE=1): the trellis WALK is NOT the
   prefill bottleneck - walk share measures ~0 percent (818.5 vs 822.9
   ms, probe inside noise) because prefill already decodes once per
   ubatch (`rt_dense_decode m=8192 nt=512` is 5.9 ms total). The cost is
   the dp4a APPLY GEMMs against the 512-token tiles:
   `exp_zdp_apply m=512` 260.9 ms + `m=2048` 119.1 ms (experts),
   `rt_apply m=8192` 172.9 + `m=2048 n=4096` 77.5 + `m=4096` 65.7 +
   `m=512` 34.1 + `m=2048 n=512` 28.3 ms (spine) - ~758 ms of ~818
   sync-inflated for the 2048-token B16 prefill. Q4 runs the same shapes
   through tensor-core MMQ. The beam: extend the existing imma8/mma16
   tile machinery (engaged at nt=16 decode and at the head) to nt=512
   apply tiles for the spine and expert paths. Do NOT spend on prefill
   walk amortization - it is already amortized.
   RUNG 1 EXECUTED - spine apply as tensor-core GEMM
   (`GGML_MACH1_RT_APPLY_TC=1`, arm p16pp1): the bank is already a plain
   fp16 [m, n], so past nt >= 64 the apply is scr_u -> fp16 (rn) +
   cublasGemmEx 16F x 16F -> 32F into the same scr_v layout. Same
   container (`pp-tc-s1`): PP 1574.45 -> 1975.27 (+25 percent), q4km
   5416.43; TG unchanged (872.0 vs 867.6, decode keeps the certified
   path - the TC branch gates at nt >= 64). Quality class: one
   activation-side fp16 rounding (weight side reads the same bank);
   RUNG 2 EXECUTED - expert zdp apply as s8 MMA
   (`GGML_MACH1_EXP_APPLY_MMA=1`, arm p16pp2 = pp1 + it): the decode
   phase expands z-nibbles to TRUE int8 (w = z-8, state sign folded into
   w0), so the m16n8k16 s8 MMA's integer result equals the dp4a form
   exactly; fsc = zs0*gv*du/127 applied per (pair, tile) on the i32
   fragment - only the cross-tile float association differs. Same
   container (`pp-mma-s1`): PP p16pp1 1638.21 -> p16pp2 2038.59, q4km
   5327.05; TG untouched (decode P < dense_min). TOKEN-EXACT
   (`pp-mma-nt-s1`): p16pp2 np16 stream sha 34cdad2c7570 == p16goal3.
   RUNG 2b - the pp2 census showed the expert prefill is DECODE-bound
   (each block expands 16 x n nibbles for ~16 pairs of apply); packing
   the expand into two u32 stores per state (was 8 conflicted byte
   stores) lifted PP again. Healthy wall `pp-mma-s6` (HOSTGATE ok,
   control B1 159.0): p16pp2 PP 2493.70 / q4km 6323.41 B16 -> gap 2.54x
   (campaign start 4.03x); TG 862.56 / 678.79 intact. B1 PP regressed
   (410 vs ~700: the TC paths engage at nt=128 where they lose) - both
   gates raised to nt/n_tok >= 256, B1 keeps the certified paths.
   Draw-quality note: three of five wall draws in this stretch came up
   degraded (mach1 B1 canary 78-125 with q4 healthy-ish) - the L40S
   pool is unstable today; trust only HOSTGATE-ok draws.
   RUNG 3 SIZING (`pp2-phase-s2`, decode-floor probe ZMMA_PROBE=1): the
   zdp MMA kernel splits 731.7/717.1 us per call into decode/expand
   254.6-265.8 us and MMA phase ~465 us. The decode floor is
   SMEM-WRITE bound (268 MB of int8 expansion per call at ~1 TB/s);
   the MMA phase runs at ~11.7 of 362 int8 TOPS = 3 percent of peak -
   30x instruction-level headroom (per tile iteration one m16n8k16
   carries ~10 overhead instructions: 2 smem A loads, 2 global record
   loads, byte_perm, 2 shfl, 4 FFMA). The warp-skip guard did not move
   the census: block runtime is dominated by hot-expert groups with
   cnt > 32, so trimming small-cnt warps is off the critical path.
   PP measurement discipline: the batched lane's single T_PP draw
   swings ~12 percent between same-container runs (rt_apply_tc read
   780.7 then 697.4 us in back-to-back runs) - judge PP by same-run
   mach1-time censuses or T-Q-T PP walls, never single draws.
   MMA-phase lever kills (`pp2-pw16-s1`, `pp2-pipe-s1`, same-container
   vs the decode floor): 16-wide pair chunks REGRESSED (+6-9 percent:
   778.5/761.5 vs 733.7/694.6) and the record-fetch software pipeline
   was neutral-to-negative (788.2/747.6) - both reverted. The decode
   floor is rock-stable across containers (264-266/253-255 us) while
   the full kernel wanders 734-788, so full-kernel deltas under ~7
   percent are unresolvable in this lane. DIAGNOSIS: neither more work
   per iteration nor prefetch helps because the kernel is
   OCCUPANCY-capped - 33.4 KB static smem (sA 16x2048 int8) allows 2
   blocks/SM = 8 warps on Ada, far too few to hide the per-iteration
   dependent chain (loads -> byte_perm -> mma -> shfl -> FFMA). The
   next real lever is smem reduction: store z-nibbles (16 KB) plus a
   4 KB sign-folded first-byte plane (the +8 overflow case blocks pure
   s4), expand to s8 in registers at A-fragment build (~12 instr via
   XOR-8 and prmt sign-extension) -> ~21 KB -> 4 blocks/SM = 16 warps.
   Before building it, print the kernel's regs/spill (mirror the
   exp_zdp_apply once_att block) to rule out a spill cap.
   RUNG 3 PARTIAL (`pp2-regs-s1`, `pp2-nib-s1`): resource print says 67
   regs zero spill (57 in the nibble form) - occupancy was the cap, not
   spills. The nibble-smem form (sZ 16x257 words + sC0 sign-folded
   first-byte plane, 21 KB -> 4 blocks/SM; s8 expansion at A-build via
   XOR-8 + s4 sign-extension; +1 pad breaks the row-bank tie) SPLITS:
   down m=2048 n=512 (tiles_y 32) improved 694-761 -> 541.16 us (-22 to
   -29 percent); gate/up m=512 n=2048 (tiles_y 128) regressed 734-788
   -> 803.44 - the ~22 expansion instructions run per (tile x chunk),
   and hot groups (cnt > 32, the runtime tail) pay them cnt/32 times.
   NET WASH per layer (2162 -> 2148 us); keep the nibble form for the
   down win. NEXT: invert the pair-chunk/tile loops (tile-outer,
   chunk-inner): expand each A tile ONCE into registers, then run all
   the tile's chunk MMAs against it - expansion becomes per-tile
   instead of per-(tile x chunk). Stage spidx for min(cnt, 256) pairs
   up front (1 KB), accumulate facc[chunks<=8][4], store after the tile
   loop; groups beyond 256 pairs take an outer wrap. This serves both
   shapes; if gate/up still trails, shape-split (expanded-int8 sA for
   n==2048 at 2 blocks/SM, nibble for n==512) is the fallback - both
   forms are in git history (9d48c4c nibble, a13e45c expanded).
   After any winning form: re-run ntcheck (token-exact receipt) before
   walls - the nibble+loop changes preserve the integer math but every
   kernel rewrite re-earns its receipt. The nibble build's receipt:
   `pp-nib-nt-s1` stream sha 34cdad2c7570 == p16goal3 (token-exact).
   LOOP INVERSION KILLED (`pp2-inv-s1`): tile-outer/chunk-inner with a
   MAXCH=8 predicated unroll read 998.15/764.64 vs the nibble form's
   803.44/541.16 - the 8x guarded chunk body pays its issue cost per
   tile even at nch=1 (regs 80, no spill; reverted, d64050e). The
   surviving best is the nibble form (9d48c4c). Remaining planned
   variants for the gate/up shape: (i) shape-split - expanded-int8 sA
   for n==2048 (2 blocks/SM but no per-tile expansion; the a13e45c
   form read 733.7 on its container) vs nibble for n==512 (541.16);
   (ii) a RUNTIME nch==1 specialization (most groups fit one chunk: no
   facc array, no chunk loop - the pre-inversion body exactly) with the
   multi-chunk path only for hot groups; (iii) cp.async double-buffer
   of the B records per tile. Same-container A/B only; ±7 percent is
   noise in this lane.
   SHAPE-SPLIT EXECUTED (template NIB, 1bb3598; `pp2-split-s1`): both
   bests land together - gate/up n=2048 expanded form 734.10 us, down
   n=512 nibble form 541.96 -> per-layer expert total 2010 us against
   the original dp4a's 2737 (-27 percent). regs 67/57, zero spill.
   Token-exact re-earned (`pp-split-nt-s1`: 34cdad2c7570 == goal3).
   Healthy wall `pp-split-wall-s1` (HOSTGATE ok): p16pp2 PP 2118.81 /
   TG 947.37 vs q4km PP 6315.02 / TG 695.18 - the TG draw reads 1.363,
   the highest decode ratio yet (PP kernels are gated off decode; the
   decode stack is unchanged-certified). PP wall draws swing 2036-2494
   across healthy hosts at near-equal builds - the same-container
   census remains the only instrument that resolves kernel deltas.
   RT_APPLY_TC SIZING (`pp2-cvt-s1`, `pp2-mchunk-s1`): the u16
   conversion is 7-11 us everywhere; the GemmEx itself runs at 175-188
   TFLOPS on the m=2048/4096 shapes but the m=8192 (qkvz) call reads
   23 TFLOPS / 68 GB/s - and m-chunking it into 4x2048 (the shape that
   runs at peak) only moved 735 -> 684 us, so it is NOT the algo pick.
   Suspects left: the qkvz bank's residency (33.5 MB persistent
   bank_get cache vs the pool-alloc'd smaller banks) or the ldc=8192
   strided fp32 C pattern. NEXT: nsysprof lane on p16pp2
   (--cuda-graph-trace=node) to see the actual kernel and memory
   behavior before touching anything else. ~500 us of PP per layer
   call hangs on this one shape. First nsys attempt (`pp2-nsys-s1`)
   exported only the API summary - the lane needs a gpukernsum report
   (nsys stats --report cuda_gpu_kern_sum on the .nsys-rep) before its
   output resolves kernels; fix the lane, rerun, then decide between
   the bank-residency and concurrency-contention hypotheses.
   COPY PROBE (`pp2-copy-s1`, GGML_MACH1_RT_TC_COPY=1, GEMM from a
   fresh pool copy of the bank): 729.45 -> 637.03 us, -13 percent -
   real but residency is NOT the 8x story. The anomaly remains open.
   INSTRUMENT VERDICT: nsys kernel tracing is broken in the Modal
   container (three attempts: `pp2-nsys-s1/-s2/-s3`; API rows export,
   kernel rows never do, with and without --cuda-graph-trace, with
   --force-export - CUPTI kernel domain dead in-container). Do not
   spend more runs on nsys. The decisive instrument is a
   chainbench micro-probe (pocs/mach1-chainbench.cu pattern): allocate
   synthetic bank/u16/C at the four production shapes, run the exact
   GemmEx calls, time with in-process CUDA events. If isolated m=8192
   runs ~200 us the production 730 is environmental (concurrency in
   the measured window); if slow it is really cuBLAS and the fix is
   cublasLt with an explicit algo search at that shape.
   PROBE VERDICT (`gemm-probe-s1`, chainbench --gemm-probe, in-process
   events, 20 reps): isolated m=8192 GemmEx runs 72.9 us / 235.6
   TFLOPS - AT PEAK, 10x under the production window. m-chunking is
   WORSE in isolation (94.1 us) and is reverted in production. All
   production shapes' isolated floors: 8192/2048 72.9, 4096/2048 37.1,
   2048/4096 41.0, 2048/2048 23.1, 2048/512 9.0, 512/2048 10.3 us.
   CONCLUSION: the census's 730 us window is concurrency/serialization
   absorbed into the timed interval, NOT the GEMM - and by extension
   other prefill census keys likely carry the same inflation. The real
   spine apply floor is ~190 us per LAYER across all shapes. The next
   PP lever is therefore NOT kernel micro-optimization but the
   launch/synchronization structure of the prefill dispatch (what the
   m=8192 window actually waits on), plus the stock-op share.
   m=8192 WINDOW: ALL HYPOTHESES KILLED, PARKED. Same-container census
   receipts: per-shape cublas handles (`pp-hmap-cen-s1`, 697.12 - no
   change), explicit 32 MB workspace (`pp-ws-cen-s1`, 698.72 - no
   change), fresh-copy bank (637, -13 percent only), m-chunking (684,
   and WORSE in isolation), concurrency knobs CONCURRENT_ALL=0 /
   MAX=0 (`pp-conc-s1` wall: 2040/2077/2058 - noise). Every other
   shape sits at its isolated floor in the same census (53.5 vs 41.0,
   49.5 vs 37.1, 19-20 vs 9-10 us), ONLY m=8192 is 10x off. The
   surface is ~90 ms of ~1000 ms B16 prefill - park it until a
   working kernel-level profiler exists; in-situ device events
   without syncs (a TIME=3 lane) would be the next instrument if
   revived. STRATEGIC NOTE: with mach1 kernels at or near isolated
   floors, prefill parity hangs on (a) the unattributed window
   inflation across the dispatch, (b) the stock-op share, (c)
   possibly running prefill ubatches under CUDA graphs (the fixed
   512-token shape is graphable; llama.cpp only graphs decode today) -
   (c) is the structural sledgehammer that erases every launch/sync
   gap regardless of cause and should be the next sized experiment.
   GRAPHSTAT RECEIPT (`pp-gstat-s2`, batched lane, TIME=4): 133 graph
   computes = 2 captures + 126 replays + 5 DIRECT. Decode replays;
   every prefill ubatch (4 at B16 + 1 at B1) executes direct - the
   prefill key never completes the 2-stable-call warmup. Decode also
   advances kv per step yet stabilizes, so the machinery already
   tolerates patchable kv offsets; something about consecutive prefill
   ubatches trips ggml_cuda_graph_update_required (candidates: the
   last-ubatch logits/inp_out_ids structure, per-ubatch mask shapes,
   or a key collision). NEXT: log the properties-changed reason at
   TIME>=4 inside update_required, find the tripping property, and
   make the prefill key warm up - in-bench ceiling is replay on
   ubatches 3-4 (half the prefill), real long-prompt serving amortizes
   far better.
   FLIP RECEIPTS (same `pp-gstat-s2` log; the TIME=4 flip logger
   already existed): exactly two property flips in the whole run -
   node 0 MACH1_EMBED_GATHER field=data (the input buffer pointer
   rotates: the build runs GGML_SCHED_MAX_COPIES=4 pipeline copies)
   and node 21 VIEW cache_r_l0(reshaped) field=ne (a conv-cache view
   whose shape follows the cache fill level across prefill ubatches).
   Each flip resets the 2-stable-call warmup. FIX CANDIDATES, in
   order: (1) pin sched copies to 1 for the mach1 lanes (stabilizes
   the input pointer; decode graphs already replay so copies gain
   nothing there); (2) make the r-cache maintenance view shape-stable
   across prefill ubatches; (3) only then consider relaxing
   update_required for data-only diffs (needs a patch-at-replay path
   the machinery lacks today - its replays are bit-identical-property
   only). Ambiguity to resolve while testing: whether one of the 2
   captures already IS a late prefill ubatch (the 5 direct evals fit
   either accounting) - add a per-phase graphstat split (prefill vs
   decode by n_tokens) before claiming any fix works.
   PER-SHAPE GRAPHSTAT (`pp-gstat-s3`, npl 16 only): decode n3966 = 1
   direct + 1 capture + 126 replays (all 128 steps accounted); prefill
   = n4267 x3 (direct, direct-after-flip, capture) + n4266 x1 (the
   odd-one-out last ubatch, direct) - the capture is taken and then
   NEVER replayed because the 4th ubatch's shape differs by one node.
   CEILING VERDICT: even fully fixed (flip kills + same-shape last
   ubatch), in-bench prefill replay covers at most 2-3 of 4 ubatches
   and saves only launch gaps between ~100 us kernels - ~8-10 percent
   of PP, NOT the parity lever. Real long prompts amortize better;
   park the graph-warmup work behind the bigger item:
   THE SYNC TRAIL: the nsys API sum shows 4124 cudaStreamSynchronize
   calls (~31 per graph eval) and 2684 cudaMemcpyAsync - the sched
   appears to SPLIT each eval ~30 ways with a host sync per boundary.
   Next instrument: run with GGML_SCHED_DEBUG=2 (sched split dump) on
   a prefill ubatch and count/name the splits - if the mach1 ops'
   supports_op/offload answers force CPU round trips or split
   boundaries at prefill, THAT is the unattributed ~170 ms/ubatch.
   ABLATION ATTRIBUTION (`pp-floor-s1`, same container, ABLATE=255
   garbage-output stopwatch): p16pp2 PP 2044 (250 ms/ubatch), ablated
   3966 (129 ms/ubatch), q4km 5819 (88 ms/ubatch). THE TWO-FRONT
   BUDGET: (a) mach1 weight stages cost 121 ms/ubatch at WALL level
   though their isolated kernel floors sum far lower - the
   serialization/dispatch inflation is real wall time, not census
   artifact; (b) the non-mach1 floor 129 ms is itself 1.47x Q4's
   ENTIRE ubatch - parity requires shrinking BOTH fronts. Candidates
   for (b): the GDN chunked delta-net prefill path, sched
   splits/copies from the mach1 arm's extra buffers, the router chain
   at nt=512. Candidate instrument for both: GGML_SCHED_DEBUG split
   dump; for (a): pairwise ablation of single stages (ABLATE bit per
   family) to see which stage's wall share exceeds its isolated floor
   most - that names the serialization point.
   FAMILY ATTRIBUTION (`pp-fam-s1`, same container, per 512-token
   ubatch): full 256 ms; ablate rt walk/apply (bit 2) -> 233 = the TC
   apply is 23 ms; ablate rt u+walk+out (7) -> 172 = the rt FAMILY is
   84 ms, so rt_u + rt_out (the SU/SV FWHT transform stages) are 61
   ms - nearly 3x the apply they wrap and never attacked at prefill;
   ablate exp walk/apply (32) -> 173 = the exp s8 MMA really is 83 ms
   wall (the census's 734/542 us per call were REAL, not inflated -
   the window inflation was spine-specific); whole exp family (120)
   -> 161 = 95 ms. Sums exceed the ABLATE=255 delta (179 vs 127) -
   ablation shifts overlap, use ranks not absolutes. NEW PP QUEUE:
   (1) exp MMA phase 83 ms - the fp16-A form. DESIGN PINNED: a full
   fp16 A tile is 16 x 2048 x 2 = 64 KB and does NOT fit shared;
   keep the nibble sZ (16 KB) + sC0 sign plane and expand to fp16 AT
   FRAGMENT BUILD through a 256-entry half2 LUT staged in shared
   (1 KB: byte value -> half2{(lo nibble)-8, (hi nibble)-8}, then
   patch the state's first half from sC0 when the fragment covers
   column 0). B comes straight from the u16 buffer the spine TC apply
   already materializes ([nt, n] fp16 global; a B col-major fragment
   is one contiguous 8-byte load per lane - no q8 records, no
   byte_perm, no du shuffles). The m16n8k16 fp16 MMA accumulates
   fp32; the epilogue scale is warp-UNIFORM per tile (zs0*gv[ti]
   only, du gone with the records) - facc[e] += c.f[e]*base. Numeric
   class: fp16 weight x fp16 activation vs int8 lattice - ulp-class,
   token receipt re-earned, KLD gate before promotion. Prereq: the
   exp path needs u16 (quantize exp_u's output to fp16 once, like
   rt_u16cvt; the exp_quant_u q8 stage is then dead in this form).
   (2) rt_u/rt_out 61 ms - ANSWERED: the prefill rt_u launches
   mach1_rt_u_kernel with grid (nt, 1, 1) - one block per token
   running the scalar butterfly FWHT (rt_out likewise). The TC-FWHT
   batched machinery (mach1_tc_fwht_apply, 16-token tiles through
   m16n8k16 GEMM passes, engaged in the decode mma16 paths) exists
   in-tree and needs a prefill dispatch branch: process nt=512 as 32
   16-token tile batches. Bitwise caution: the TC-FWHT basis order
   differs from the butterfly's float order - decode certified it
   inside its own paths; the prefill swap is a numeric-class change
   (token receipt + KLD, same lane as the other PP rungs).
   FP16-A EXPERT APPLY EXECUTED (`pp-f16-s1`,
   GGML_MACH1_EXP_APPLY_FP16, arm p16pp5 = pp4 + it; compiled and ran
   first try): same container PP p16pp4 1698.21 -> p16pp5 2151.74
   (+27 percent), TG untouched (875.0 vs 877.8). The kernel is as
   designed: nibble sZ + half sC0 sign plane + 256-entry half2 LUT
   (one lookup per A register - a byte's two nibbles are the
   register's two columns), B = two aligned u32 loads from a fp16
   copy of scr_u (mach1_rt_u16_kernel reused), m16n8k16 f16 MMA with
   warp-uniform zs0*gv[ti] scale - the q8 records, byte_perm and du
   shuffles are gone. TOKEN-EXACT (`pp-f16-nt-s1`): p16pp5 np16
   stream sha 34cdad2c7570 == goal3 under the fp16 lattice. p16pp5
   is the composed PP candidate. KLD pass still owed before
   promotion (token-level receipts only across the PP stack).
   HEALTHY WALL (`pp5-wall-s1`, HOSTGATE ok): p16pp5 PP 2257.71 /
   q4km 6290.16 B16 - the campaign's best healthy reading (start
   1574, gap 4.03x -> 2.79x). B1 PP dipped to 386 (TC_NT=512 also
   engages the B1 nt=128 prefill where the TC stages lose; give
   TC_NT a lower floor of 256 the way the other PP paths gate when
   promoting). TG on this draw 838/674 - a low candidate draw;
   decode certification is unchanged (pp5's decode path == goal3).
   PP5 FAMILY SHARES (`pp5-fam-s1`, same container, per ubatch; a
   slower host - pp5 full 255 ms, q4 97): rt TC apply ~42, rt u+out
   ~40 (down from 61 - the TC transforms are working), exp fp16
   apply 83 (SAME as the s8 form's 83: per-call ~690 us is
   unchanged - the +27 percent PP came from deleting exp_quant_u
   and overlap, not the kernel), exp family 99, all-mach1 floor 122.
   The expert apply kernel remains the single wall: its decode
   phase (~255 us) plus an MMA phase far from fp16 peak. The
   structural next ideas: (i) optimize the fp16 MMA phase (ldmatrix
   the B tiles, double-buffer sZ decode, k-major two-tile MMAs);
   (ii) grouped-GEMM form: decode each group's rows once to a
   transient global fp16 tile and run per-group cuBLAS batched
   GEMMs (ragged cnt on device is the blocker cuBLAS needs host
   sizes for; prefill has no graphs so ONE d2h size fetch per call
   is legal - price it); (iii) shrink the decode phase (it is
   smem-write bound; the fp16 form could decode straight to
   fragment registers for cnt <= 32 groups).
   PLATEAU RECEIPT (`pp-f16-cen-s1`): the fp16 form censuses at
   797.4/608.2 us per call (gate-up/down) - the same band as the s8
   nibble (803/541) and s8 expanded (734/695) forms, B-prefetch
   included. THREE kernel forms land in one 600-800 us band while
   the contained GEMM's isolated floor is ~73 us: the fused
   per-group apply SHAPE is the wall (8-32k blocks x O(n) decode +
   low-warp MMA), not any single implementation choice. Do not
   iterate more forms inside this shape. The remaining prefill
   moves, in expected-value order: (1) the grouped-GEMM restructure
   (transient fp16 banks for hot groups + cublasGemmGroupedBatchedEx,
   one legal D2H size fetch per call; ~100 MB transient, opt-in);
   (2) the non-mach1 floor (122 vs q4's 97 whole-ubatch on the same
   host); (3) the rt TC apply's 42 ms (its GEMMs are at peak - the
   share is the u16cvt + window structure around them).
   UB2048 RECEIPT (`pp-ub2k-s1`, same container, per-arm -ub override
   in the batched lane): raising n_ubatch 512 -> 2048 lifts BOTH
   stacks ~50 percent - p16pp6 2066.2 -> p16pp7 3165.1, q4km 5737.8
   -> q4km2k 8564.5; the ratio barely moves (2.78x -> 2.71x). The
   asymmetric-amortization hypothesis (our per-ubatch weight-decode
   tax vs q4's compute-bound prefill) is FALSIFIED: q4 was NOT at
   peak either. The shared ~40 ms/ubatch that vanishes with fewer
   evals is per-eval machinery (sched splits/syncs, input copies,
   graph-eval overhead) - this also explains most of the earlier
   non-mach1 floor premium (122 vs 97): it is per-eval cost, not
   stock-op compute. IMPLICATIONS: (a) any parity claim must state
   its n_ubatch basis; at matched ub2048 the gap is 2.71x; (b) the
   per-eval machinery is a shared tax - reducing OUR evals' overhead
   below q4's is not a lever (same code); (c) the remaining honest
   levers stay kernel-side: the cold-group tail, then the decode-tax
   floor itself. TG unchanged in all four arms (ub is prefill-only
   here). Token receipt for pp7 pending an ntcheck-lane ub plumb
   (the kernels are pp6's, already token-exact; ub is a stock llama
   serving parameter).
   GROUPED-GEMM EXECUTED (`gg-s6`, GGML_MACH1_EXP_APPLY_GG, arm
   p16pp6 = pp5 + it). Structure: ONE D2H fetch of the 256 group
   counts per call picks hot groups (cnt >= GGML_MACH1_GG_MIN); those
   decode once to transient [m, n] fp16 banks with zs0*gv FOLDED IN
   (gv varies per k-tile, so it must live in the bank, not an
   epilogue), pair rows gather to a padded fp16 B (bucket ladder
   32/48/64/96/... so every cuBLAS handle only ever sees one shape),
   C scatters back by pair list; cold groups keep the fused fp16
   kernel through a hotmask early-return (nullptr = old behavior
   bit-identical). The D2H sync is illegal mid-capture, so graph
   gating (check_compability) keeps EXP_MM cgraphs with n_tok >= 256
   direct when the env is set - decode graphs untouched. THREE
   pricing lessons: (1) cublasGemmGroupedBatchedEx REJECTS 16F inputs
   (status 15) on CUDA 12.8 - the ragged-size API is fp32/fp64 only;
   (2) a per-group GemmEx loop is host-dispatch-bound (~6-10 us a
   call, 40-86 hot groups at GG_MIN=16) and LOSES 5 percent PP; the
   shipping form runs one cublasGemmBatchedEx per distinct pad bucket
   (~3-8 dispatches) with the cold kernel queued FIRST so its GPU
   time hides them; (3) the hot threshold wants to be HIGH: mean cnt
   is only 16 (P=4096 over 256 groups), GG_MIN=16 banks 40-90 groups
   for pairs the fused kernel absorbs cheaper - the 32-128 sweep is
   flat within noise and 64 is the default (n_hot ~10-14 holding
   25-50 percent of pairs). Same container PP: p16pp5 2018.17 ->
   p16pp6 2232.93 (+10.6 percent), q4km 5711.45 (gap 2.83x -> 2.56x),
   TG untouched (913.1 vs 906.5). TOKEN-EXACT (`gg-nt-s3`): p16pp6
   np16 stream sha 34cdad2c7570 == goal3 - the fp16 zs0*gv bank fold
   and cuBLAS accumulation order change no token on this draw (KLD
   still owed, same lane as the rest of the PP stack). CENSUS
   (`gg-cen-s3`, sync-inflated per call): exp_gg 707.6/666.1 us
   (gate-up/down) vs the fused 774.4/603.0, inside: cold fused 423/414
   (~200 cold groups' per-group decode floor - STILL the wall), gemm
   140/107, dec 57/42, sync 12/12 - the census serializes what
   production overlaps, so the wall win exceeds the census delta.
   Remaining exp rungs: the fused shape's per-group decode floor for
   the cold tail (a cnt-aware cold kernel, or banking the tail), then
   the non-mach1 floor.
   COLD-TAIL RECEIPTS (`ct-prof-s1`..`ct-nt-s8`, each rung same
   container). PROBE (`ct-prof-s1`; ZMMA_PROBE=1 plumbed into the fp16
   kernel, same early-return the s8 form has): the cold fused kernel is
   92/85 percent DECODE (down 305 of 330 us, gate-up 332 of 391) - the
   MMA phase is 25-59 us, so every MMA-side restructure (the cnt<=8
   tiny-kernel idea included) is capped below 15 percent and dead.
   OCCUPANCY KILL (`ct-s2`, `ct-cen-s3`): a launch_bounds MINB 3-6 cold
   instantiation with a strength-reduced decode loop measures occ=8
   blocks/SM and 48-56 regs ALREADY on H200 (the "3 blocks/SM" note was
   Ada smem math) and runs 2-8 percent SLOWER at kernel level; the
   end-to-end sweep is flat - reverted. The decode floor is
   gather/issue-bound at full occupancy, not occupancy-starved.
   GG FORK EXECUTED (GGML_MACH1_GG_FORK, arm p16pp6f): the cold kernel
   moves to a side stream and overlaps the hot banks' dec/gather/GEMM/
   scatter chain (disjoint scr_v rows; join event before the epilogue).
   GPU-level real: census (`ct-cen-s7`) exp_gg 1277.1 -> 1145.5 us
   (down, -10.3 percent), 1001.4 -> 947.3 (gate-up, -5.4), the GEMM
   window absorbing the cold tail. WALL-NEUTRAL: the cpu=16 llama-bench
   lane (`ct-s6`, 3 interleaved rounds, +-0.2 percent spread) reads
   pp6 2999.4 vs pp6f 2997.6 (q4km 6163.7 - gap 2.06x on this healthy
   host); at ub512 the prefill wall does not price exp GPU time,
   consistent with the UB2048 shared-tax receipt. Knob kept, opt-in,
   unpromoted. MORE KILLS: GG_MIN 32/96 under the fork LOSE (1853/1694
   vs 1999 same container, `ct-s4`) - banking the middle cannot win,
   the bank decode runs 2.6-3.5 us/group vs the fused cold's 1.7-2.0
   and the GEMM adds on top. MEASUREMENT NOTE (`ct-s5`): the npp=128
   npl=16 batched race carries ~20 percent intra-container spread (ABAB
   1670/1832 vs 1524/1884) - use the cpu=16 bench lane for wall
   verdicts. TOKEN-EXACT (`ct-nt-s8`): p16pp6f np16 stream sha
   34cdad2c7570 == goal3, intra-batch AGREE; greedy smoke pp6 == pp6f.
   VERDICT: the cold slice sits at its decode floor and the ub512 wall
   is per-eval-machinery-bound - further exp kernel work should be
   priced on the ub2048 lane (where GPU time surfaces) or aimed at the
   per-eval machinery itself.
   PP KLD GATE PASS (`kld-pp-s1`, prefill-shaped: gates mode grew a ub
   knob, ub=512 so every nt>=256 lane engages; TC-apply marker fired):
   m1 mean KLD 0.360835 +- 0.005286 / same-top-1 75.680 percent, p16pp6
   0.360928 +- 0.005275 / 75.748 - delta inside 0.02 sigma. The whole
   PP stack (TC apply + s8/fp16 MMA + GG) is quality-neutral vs the m1
   baseline; the owed gate for PP promotion is cleared.
   UB2048 LANE RECEIPT (`ub2k-s1`, cpu=16 bench lane, healthy host -
   mach1 B1 147-150): -b/-ub 2048 via BENCH_UB now honored in the bench
   lane. p16pp7 5510/5751/5809, p16pp7f 5821/5782/5852 (warm rounds:
   fork +0.5-0.7 percent, now WALL-POSITIVE - GPU time prices at
   ub2048), q4km2k 8740/8752/8793, q4km(default ub512) 6138/6142/6129.
   Greedy smoke pp7 == pp7f MATCH. READING: mach1 gains +94 percent
   from ub512->2048 (2999->5818) vs q4's +43 (6136->8762) - the
   per-eval tax lands ~2x harder on the mach1 graph and amortizes
   away with ubatch size. Position: 0.95x of q4's DEFAULT config (5
   percent from parity-as-shipped), 0.66x matched-ub2048 (gap 1.51x,
   from 2.05x at ub512 and 4.03x at campaign start).
   CENSUS DIFF AT UB2048 (`cen-ub-s1`/`cen-ub-s2`, prefill2k phase; a
   census env-merge TypeError on arms that pin TIME was fixed en
   route): per 2048-token prefill, mach1 exp apply ~350 ms (gate/up
   2814 us x80 + down 3132 us x40) vs q4 mul_mat_id ~43 ms - the 8x
   is the known trellis-decode plateau, amortization is the lever.
   rt spine ~86 ms vs q4 spine ~38. TWO STOCK-OP ANOMALIES: (a)
   MACH1_EMBED_GATHER 16 ms/prefill - the loader parks
   token_embd.m1_codes/m1_lut in the input layer's HOST buffer and the
   kernel zero-copies both tables over PCIe every call; fixed by a
   one-time per-device mirror (tc_htab idiom, bit-identical bytes,
   GGML_MACH1_EMBED_DEV=0 opt-out, control arm p16pp7fe0). (b) the GDN
   small projections MUL_MAT s0=2048x32 d=32x2048 read 752 us/call on
   mach1 vs 128 on q4 (same op+shape, ~37 ms/prefill) AND ~47 us x60
   calls per DECODE step - one fix may pay on both fronts; census keys
   now carry src dtypes + dst ne[2] to attribute the dispatch family.
   UB4096 RUNG (`ub4k-s2`, cpu=16 bench lane, p=4096 held fixed via
   BENCH_P so ub is the only mover; -b now rides -ub): p16pp8f
   7098/7188/7175 (mean 7153) vs p16pp7fp4 5932/5955/5913 (mean 5933)
   -> ub2048->4096 is +20.6 percent for mach1; q4km4k 9234 vs q4km2kp4
   9148 -> +0.9 for q4 (saturated). q4km default 6170. POSITION:
   mach1 @ ub4096 = 1.16x q4's DEFAULT-config prefill - the
   parity-as-shipped bar is CLEARED, token-exact (greedy smoke MATCH).
   Matched-best 0.775x (gap 1.29x).
   UB8192 RUNG (`ub8k-s1`, p=8192): p16pp9f 7434/7450/7480 (mean
   7454) vs p16pp8fp8 7071 -> +5.4 percent (amortization tapering);
   q4km8k 8666 REGRESSES vs q4km4kp8 9628 (q4's optimum is ub4096).
   POSITION: 1.21x q4-default, 0.86x matched-ub8192, 0.774x
   matched-best - the matched-best gap is stable at ~1.29x across
   both rungs, so the rest of parity is GPU structure, not batching.
   TYPED CENSUS (`cen-ub-s3`): the GDN beta/alpha projections and
   router gate are t0=f32 on mach1 (q4 stores them q4_K) and fall to
   the sgemm fallback at prefill - 673 us/call s0=2048x32 d=32x2048
   (~40 ms/2048-token ubatch over 60 calls) + router 327 us x40.
   EMBED MIRROR receipts: the census EMBED_GATHER rows are dominated
   by the one-time upload (call 1 ~30 ms, call 2 fast) - steady-state
   gather is sub-ms; wall A/B (`emb-s1`) pp7f 5756 vs pp7fe0 5705
   (+0.9 percent, round-noisy, bit-identical) - kept, default ON.
   SKINNY MM LANE built (GGML_MACH1_SKINNY_MM=1, arms p16pp7fs/9fs):
   F32xF32 mul_mats with <=256 out rows at nt>=256 route through a
   one-time f16 weight mirror + per-call b16 convert + per-shape
   GemmEx handle (32F acc). Decode untouched by the nt gate.
   RECEIPT (`skn-s1`): WALL-NEUTRAL to +1 - pp7fs 5840 vs pp7f 5773,
   pp9fs 7449 vs pp9f 7467, greedy smoke MATCH all arms. The census's
   40 ms/prefill for this family did not exist on the wall. Kept as
   an opt-in knob, NOT promoted into the composed stack. THIRD
   census-inflation instance (m=8192 GEMM, embed 16 ms -> +0.9, now
   skinny 40 ms -> +1): graphs-off event windows absorb waits and
   launch latency - a census row is a HYPOTHESIS, only the wall A/B
   is a receipt.
   TOPK-MOE GATE NAMED (`topk-s1`, diag splits the three fuse
   conditions): DECODE gate_ok 4842 / gate_memrange 78 - the stock
   fusion already engages 98.4 percent at decode, so the ledger's
   "router-chain fuse ~0.4 ms/step" decode lever is DEAD (the goal2
   note "counter fires but the chain still executes" was the
   precheck-vs-engage distinction, now measured). PREFILL is the real
   finding: ggml_cuda_check_fusion_memory_ranges rejects 47 percent
   of layers at ub512 (320/680) and 22 percent at ub2048 (80/440),
   and each reject runs the 10-op chain per-op (~12 ms per
   2048-token ubatch). Allocator-placement luck; parked as a small
   prefill item. DECODE 1.50x consequence: with the router lever
   dead and skinny decode-neutral, the only certified in-hand lever
   remains the standard-output runtime (+1.31 ms/step) - still
   BLOCKED on the user pushing llm-compression@75a3b09b2. Next
   decode probe: graphs-on nsys B16 kernel diff vs q4 on L40S.
   B16 TYPED CENSUS OVERTURNS THE B1 EXONERATION (`b16cen-s1`,
   llama-batched-bench npl=16 L40S): at B16 the memrange gate rejects
   the topk-moe fusion for ~100 percent of layer-steps on BOTH stacks
   (mach1 5192/5320, q4 5320/5320) - the B1 llama-bench probe passed
   because of the nrows==1 exception inside
   ggml_cuda_check_fusion_memory_ranges. Every B16 layer-step pays the
   10-op router chain (~1.2 ms/step, both stacks).
   TOPK FIX EXECUTED (GGML_MACH1_TOPK_FIX=1, lane p16goal4 = goal3 +
   fix; commit 16574f6): the only non-elided src the fused outputs can
   overlap is logits (live tensors never overlap), so a pool-scratch
   copy of logits removes the intra-kernel hazard and the stock fused
   kernel engages - 2 launches per layer instead of 10. KLD-CLASS
   (new engagement sites shift ulps): goal4 is a new candidate lane,
   NOT an np16 sibling of goal3.
   RECEIPTS: engagement (`g4cen-s1`) gate_fixed 5192 + gate_ok 128 =
   all 5320; ARGSORT/SOFT_MAX/SUM_ROWS rows GONE from the census.
   KLD GATE PASS (`kld-g4-s1`): goal4 0.360773 +- 0.005281 / top-1
   75.778 and pp6k 0.360426 / 75.692 vs m1 0.360835 / 75.680 -
   quality-neutral on both stacks. WALL: degraded-host sandwich
   (`g4-b16-s1`, goal3 681 / goal4 852 / j0 666) shows +25-28 percent
   - launch-queue relief where the host glue is the bottleneck;
   HEALTHY-host HOSTGATE-ok race (`g4-wall-s1`, q4 B1 166.1, control
   161.0): goal3 854.96 / goal4 875.77 / q4 674.74 / j0 834.42 ->
   +3.7 percent vs the goal3 flanks, goal4/q4 = 1.298 same-container
   (flanks 1.237-1.267). PP at ub512 B16 lane +1.5 percent (the
   prefill rejects fixed too). TOKEN-EXACT ON TOP (`g4-nt-s1`):
   goal4 np16 stream sha 34cdad2c7570 == goal3, intra-batch AGREE -
   the ulp shifts never moved a greedy token on the probe stream.
   PROJECTION vs certified goal3 1.334: goal4 ~1.38x decode;
   remaining ~8.5 percent to the 1.50 goal is the certified size of
   the user-blocked standard-output lever.
   SECOND HEALTHY DRAW (`g4-wall-s4`, HOSTGATE ok q4 165.8 control
   161.8; s2/s3 discarded DEGRADED by predeclared gate): goal3 879.47
   / goal4 888.41 / q4 675.29 / j0 874.30 -> goal4 +1.3 vs flanks,
   goal4/q4 1.3156. CERTIFICATION ACROSS TWO HEALTHY DRAWS: fix
   value +1.3/+3.7 vs same-container goal3 flanks (mean ~+2.5),
   goal4/q4 1.298/1.316; degraded-host value +25-28 (launch-queue
   relief - a fleet-median win too). goal4 ~1.37x projected decode.
   COMPOSED PP (`pp9ff-s1`, H200 bench lane): pp9ff (= pp9f + fix)
   7533/7645/7628 mean 7602 vs pp9f 7559 (+0.6 at ub8192 - rejects
   are rare there) vs q4km default 6171 -> NEW COMPOSED H200
   POSITION 1.232x q4-default. Greedy smoke pp9ff DIFFERS vs pp9f
   as expected for the KLD-class lane (gate passed, kld-g4-s1).
   FULL COMPOSITION p16pp10 = pp9ff + GG_I8 + GG_COLD_S8
   (`pp10-s1` H200 / `pp10-l40s-s1` L40S, same-container):
   H200 pp10 7740/7854/7862 mean 7819 vs pp9ff 7391 (+5.8) vs q4km
   default 6140 / q4km4k 8935 -> H200 POSITION 1.273x q4-default,
   0.875x matched-best. Greedy smoke MATCH vs pp9ff.
   L40S: pp10@ub8192 4569 vs pp8fb8@ub4096 4618 - ub8192 still
   regresses on Ada, p16pp8fb8 stays the L40S candidate; this
   container reads pp8fb8/q4km-default = 4618/6734 = 0.686 (the
   agent's three containers read 0.700-0.703 - host variance, both
   recorded). Ada prefill parity NOT met; the remaining Ada gap is
   expert trellis decode + Ada's GEMM rates, both at receipted
   plateaus. CAMPAIGN TOPLINE at this checkpoint: PP vs q4-default
   H200 1.273x / L40S 0.69-0.70x; PP matched-best H200 0.875x;
   decode (L40S B16) goal4 ~1.37x with the remaining ~9 percent
   being the certified, user-blocked standard-output lever.
   FULL-STACK KLD GATE PASS (`kld-all-s1`, ub512): p16pp6all (fork +
   topk fix + GG_I8 + GG_COLD_S8 composed) 0.360787 +- 0.005269 /
   top-1 75.784 vs m1 0.360835 / 75.680 - the whole composed PP
   stack is quality-neutral in one measurement.
   GOAL4 T-Q-T CERTIFICATION (`g4-cert-s1`, HOSTGATE ok q4 166.9
   control 162.2, goal2-wall protocol): j4a 880.31 / q4km 676.13 /
   j4b 888.59 -> goal4/q4 = 1.302 / 1.314. Three healthy-gated
   draws now span 1.298-1.316; these hosts' mach1 band sits below
   the containers that certified goal3 at 1.3378/1.3343 (the
   above-canary "bimodal glue" spread), and goal4's same-container
   gain vs goal3 flanks stays +1.3-3.7. HONEST DECODE POSITION:
   ~1.30-1.38x by host, with the certified +10 percent
   standard-output lever still blocked on the user-side
   llm-compression push. UB512-DEFAULT PP note (`ub512-all-s1`,
   L40S): pp6all 2759 vs pp6f 2709 - neutral within the ±6 noise;
   at ub512 nearly all groups are cold so the i8 hot lane barely
   engages.
   BATCH-WIDTH PROBE (`g4-b32-s1`, degraded host - shape receipt
   only, wall discarded per gate): mach1 TG COLLAPSES above B16
   (838-862 -> B24 443 -> B32 300) while q4 scales (667 -> 821 ->
   886). The exact-nt16 fused decode stack (mega QONCE tw=8,
   rt_imma8 split-K nt16, head_mma16, ZDP regions) disengages
   wholesale above 16 and the generic fallback eats the step.
   STRATEGIC READ: mach1's step is decode-dominated and fixed per
   step, so an efficient nt32 stack projects B32 TG well above
   1.5x q4's ~886 ceiling - batch-width generalization is the
   remaining in-tree road to the decode 1.50x goal (pure runtime,
   no weight changes). nt32 extension campaign delegated: phase 1
   attribution of every nt16 gate + B32 fallback census, phase 2
   template-generalization biggest-first (expert mega/ZDP, rt
   imma8 split-K, head, GDN/qkv siblings), additive-only with B16
   paths untouched, np32 intra-batch + wall gates per rung.
   NT32 PHASE 1 - GATE ATTRIBUTION (source map, this session): the
   exact-nt16/exact-B16 admissions and their nt=24/32 fallbacks are
   (1) expert mega QONCE - n_tok==16 pin + MEGA_NT clamp 16 + the
   128-pair counter bank (mach1.cu exp_ffn_fuse); fallback = unfused
   EXP_MM chain (exp_group/u/walk/out per op). (2) rt_mma16/rt_imma8
   spine + split-K4 - nt==16 pins (rt_mm dispatch); fallback = WALK_TT
   owns nt in [2,16] only, so nt>=17 takes the transient dense
   decode-bank + apply round trip per op per step. (3) qkvz pair /
   qkv trio / shexp gate-up sibling batching - b_nt==16 pins in the
   matchers and batch exec; fallback = per-op rt_mm, then (2).
   (4) head_mma16 - nt==16 pin; fallback = generic head_mm kernel.
   (5) GDN_SGF - BUILDER-side n_seqs==16 pin (src/models/mach1.cpp),
   fallback = build_rs get_rows state copies; GDN prep fuse ns<=16,
   fallback = 6 launches x 30 layers. (6) TC_NT=16 env caps the TC
   u/out transforms (env-only). FORK_NT/FORK_Z/GDN_NT cap at 4 and
   are not part of the B16->B32 delta.
   NT32 PHASE 1 - B32 TYPED CENSUS (`nt32-cen-s1`, L40S, TIME=3
   graphs-off, one batch width per process; HOSTGATE context arm
   read 89.7 B1 = DEGRADED host, so census totals are shapes, not
   walls): B16 census total 2140.8 ms (fused regions untimed by the
   per-op timer), B32 10205.5 ms. Top B32 fallback families
   (whole-run ms across 128 decode steps): expert region unfused
   EXP_MM d=512x8x32 3142 + d=2048x8x32 1132 = 4275 (~33 ms/step);
   GDN state GET_ROWS 524288x32 704 (~5.5 ms/step, SGF off above 16);
   MACH1_HEAD_MM 697 (5.4 ms/step vs 0.76 at B16); dense rt shapes
   (8192x32 610, 2048x1x32 295, 512x32 277, 4096x32 267, 2048x32
   142+84+43) ~1.7 s total. KILL: env-only WALK_TT=32+TC_NT=32 flank
   (`p16g4t32c`) REGRESSES B32 (TG 270.6 vs 280.7 in-census; m8192
   row 610 -> 923 ms) - the TT walk re-decodes per 4-token chunk and
   loses even to the dense round trip at nt32 on Ada. The TT route to
   32 is dead; the spine/mega/SGF/head extensions are the road.
   NT32 PHASE 2 - EXTENSIONS LANDED (commit f3117ff, master switch
   GGML_MACH1_NT32=1 default off, nt==16 branches byte-identical):
   (a) expert mega QONCE admits n_tok in (16,32] through a widened
   PCAP=256 counter layout (new template param; PCAP=128 constants
   static_asserted equal to the certified macros; counter bank
   400 -> 1040 ints; wide instantiations for plain QONCE dnct 32/128
   only, occupancy slots 16/17). (b) rt_mma16/rt_imma8 per-op spine
   serves nt==32 as two 16-token halves (records/scr_v rows are
   token-major, so halves are pointer offsets; split-K4 stays
   nt16-only). (c) qkvz pair + qkv trio + shexp gate-up sibling
   batching admit b_nt==32 (u/out batch grids are z-general, spines
   per 16-half, mixed grids generalized 16 -> b_nt). (d) head_mma16
   serves nt==32 as two 16-token halves. (e) GDN_SGF admits n_seqs
   in (16,32] (indexed conv/delta kernels are seq-general), GDN prep
   ns<=32. Candidate arm p16nt32 = goal4 + NT32=1 + MEGA_NT=32 +
   TC_NT=32 (WALK_TT stays 16 per the census kill).
   NT32 RUNG 1 WALL (`nt32-wall-s1`, L40S same-container, HOSTGATE
   ok q4km 168.6 control 160.7): B1/B16/B24/B32 TG -
   goal4 control 160.7 / 817.4 / 441.4 / 300.2;
   q4km 168.6 / 673.1 / 824.4 / 887.1;
   p16nt32 156.2 / 831.8 / 709.1 / 471.8;
   p16n32b dup 146.8 / 857.2 / 708.2 / 470.3;
   p16g4dm256 156.2 / 816.9 / 440.6 / 352.3.
   VERDICTS: candidate B32 300 -> 471.8/470.3 (+57, reproducible),
   B24 441 -> 709 (+61); B16 flanks 831.8/857.2 vs control 817.4 -
   NOT regressed. Bracket kill: DENSE_MIN=256 group-dense apply
   reaches only 352 at B32 (and cannot engage at B24, P=192<256) -
   the wide QONCE mega beats the group-dense chain; dm256 not
   composed. POSITION: B32 nt32/q4 = 0.531 (B24 0.860, B16 1.236
   this container). ANOMALY for the next rung: per-token cost B16
   1.20 -> B24 1.41 -> B32 2.12 ms/tok - the 24->32 step is
   superlinear even though MORE families are extended at 32; a
   B24/B32 typed census pair on the candidate stack (nt32-cen-s2)
   is the naming instrument.
   NT32 CORRECTNESS (`nt32-nt-s1`, L40S llama-batched temp 0):
   p16goal4 np16 AGREE sha 34cdad2c7570; p16goal4 np32 (the
   fallback reference) AGREE sha 34cdad2c7570; p16nt32 np16 AGREE
   sha 34cdad2c7570 (byte-level B16 non-regression); p16nt32 np32
   AGREE sha 34cdad2c7570 - the nt32 stack is TOKEN-EXACT vs both
   the np32 fallback reference and the certified goal4 stream.
   (KLD gates lane is ub=1 where no nt32 path engages; the honest
   quality lane is exactly this np32 agreement + token-exact match.)
   NT32 SUPERLINEARITY NAMED PARTLY (`nt32-cen-s2` typed census +
   `nt32-time-s1` per-kernel stopwatch, both L40S): at B32 on the
   candidate every targeted fallback family is GONE from the census
   (no d=512x8x32/2048x8x32 EXP_MM, no 524288-row GET_ROWS, head
   5.4 -> 1.5 ms/step, spine shapes per-token flat vs B16). The wide
   mega itself scales SUBLINEARLY per token: exp_mega 167.0 us
   (nt16) -> 229.7 (nt24) -> 292.6 (nt32) per launch - per-token
   10.4 -> 9.6 -> 9.1 us. exp_mega slot=17 (wide dnct=32) probes
   regs=64 spill=48B 1 blk/SM, same as slot 15. Remaining puzzle:
   summed kernel time accounts for ~26 ms/step at B32 vs the ~68
   ms/step wall - an unattributed host/gap share that grows ~4 ->
   ~12 -> ~42 ms/step across 16/24/32 while the per-step launch
   count FALLS (more fusion at 32). Family bisect run
   (`nt32-bis-s1`, candidate minus one family per arm at B32) is
   the next instrument; stage-2 in-kernel 32-wide forms are NOT the
   bottleneck until this gap is attributed.
   NT32 FAMILY BISECT (`nt32-bis-s1`, HOSTGATE ok q4 167.3 control
   155.0, B32 TG): full candidate 470.7; minus mega-wide 355.2
   (mega worth +115); minus sibling batching 468.3 (NEUTRAL - the
   sibling-32 structure only matches the per-op spine halves it
   replaces); minus spine (RT_MMA16=0) 431.9 (+39); minus SGF 434.9
   (+36); minus head 446.9 (+24); q4km 886.2. No family regresses -
   the B32 gap is NOT owned by any single extension.
   KV-PRESSURE KILL (`nt32-kv-s1`, HOSTGATE ok): B32 candidate at
   c=8192 470.9 vs c=16384 471.4 (q4 886.1/882.2) - the exact-fill
   c=8192 hypothesis for the B32 host share is dead.
   NT32 RUNG 1 POSITION (certified, three healthy-host same-container
   draws): B32 mach1 471.8/470.3/470.7 vs q4 886-887 = 0.531x
   (control goal4 was 0.338); B24 709 = 0.860x (control 0.535);
   B16 unregressed (831.8/857.2 flanks, np16 stream sha
   34cdad2c7570 byte-identical), B1 unaffected. The bisect is
   ADDITIVE: 67.9 ms/step full + mega 22 + spine 6.2 + SGF 5.7 +
   head 3.7 + sibling 0.4 = 105.9 ~= control 106.7 - no family
   interaction. WHAT REMAINS: (1) a fixed ~30 ms/step B32-specific
   base present in EVERY mach1 config at B32 (control and candidate,
   graphs on or off, c=8192 or 16384) and ABSENT at B24 - it is not
   any extension, not the mega (167->230->293 us sublinear), not KV
   pressure; L40S-container nsys records no kernel data (dead
   instrument on Ada; the H200 nsys lane works and this model runs
   there - profile npl 24 vs 32 on H200 to name the share, or
   bracket llama_decode with host/GPU events). Removing that base
   projects B32 ~840-860 = ~0.95x q4. (2) The road from 0.95x to
   1.50x is true batch amortization of the weight decode: the mega
   is per-(token,slot)-pair and the spine serves 32 as 2x16 halves,
   so per-token decode cost is flat at the B16 level (1.20 ms/tok);
   1.50x needs decode-once-per-step forms - in-kernel NTOK=32
   A-fragment spine (b-frag decode once, MAC both halves; smem
   17.4 KB fits, watch launch_bounds(128,6) on the cp.async form)
   and expert-group decode dedup inside the mega (tokens sharing an
   expert re-decode today; ~1.6x decode dedup available at B32).
   NEXT SESSION RESUME: (a) attribute the ~30 ms/step B32 base
   (H200 nsys npl 24/32, or host/GPU event bracket in
   llama-batched-bench); (b) land NTOK=32 in-kernel spine forms
   env-gated; (c) mega group-dedup sketch. All nt32 work is behind
   GGML_MACH1_NT32=1 (default off), arm p16nt32; nt16 paths
   byte-identical (np16 sha receipt).
7b. HANDOFF - B16 DECODE BUDGET AND THE CROSS-BATCH PICTURE
   (2026-08-17, hand-off to a fresh agent; read this block first).
   B16 DECODE ATTRIBUTION (`g4ab-s1`, L40S batched npl=1,16, graphs
   ON, HOSTGATE ok q4km=166.2 control=158.5; ms/step = T_TG/128):
     goal4 control      17.77 ms/step (S_TG 900.21)
     ABLATE=255 (all mach1 weight-decode off)  8.65 (1850.31)
     ABLATE=120 (expert mega off)             12.53 (1276.68)
     q4km                                     23.70 ( 675.27)
   => expert mega 5.24 ms/step (29 percent), rest-of-mach1 ~3.88 by
   subtraction, SHARED NON-MACH1 FLOOR 8.65 (49 percent of the step).
   The 1.50x bar needs 15.79 ms/step, i.e. -1.98 ms from 17.77.
   CAUTION, the two other ablation arms are NOT clean subtractions:
   ABLATE=7 (rt spine) reads 727.71 and ABLATE=128 (head) 541.75,
   both SLOWER than the 900.21 control - those bits drop the fused
   path to a generic fallback instead of removing work. Only 255 and
   120 subtract cleanly; do not quote a7/a128 as family shares.
   SIZING NOTE for the next lever: the mach1 expert phase moves
   ~2.7 GB/step of compressed bytes in 5.24 ms (~510 GB/s, ~60
   percent of L40S peak) - close to a bandwidth floor. The spine's
   ~3.88 ms moves only ~640 MB/step (~165 GB/s) - NOT bandwidth
   bound, so the trellis-walk cost there is the compute headroom
   worth attacking first. A decoded-weight VRAM cache was sketched
   and REJECTED on arithmetic before building: dense-spine caching
   raises per-step bytes (638 MB compressed -> ~3 GB int8) and
   expert caching needs ~12.6 GiB to reach a 40 percent hit rate
   under near-uniform routing, which would erase the 7.3-vs-20.2
   GiB memory story. Do not spend a rung there.
   *** CROSS-BATCH REALITY (the decode win is a B16 PEAK, not a
   curve) *** - the user asked for speedups across batch sizes and
   the honest profile today is:
     B1  ~0.95x q4 (mach1 158.5 vs q4 166.2, the g4ab-s1 HOSTGATE
         canary; mach1 LOSES at single-stream - launch/latency
         bound, the mach1 graph replays more nodes per step)
     B16 1.30-1.38x by host (certified, goal4)
     B24 0.860x with nt32 (control 0.535)
     B32 0.531x with nt32 (control 0.338)
   SWEEP LANDED (`sweep-s3`, L40S, HOSTGATE ok q4km=166.0
   control=163.0; two earlier draws discarded DEGRADED). TG tok/s,
   goal4 / nt32 / q4km, and best-mach1 over q4:
     B1   162.98 / 164.89 / 166.01  = 0.993x
     B2   228.64 / 250.33 / 272.52  = 0.919x
     B4   356.52 / 356.26 / 405.32  = 0.880x
     B8   413.59 / 417.54 / 558.43  = 0.748x  <- worst interior
     B16  907.17 / 918.59 / 679.06  = 1.353x  <- the ONLY win
     B24  452.37 / 744.66 / 824.23  = 0.903x
     B32  301.15 / 471.59 / 886.92  = 0.532x
   *** THE HEADLINE: the decode win is a SPIKE AT B16, not a curve.
   mach1 is BEHIND q4 at every other batch size measured. *** The
   H200 draw (`sweep-h-s2`, HOSTGATE degraded q4km=228.6, shape only)
   reproduces the spike: B16 1030 vs 857 = 1.20x, B8 410 vs 778 =
   0.53x, B32 367 vs 1225 = 0.30x.
   *** THE SAME SWEEP CONVICTS PREFILL TOO - READ THIS BEFORE
   QUOTING ANY PARITY NUMBER. *** `sweep-s3` runs npp=128, i.e.
   SHORT prompts, and the S_PP column (goal4 vs q4km, same
   container) reads:
     B1  647.7 / 1142.3 = 0.567x    B2   1297.4 / 3700.1 = 0.351x
     B4  1798.2 / 6305.5 = 0.285x   B8   1806.7 / 6603.5 = 0.274x
     B16 1723.8 / 6295.5 = 0.274x   B24  1770.3 / 6375.3 = 0.278x
     B32 1769.1 / 6299.3 = 0.281x
   So the prefill parity this campaign certified (H200 pp10 1.273x)
   is a LONG-PROMPT result - it was measured at p=ub=2048/4096/8192.
   At a 128-token prompt mach1 runs 0.27-0.57x of q4, i.e. up to
   3.6x SLOWER, because every PP lane (RT_APPLY_TC, EXP_APPLY_MMA,
   EXP_APPLY_FP16, GG, and the skinny mm) gates on nt >= 256 and a
   128-token ubatch engages NONE of them.
   *** THE UNIFYING DIAGNOSIS: mach1's fast paths are admitted only
   inside narrow token-count windows - nt >= 256 for prefill,
   nt == 16 exactly for decode - and OUTSIDE those windows the stack
   falls back to generic paths and loses to q4, badly. Both headline
   wins (1.273x prefill, 1.353x decode) sit inside their window. The
   campaign has been optimizing the peaks of a comb rather than the
   envelope. Widening the admission windows is worth more than any
   further kernel tuning inside them. ***
   CAUSE, and it is not subtle: every fused decode path is pinned to
   nt == 16 EXACTLY - mega QONCE (`n_tok==16` + MEGA_NT clamp),
   rt_mma16/imma8 spine and split-K (`nt==16`), head_mma16
   (`nt==16`), sibling batch (`b_nt==16`), GDN_SGF builder
   (`n_seqs==16`). At B2/B4/B8 every one of those gates FAILS and the
   step falls back to generic paths; that is the whole 0.75-0.92x
   interior. The nt32 rung proved the fix generalizes (B24 0.535 ->
   0.903, B32 0.338 -> 0.532 by admitting (16,32]); nobody has yet
   done the same for nt < 16.
   *** FIRST TASK for the next agent: generalize the fused decode
   paths DOWNWARD to nt in {2,4,8} (and fill 24), the same way the
   nt32 rung went upward. That lifts four rungs of the curve at once
   and is worth far more than the last few percent at B16. *** Treat
   any headline number as batch-conditional until this lands: the
   certified "30 percent faster" is a B16 figure and must be quoted
   with its batch size.
   RANKED ROAD (all pure-runtime; weights untouched). Note (a) and
   (a2) are the same edit in two places - widen the admission window:
   (a) nt in {2,4,8} fused-path generalization - four rungs at
       0.748-0.919x, the largest pool of value on the board, and the
       nt32 rung is a worked template for exactly this edit.
       NTLO RUNG 1 LANDED (commit d6de150, GGML_MACH1_NTLO=1 default
       off, arm p16ntlo = goal4 + NTLO): QONCE mega and GDN_SGF admit
       nt >= 2 (producer/counters/indexed kernels are P/seq-general -
       verified in source, P = 8*nt <= 120 inside the certified
       128-pair layout); rt_mma16/imma8 per-op spine, qkvz pair / qkv
       trio / shexp gate-up sibling batching, and head_mma16 admit
       nt in (4,16) via zero-padded scratch token rows (records stay
       token-major, pad rows decode to zero; head gets a PARTIAL
       template with guarded x loads/dst stores since x/dst are graph
       tensors). nt <= 4 keeps the tuned TT-batch/fork regions
       (FORK_Z/FORK_NT live there; padded MMA at nt=2 also loses the
       head arithmetic: warp-head ~0.17 ms/tok vs 0.78 fixed).
       Sub-16 QONCE matters because the non-QONCE mega recomputes
       the activation FWHT per tile task (8-32x redundant). Bisect
       arms p16xloq/r/h/s/g (minus one family each), p16log8 adds
       GDN_NT=8. EXPECTATION, wrote down before the wall: this fixes
       the fused-path CLIFF, not the fixed decode cost - projected
       B8 ~1.0-1.1x, B4 ~0.9x, B2 ~0.8x; 1.5x at every batch in
       {1,2,4,8,16} additionally needs the fixed spine+mega decode
       (~9 ms/step) roughly halved and the B1 launch floor cut.
       COMPILE GATE PASS (`ap-yh1C5ZOLbdXuQoCl1OyQaW`). CORRECTNESS
       PASS (`ntlo-nt-s1`, L40S, `ap-fuHguncaslKWn3xsVWe8nj`):
       p16goal4 AND p16ntlo intra-batch AGREE at np 2/4/8/16 and all
       eight legs read stream sha 34cdad2c7570 - the ntlo stack is
       TOKEN-EXACT vs the fallback reference at every sub-16 width
       and byte-stable at np16. The np8 log carries the ntlo
       engagement prints, so these receipts are non-vacuous.
       SWEEP-S1 DISCARDED, MARKERS DID THEIR JOB: the `ntlo-sweep-s1`
       container ran a build with ZERO ntlo engagement (all five
       enforced markers counted 0, arm rows tracked goal4) while the
       same arm envs on a fresh container engage everything - the
       lane raised instead of publishing a silent no-op A/B. Root
       cause is a stale /vol/src/tree.tar.gz snapshot in that one
       container; ntcheck (launched earlier, same tarball) built the
       current tree. Once-diagnostics now land in-tree ("NTLO master
       switch ON" + per-gate reached lines), so a recurrence is
       self-diagnosing from the arm log alone.
       DEBUG RECEIPT (`ntlo-dbg-s1`, `ap-xEtrDSbHPyesIfcSSe5Gjw`,
       p16lodbg = ntlo + TIME=4, npl 2,8, graphs on): all five
       families ENGAGED - QONCE mega nt=2 P=16, rt spine nt=8
       (m=2048 n=4096), sibling pair nt=8, sibling trio nt=8,
       head_mma16 nt=8. expffn:nofuse:line12007 rows are the
       expected prefill n_tok>16 rejects.
       NTLO RUNG 1 WALL CERTIFIED (`ntlo-sweep-s2`, L40S
       same-container, HOSTGATE ok q4km 174.2 control 164.3, all
       five engagement markers exactly 1 on both ntlo arms):
       TG by B, q4km / goal4 / p16ntlo / p16log8:
         B1   174.22 / 164.31 (0.943) / 165.94 (0.952) / 167.26
         B2   278.58 / 264.60 (0.950) / 275.56 (0.989) / 275.74
         B4   417.84 / 374.27 (0.896) / 397.90 (0.952) / 399.40
         B8   576.58 / 418.11 (0.725) / 632.54 (1.097) / 633.63
         B16  693.43 / 937.78 (1.352) / 939.52 (1.355) / 939.03
       VERDICTS: B8 0.725 -> 1.097 (+51 percent over goal4, the
       dead zone between the nt<=4 regions and the nt16 stack is
       CLOSED); B2 +4.1 and B4 +6.3 points (QONCE+SGF only, by
       design); B1 and B16 unchanged (wall-level non-regression of
       the byte-identical paths). GDN_NT=8 on top is NOISE (B8
       633.63 vs 632.54) - SGF owns the state path at 8; drop log8,
       p16ntlo alone is the new sub-16 candidate stack.
       POSITION AFTER RUNG 1: 0.952 / 0.989 / 0.952 / 1.097 / 1.355
       at B 1/2/4/8/16. Remaining gap to 1.50x everywhere is a
       roughly uniform fixed-cost deficit at the small widths
       (~2.2-3.7 ms/step) - the spine's compute-bound 3.88 ms and
       the launch floor are the named pools; B16 needs -1.9 ms.
       RUNG 2A KILLED (`ntlo-min2-s1`, HOSTGATE ok q4km 168.5
       control 162.9, markers exactly 1 on both arms): NTLO_MIN=2
       admits the padded spine/pair/trio at nt in {2,4} and LOSES -
       B2 241.62 -> 193.68 (-20 percent), B4 359.14 -> 339.80
       (-5.4) vs the min=5 control in the same container. A full
       16-token spine launch for 2-4 real tokens loses to the tuned
       TT-batch/fork regions, the same arithmetic that kept the head
       at >4. Token gate had passed (`ntlo-min2-nt-s1`, AGREE +
       34cdad2c7570 at np 2/4/16), so this is a pure scheduling
       kill. GGML_MACH1_NTLO_MIN stays default 5; do not re-race
       without a partial-spine form whose decode cost scales with
       nt. Side receipt: this draw read B16 1.448-1.490 and B8
       1.10-1.15 - interior ratios are host-conditional, quote the
       certified s2 draw.
       NEXT RUNGS: (i) the fixed-cost attack on the spine decode
       (B2/B4/B8/B16 all bounded by it); (ii) B1 launch floor.
       B1 INSTRUMENT NOTES: nsys on these containers records NO
       kernel data (graphs on or off, `ntlo-b1-nsys-s1/-s2`) - the
       lane needs a newer nsys or a driver fix before the B1 kernel
       decomposition exists. The same-container graphs A/B
       (`ntlo-b1-graphs-s1`) came back HOSTGATE DEGRADED but read
       mach1 graphs-on 122.37 vs graphs-off 118.34 - the apparent
       +19 percent graphs-off win across the two nsys containers was
       host variance, not replay cost. B1 graph replay itself is
       healthy (117/120 steps replay, `ntlo-b1-stat-s1`, mach1 graph
       4246 nodes vs q4km 3726).
       RUNG 2B IN FLIGHT - QONCE mega pair-order (commit e051324,
       GGML_MACH1_MEGA_ORDER=1 default off, arm p16mo): block 0
       counting-sorts the (token,slot) pairs by routed expert into
       the counter-bank margin ([900,1028)+flag 1032, outside every
       certified layout); the tile task loops visit pairs through
       the permutation with identity/counters/sums keyed by the
       remapped q - values bitwise by construction. TOKEN GATE PASS
       (`mo-nt-s1`): AGREE + sha 34cdad2c7570 at np 2/8/16. WALL
       `mo-sweep-s1` was HOSTGATE DEGRADED (control B1 137.5):
       within that container mo led ntlo at every interior width
       (B1 161.1/137.5, B2 218.1/201.7, B4 342.9/329.0, B8
       543.4/508.0) but npl16 read 204.6 tok/s (78 ms/step, 4x
       collapse). THE COLLAPSE DOES NOT REPRODUCE: `mo-dbg-s1` on a
       fresh container reads npl16 866.93 graphs-on (TIME=4 table:
       126 replays / 2 captures - NO recapture loop) and 725.36 in
       the graphs-off census arm - the stall was a host-level event
       on the degraded draw, not the kernel.
       RUNG 2B KILLED on the healthy redraw (`mo-sweep-s2`, HOSTGATE
       ok q4km 172.9 control 166.3, markers exactly 1): pair-order
       is neutral-to-slightly-negative at every width vs the ntlo
       control (B1 164.36/166.32, B2 274.79/278.45, B4
       399.30/399.23, B8 631.47/637.08, B16 938.54/949.47) - the
       degraded-draw interior wins were host artifacts. The
       L2-locality bet did not cash; keep GGML_MACH1_MEGA_ORDER
       default-off as a probed ingredient and do not respend on mega
       byte-locality without a hardware-counter receipt. BONUS
       RECEIPT: the rung-1 position REPRODUCES on this second
       healthy draw - 0.962/0.997/0.954/1.099/1.359 at B 1/2/4/8/16
       (vs s2's 0.952/0.989/0.952/1.097/1.355).
       RUNG 3 KNOB RACE - ALL DEAD (`knob-sweep-s1`, HOSTGATE ok
       q4km 173.0 control 166.2, ntlo markers 1 on every arm):
       ratios vs same-draw q4km at B 1/2/4/8/16 -
         ntlo control 0.961 / 0.999 / 0.941 / 1.054 / 1.344
         +GDN_FULL=3  0.969 / 0.996 / 0.949 / 1.094 / 1.332
         +FORK_Z_NT=4 0.972 / 0.998 / 0.953 / 1.099 / 1.332
         +GRAPH_OPT   0.701 / 0.997 / 0.947 / 1.097 / 1.331
       GRAPH_OPT KILLS B1 (121.24 vs ~167, -27 percent) - reject.
       g3f and zf4 sit inside the +-1.5 percent single-draw noise
       floor at every width (the apparent B8 wins are the control
       drawing its documented low-bimodal number 609.46 while all
       three knob arms cluster 632-636). The B2/B4 gap is NOT in
       these knobs: it is the fixed per-step decode cost.
       INTERIOR CENSUS READ (`lo-census-s1`, TIME=3 graphs-off,
       npl 2): SGF level 2 IS engaged at sub-16 - the per-step
       GET_ROWS 24576 + CPY 3x8192 rows are its extra-state
       maintenance nodes (launch-inflated bookkeeping, ~13 us avg),
       not the copying fallback; the 524288-wide state gather shows
       prefill-only, confirming the indexed state path. A micro-rung
       exists there (skip identity extra copies, ~60 launches + 6 MB
       per step) but it is worth at most ~0.1-0.3 ms/step. The REAL
       named cost: MACH1_HEAD_MM 506 us/call at nt=2 (the warp
       head; one full 5-bit vocab decode per <=4-token chunk,
       ~2.5x its bandwidth floor) - decode-ALU-bound, the same
       disease as the spine's 165 GB/s. The fused spine/mega regions
       are invisible to TIME=3 (known limitation); their per-kernel
       ranking needs TIME=1 or a hardware profiler. CONSOLIDATED
       THESIS AFTER RUNGS 1-3: every width below 16 is bounded by
       the 5-bit/trellis decode ALU plus the shared stock floor; the
       cheap admission/order/knob levers are exhausted. RUNG 4 is
       the decode-ALU beam, instrument-first.
       THE INSTRUMENT EXISTS: ncu COLLECTS on Modal L40S
       (ncuprof_l40s lane, NCU_IMAGE, --clock-control none since the
       container denies clock locking; `ncu-probe-s2`,
       `ap-OV455FPwFnTOmXdJhysaxN`). First capture = the 16-token
       warmup decode step, and it REVISES THE MEGA SIZING: the QONCE
       mega (dn32, P=128) runs 171.8 us at SM 36.5 / MEM 38.1
       percent - LATENCY-BOUND with ~2.5x headroom on BOTH axes, not
       near a bandwidth floor as the ablation arithmetic (510 GB/s)
       suggested. 40 launches/step is ~6.9 ms of the B16 step; a 30
       percent latency recovery is ~-2 ms/step, i.e. the whole B16
       gap to 1.50x, and the same spin structure at P=16-64 covers
       B2-B8. The imma8 spine reads MEM 65 percent (m8192 17.4 us,
       m4096 12.2, split4 10.0) - closer to its floor, second
       priority. rt_u/out stages are tiny and launch-bound.
       STALE-TARBALL INCIDENT #2 (STRUCTURAL FIX LANDED): the
       `ncu-probe-s1/-s2`, `ncu-mega-s1` and `ms-sweep-s1`
       containers all built a pre-ntlo tree - the fixed
       /vol/src/tree.tar.gz path served stale cached bytes (second
       occurrence; the engagement markers caught both). Source
       uploads are now rev-suffixed (/src/tree-<rev>.tar.gz via
       benches/modal/upload_tree.sh, SRC_REV in the mounted script)
       so a stale cache can only fail loudly. READING TRIAGE: the
       nt16 QONCE mega profile SURVIVES (byte-identical path in both
       trees) - 171.8 us at SM 36.5/MEM 38.1, latency-bound;
       `ncu-mega-s1`'s nt8 capture is accidentally a clean profile
       of the OLD-stack nt8 fallbacks (non-QONCE mega ZDP=2 at
       147-166 us SM 26-27/MEM 25-26, Eligible 1.5 of 8 active
       warps/scheduler, 20 cyc/issued-inst; TT walk m8192 33 us at
       MEM 58) - a direct before-picture for the ntlo comparison.
       `ms-sweep-s1` is void (marker raise). Reruns on the fresh
       tree: `ncu-mega-s2` (decode stall profile, npl 8) and
       `ms-sweep-s2` (MEGA_SPLIT race; QONCE and SPLIT are mutually
       exclusive in the gate, so a split win despite losing QONCE
       would prove the barrier cost dominates).
       THE MEGA IS P-FLAT (`ncu-mega-s2`, fresh tree, env check
       NTLO=1 n_extra=34): the ntlo QONCE mega at nt=8 (P=64) runs
       165 us at SM 36.8 / MEM 39.8, eligible warps 1.65 of 8 - the
       SAME wall as the P=128 form (171.8). Half the pairs, same
       time: the barrier/latency structure, not the work, sets the
       launch cost. Per step that is a flat ~6.6 ms across B8-B16 -
       about 52 percent of the B8 step and the single largest item
       on the whole interior curve. The old-stack nt8 mega
       (non-QONCE, `ncu-mega-s1`) read 147-166 us, so QONCE buys
       nearly nothing at nt=8 and rung 1's B8 win came from the
       spine/sibling/head/SGF families. Spines at nt8: imma8 m8192
       17.9 us MEM 65 (near floor), m4096 12.4, split4-4096 10.4,
       split4-512 4.2; walk_tt_pair 16.1 x40 = 0.64 ms/step. The
       mega barrier floor IS rung 4. `ms-sweep-s2` drew a THIRD
       degraded host (control 113.2, discarded); `ms-sweep-s3` drew
       a FOURTH (control 108.2) - the MEGA_SPLIT race is PARKED
       until the pool recovers.
       STALL BREAKDOWN (`ncu-stall-s1`, mega at nt=8): No-Eligible
       63.4 percent of scheduler cycles; per-issue-active stalls -
       mio_throttle 4.50 (shared-memory instruction queue), barrier
       4.37, long_scoreboard 3.18 (global gathers), not_selected
       2.73, wait 1.15, lg_throttle 1.12, short_scoreboard 0.75,
       membar 0.56. Shared-op pressure and barriers dominate; the
       fix class is fewer barriers + fewer/wider shared ops, not
       more bandwidth.
       RUNG 4A LANDED (commit e4bde51, GGML_MACH1_EXP_SLOT_NT
       default 1, arm p16sl8 = ntlo + EXP_SLOT=1 + SLOT_NT=8): the
       round-226 per-slot mega generalizes to one plain 1024-thread
       block per (token, slot) pair at nt in [2, 8] - P independent
       barrier-free chains (16-64 blocks on 142 SMs) replace the
       P-flat cooperative mega at exactly the widths its spin
       structure starves. nt == 1 indexing is identity (tk = 0);
       the q8 z-walk form matches the QONCE class those widths
       already run, so the token gates apply unchanged. The final
       sum keeps the mega's single-writer contract (last pair sums
       all tokens).
       RUNG 4A KILLED AT THE WALL (`sl8-sweep-s1`, HOSTGATE ok q4km
       176.4 control 166.3, markers exactly 1 incl. exp_slot nt=2
       P=16): the slot form LOSES at every width it engages - B2
       278.38 -> 197.15 (-29 percent), B4 397.36 -> 329.97 (-17),
       B8 616.44 -> 569.77 (-7.6). One serial chain per pair costs
       more than the P-flat mega: the coop form's cross-block tile
       parallelism per pair beats full per-pair serialization even
       at 63 percent scheduler idle. (sl8's B16 row 747.93 is
       within-arm wobble - slot never engages at 16.) Keep
       EXP_SLOT_NT default 1; do not re-enter per-pair
       serialization. BONUS: the ntlo control reproduced its
       healthy curve a THIRD time (166.3/278.4/397.4/616.4/942.6).
       RUNG 4B KILLED (commit 750118b built it; `pf-nt-s1` token
       gate PASSED - AGREE + 34cdad2c7570 at np 2/8/16 including
       the certified 16 leg; `pf-sweep-s1` HOSTGATE ok q4km 168.4
       control 162.9, markers exactly 1): per-(proj,pair) record
       flags replacing the producer barrier LOSE at every width -
       B1 -1.7, B2 -8.0, B4 -11.5, B8 -6.7, B16 -3.3 percent vs the
       ntlo control. AUTOPSY: the flag wait put a spin +
       syncthreads + threadfence PREAMBLE ON EVERY TILE TASK
       (ntask_gu = 2P*tpo of them) - a device fence per task
       replaced one fence per block after the barrier, and that
       overhead swamps the producer-overlap gain. Keep
       GGML_MACH1_QONCE_PFLAGS default off as a probed ingredient.
       MEGA FLOOR SCORECARD after three structural attacks: the
       165-172 us P-flat launch resisted per-pair serialization
       (4a, -7 to -29 percent), expert-sorted visitation (2b,
       neutral), and barrier granularity (4b, -2 to -11 percent).
       What remains against it: (i) vectorized/wider shared reads
       inside mach1_exp_mega_tile (mio_throttle 4.50 is the top
       stall; value-identical loads, but it is surgery on the
       certified walk), (ii) the MEGA_SPLIT race, still unraced on
       a healthy host (FIVE degraded draws - `ms-sweep-s4` control
       130.9; the pool is consistently degrading the mach1 control
       arm today, so the race is PARKED - redraw at a different
       hour rather than hammering the pool), (iii) accept the mega
       floor and work the spine's MEM-65 headroom and the nt 2-4
       head instead. The certified curve stands at
       0.95-0.96 / 0.99-1.00 / 0.94-0.95 / 1.05-1.10 / 1.33-1.36
       (B 1/2/4/8/16, three healthy draws), against a 1.50 target
       at every width.
       CORRECTION - THE SPLIT RACE WAS VACUOUS: mach1_exp_mega_grid
       template evidence (`ncu-mega-s1`) shows the goal stack runs
       zmode == 2, and BOTH the MEGA_SPLIT and MEGA_HALF gates
       require zmode == 1 - p16ms never engaged anything in any of
       the five draws (its rows tracked the control within noise
       for exactly that reason). GGML_MACH1_MEGA_NOSTAGE no longer
       exists in the CUDA tree at all (bench arm name only). The
       degraded hosts were real but irrelevant; the race as
       constructed could never run. LESSON, again: no wall arm
       without its own engagement print.
       RESUME POINTS FOR THE NEXT SESSION: (1) the mega mio fix is
       NOT load widening - the tile walk's hot loop issues 32
       DATA-DEPENDENT single-word shared lookups per thread per
       tile (p4[zr], zr pseudorandom from the code stream;
       mach1_exp_mega_tile ~9913-9928) - unvectorizable by
       construction. The fix class is TABLE RESIDENCY/OCCUPANCY:
       serve the 64 KB p4 table from L1/L2 (rebuild the round-230
       nostage form - it is GONE from the tree) and/or a WG512
       2-blocks/SM PLAIN-launch split ported to the QONCE q8 class
       (the existing split/half forms are zmode==1-era and dead on
       this stack; the r241 coop hang was the residency promise,
       which plain launches do not make). Every new arm gets its
       own engagement print BEFORE its first wall.
       RUNG 6 BUILT AND BROKEN - DO NOT USE WITHOUT A BISECT: the
       QONCE split (commit 4cc1e02 + rev; GGML_MACH1_MEGA_SPLIT=1
       now composes with QONCE as two plain PHASE launches, WG 512
       half-table 2/SM, qsplit gate P <= 128). It ENGAGES (prints
       nt=2 P=16 and nt=8 P=64) but the OUTPUT IS WRONG: `qs-nt-s1`
       np2/np8 continuations die after ONE decode step (immediate
       stop tokens - step-1 logits are garbage) and np16 TIMES OUT
       (180 s = hang or a ~40x pathology at P=128). The wall sweep
       was killed before burning a container. The failure sits in
       the QONCE x PHASE x WG512 intersection - the non-QONCE
       msplit forms were historically correct, so bisect the
       QONCE-only deltas first: the producer at WG 512 under
       PHASE=1, the gu-completion GLU/down-u record pack span
       (~10230-10300, never audited under the split), and the
       phase-2 record reads. This is the r241 hang-family class;
       give it a fresh-context bisect with the san/dbg lanes before
       any retry. The env default is off, so the tree is safe as
       committed.
       INFRASTRUCTURE HALT - MODAL SPEND LIMIT: the compute-sanitizer
       run on the broken qsplit (`qs-san-s1`) built tree-4cc1e02 in
       143 s, found compute-sanitizer, then the app was killed
       mid-run: "workspace ac-B7oeE2dJFEE3Y42Fud5kvk is disabled".
       A CPU-only probe confirms "Workspace ... has exceeded its
       spend limit" - ALL Modal compute (walls, ntchecks, san, ncu)
       is blocked until the limit is raised or the billing cycle
       resets; read-only volume API still responds. The dev
       container has no GPU and no system nvcc. WORK THAT CONTINUES
       WITHOUT MODAL: (a) static bisect of the qsplit fault by
       inspection, (b) a local compile gate via pip cuda-nvcc
       wheels (arch 89 cubin, no GPU needed), (c) ledger/handoff
       hygiene. Everything measured stays measured; nothing new can
       be PROMOTED until compute returns - fixes queue behind the
       compile gate with their token/wall gates listed as PENDING
       COMPUTE.
       RUNG 6 STATIC BISECT LANDED (no GPU needed) - THE KILL NOTE
       WAS HALF WRONG AND THE FAULT IS NAMED. Pulling the saved
       `qs-nt-s1` legs off the volume (read-only API survives the
       halt): np2 AND np8 produced the CORRECT text - every
       sequence AGREEs on "The theory of general relativity says
       that the" and the streams end at the n_predict=240 length
       cap. "decoded 2/8 tokens" is the harness's normal
       short-decode accounting (~238-token prompt + 240 cap = 1-2
       steps); the "garbage step-1 logits" reading came from the
       launcher tail truncation, not from the logs. The real fault
       is ONLY the np16 hang, and the line audit names it: the
       PHASE=1 plain launch keeps the QONCE producer barrier
       (while cnt[CNT_UGU] < 2*P, mach1.cu ~10180) - a cross-block
       spin that needs all 2*P producer blocks co-resident.
       grid_gu = 2*P*tpo_gu = 16P blocks; at 44 KB dynamic smem
       the driver settles at 1 block/SM without a carveout hint,
       so ~142 resident. P <= 64 (nt <= 8): <= 128 producers fit
       and it works by scheduling order (fragile, no guarantee).
       P = 128: np16 passes the qsplit gate too (P <= 128), so the
       leg ran the SPLIT, not the certified coop - 256 producers
       > 142 resident and the residents spin on producers that can
       never schedule. DEADLOCK: the r241 class reintroduced
       INSIDE phase 1. Everything else audits bit-exact against
       the coop QONCE: fwht_block is wg-general by construction
       (per-element butterfly tree fixes the fp32 association),
       the tile's HALF fallback reads the identical word from L2,
       the SoA record staging geometry matches both phases, and
       the counter resets are complete across the pair (phase-2
       final block clears SUM/H/UGU/UDN/HS; phase 1 self-resets
       cnt[q]).
       THE FIX (this commit): the split pair now runs the PFLAGS
       producer edge instead of the barrier. Task block b waits
       only on record o = b/tpo_gu <= b, produced by block o <= b,
       so every scheduled prefix of blocks is self-sufficient -
       provably deadlock-free at any residency and any grid size.
       The flag values are already token-exact certified
       (`pf-nt-s1` np 2/8/16), and rung 4b's perf autopsy (fence
       preamble on every tile task) does not apply here: the split
       runs ONE task per block, one preamble each, and tasks start
       as soon as their own record lands (producer/consumer
       overlap the barrier form never had). Phase 2 carries PFLAGS
       for the final flag reset - without it stale flags would
       release the next region's consumers before their records
       land (the static_assert now enforces PFLAGS on both split
       phases). Also cudaFuncAttributePreferredSharedMemoryCarveout
       = 100 on the four split instantiations so two 44 KB blocks
       actually share an SM - the 2/SM intent was silently failing
       (a perf hint only; correctness no longer needs residency).
       Engagement marker now prints "... wg512 half 2/SM pflags"
       (bench marker matches on the stable prefix).
       COMPILE GATE PASSED LOCALLY: full nvcc 12.8.61 redist +
       cuda_cccl 12.8.55 + pip cudart/cublas headers compile
       mach1.cu to a sm_89 object clean (-use_fast_math
       -extended-lambda -std=c++17; only pre-existing warning
       classes). The dev container has no GPU, so this replaces
       the Modal compile leg during the spend halt.
       PENDING COMPUTE, in order, when the workspace returns:
       (1) benches/modal/upload_tree.sh (rev tarball), (2) qs
       token gate np 2/8/16 - np16 MUST reproduce stream sha
       34cdad2c7570 since the split engages at nt=16; that leg is
       the byte-identity proof for the whole split pipeline,
       (3) hostgated wall p16ms vs the ntlo control across
       B1-B16. If np16 still stalls after this fix it is not a
       deadlock (the flags ordering proof forbids one) - look at
       serialization: 16P blocks of which most wait on flags
       immediately; the fallback shape is occupancy-sized grids.
       RUNG 7 BUILT (head admission floor, same commit family):
       GGML_MACH1_HEAD_MMA_NTMIN (default 5 = the shipped nt > 4
       gate exactly) lowers the head_mma16 PARTIAL admission to
       nt >= 2 (arm p16hd2) or nt >= 1 (p16hd1). Thesis: the warp
       head at nt 1-4 pays a full 5-bit vocab decode per <=4-token
       chunk (506 us/call at nt=2, ~2.5x its bandwidth floor,
       decode-ALU-bound) while the PARTIAL mma16 decodes once
       regardless of nt - the rung 2a spine kill was PADDING
       waste, which does not price a decode-bound kernel. B1 is
       0.95x and the head is ~8 percent of its step; this is one
       of the only levers that reaches B1 at all. Wall-lane
       marker note: the lane is ONE process over npl 1-16 and the
       head print is a once-flag, so the lowered floor MOVES the
       single print to nt=2/nt=1 (nt=8 never prints; the warmup
       decode carries no logits). Gates PENDING COMPUTE: ntcheck
       np 1/2/8/16 for p16hd1 (np1 leg required - nt=1 changes
       the B1 head path), then the hostgated wall.
       RUNG 8 BUILT (nostage table on the pflags split, resume
       point (1) executed): GGML_MACH1_MEGA_NOSTAGE on top of
       MEGA_SPLIT=1 drops the smem table staging entirely - TABW=0
       instantiations read every walk lookup through the tile's
       compile-time global path (__ldg on the 64 KB z-table; r230
       forbids a runtime shared-or-global pointer switch and this
       stays compile-time). smem falls 44->12 KB and the carveout
       hint flips to prefer L1, so the table can sit IN L1 while
       smem instruction pressure (mio_throttle 4.50, the top mega
       stall) leaves the pipe. NOSTAGE=1 keeps MINB=2 (isolates
       pure table-residency vs p16ms at equal occupancy);
       NOSTAGE=2 sets launch_bounds MINB=3 (regs capped ~42,
       compiles without ptxas complaint - spill cost is the race's
       question). Arms p16msn / p16msn3; the split engagement
       print now carries the mode ("wg512 half|nostage 2|3/SM
       pflags") and the wall markers pin it by substring. Values
       are word-identical by the HALF-fallback argument. Gates
       PENDING COMPUTE: qs token gate then the five-way wall
       p16ntlo / p16ms / p16msn / p16msn3 / q4km - one container
       answers barrier-vs-flags, smem-vs-L1 table, and 2-vs-3
       blocks per SM in a single hostgated draw.
       LOCAL COMPILE GATE RECIPE (works in this GPU-less dev
       container): nvcc 12.8.61 redist + cuda_cccl 12.8.55
       (developer.download.nvidia.com/compute/cuda/redist) + pip
       nvidia-cuda-runtime-cu12/nvidia-cublas-cu12 headers;
       nvcc -c ggml/src/ggml-cuda/mach1.cu -arch=sm_89 -std=c++17
       -O1 -use_fast_math -extended-lambda -DGGML_CUDA -ccbin g++.
       All three gate runs produced the sm_89 object with only
       pre-existing warning classes.
       SPEND LIMIT LIFTED SAME SESSION; TOKEN GATES ALL PASSED
       (tree-6a83ed8, L40S): `hd-nt-s1` p16hd1 np 1/2/8/16 -
       "head_mma16 nt=1 ENGAGED" at np1, AGREE at every width,
       stream sha 34cdad2c7570 on the certified legs. `qs2-nt-s1`
       p16ntlo + p16ms + p16msn + p16msn3 at np 2/8/16 - ALL
       twelve legs AGREE with sha 34cdad2c7570, and the np16 legs
       (the width that deadlocked pre-fix) print their modes:
       "QONCE mega split ENGAGED (nt=16 P=128 wg512 half 2/SM
       pflags)" / "... nostage 2/SM pflags" / "... nostage 3/SM
       pflags". The pflags residency fix is CONFIRMED ON HARDWARE
       and the nostage walk is byte-identical as argued. Rungs
       6/7/8 are value-certified; the seven-arm hostgated wall
       (q4km, ntlo control, ms, msn, msn3, hd1, hd2 - tag
       `qs2-wall-s1`) is the remaining receipt.
       RUNGS 6/7/8 KILLED AT THE WALL (`qs2-wall-s1`, HOSTGATE ok
       q4km 174.6 control 163.3, all receipts engaged_once). TG:
         B     q4km   ntlo    ms     msn    msn3   hd1    hd2
         1    174.6  163.3  162.6  165.1  167.2  160.9  166.3
         2    279.9  277.5  242.6  269.4  262.2  269.7  270.3
         4    418.0  395.4  327.9  365.8  360.7  394.6  393.3
         8    575.8  635.6  445.3  497.0  482.1  634.3  634.6
         16   700.6  942.2  589.5  681.8  656.6  952.0  936.6
       (a) THE SPLIT FAMILY LOSES EVERYWHERE it engages: ms -13/
       -17/-30/-37 percent vs control at B2/4/8/16. The kernel
       boundary join + one-task-per-block launches cost far more
       than the coop mega's 63 percent scheduler idle. Flags vs
       barrier, L1 vs smem table, 2 vs 3 blocks/SM - no split
       shape approaches the coop form. MEGA FLOOR SCORECARD is now
       FIVE structural attacks, five losses; the P-flat coop mega
       is the best known form of this work on this hardware. Stop
       attacking its structure.
       (b) HEAD FLOOR KILLED: hd1 B1 160.9 vs control 163.3
       (-1.5), hd2 B2 -2.6 percent, B4 flat. The padded mma16
       head does not beat the warp head at nt <= 4 (and hd2's B1
       166.3 vs 163.3 with an IDENTICAL B1 path calibrates draw
       wobble at ~2 percent - the hd deltas are at or inside it,
       but never positive). Keep HEAD_MMA_NTMIN default 5.
       (c) THE SURVIVING SIGNAL: WITHIN the split family, nostage
       beat the half-smem table +15.6 percent at B16 (682 vs 590)
       and +11.6 at B8 - the cleanest table-residency A/B this
       campaign has produced. The L2/L1-served table is FASTER
       than the smem-served table at these access patterns even
       at equal occupancy (msn3 ~= msn says occupancy added
       nothing; the carveout/L1 did). NEXT RUNG: port NOSTAGE to
       the COOPERATIVE QONCE mega itself - same certified launch,
       same barrier structure, TABW=0 + carveout-prefers-L1. The
       coop kernel drops 64 KB of staged table; mio_throttle
       (4.50, stall number one) loses its source; values are
       word-identical. The control curve reproduced a FOURTH time
       (163/277/395/636/942).
       RUNG 9 BUILT, CERTIFIED, AND KILLED. Built (commit 8fc572e,
       GGML_MACH1_MEGA_NOSTAGE=1 without MEGA_SPLIT -> TABW=0 coop
       instantiations, slots 22/23, carveout prefers L1, arm
       p16qns). Token gate `qns-nt-s1` PASSED (AGREE + sha
       34cdad2c7570 at np 2/8/16, "nostage coop ENGAGED (nt=16
       P=128 tab=L1)"). WALL KILLED on two draws: `qns-wall-s1`
       (DEGRADED control 129.9, discarded, but qns trailed control
       -5 to -7 percent at B2-16 within-draw) and `qns-wall-s2`
       (HOSTGATE ok q4km 162.8 control 153.7 - a soft host, gate
       met at the edge; within-draw qns 210.5/308.9/522.4/822.5 vs
       control 240.2/356.3/584.0/903.8 = -12/-13/-11/-9 percent at
       B2/4/8/16). Same sign both draws, double digits on the
       usable one: the COOP KERNEL HIDES SMEM LATENCY BETTER THAN
       L2/L1 LATENCY - its 32 warps/block cover the smem queue,
       and moving 32 scattered reads/thread to L2 costs more than
       the mio relief buys. The split-family L1 signal did NOT
       transfer; it was an artifact of the split's own losing
       structure (small blocks, nothing else to hide behind).
       MEGA SCORECARD CLOSED - SIX STRUCTURAL ATTACKS, SIX LOSSES:
       per-pair slot serialization (-7..-29), expert-order
       (neutral), barrier->flags granularity (-2..-11), the plain
       split family in three shapes (-13..-37), and the L1 table
       on the coop form (-9..-13). The P-flat cooperative mega
       with the smem-staged table is locally optimal on this
       hardware among every form tested. STOP ATTACKING THE MEGA'S
       STRUCTURE; its 165-172 us floor is the accepted price.
       CORRECTED FIX-CLASS NOTE for the resume points: "table
       residency/occupancy" is now MEASURED DEAD in both
       directions (occupancy via the split, residency via
       nostage). The mio_throttle stall is load-bearing only in
       the sense that the warps ARE covering it - there is no
       free latency to reclaim there.
       NEXT INSTRUMENT (in flight, `abl-ceiling-s1`): the
       ABLATE=255 ceiling at B1/2/4 - all mach1 weight stages off,
       timing shape only. If ablated-mach1 t/s < 1.5x q4 at a
       width, NO decode-side work can reach the goal there and the
       remaining gap is graph shape (node count) plus the shared
       stock floor - the beam pivots accordingly with a measured
       budget.
       *** THE CEILING TABLE AND THE FEASIBILITY ARITHMETIC ***
       (`abl-ceiling-s2/-s3`; s1 died on the marker gate - npl 1,2,4
       cannot fire the control's nt=8 markers, run ablate probes
       without marker-gated arms). s3 draw (q4 B1 183.9, fast
       healthy host; s2's g4ab B1 58.65 was a bad first leg - s3
       supersedes it):
         B      q4km    a255(g4ab)   a120(mega off)
         1     183.9      413.0        129.5
         2     302.1      722.0        286.3
         4     453.7     1071.7        436.2
         8     628.6     1772.0        647.0
       CEILINGS (a255/q4): B1 2.25x, B2 2.39x, B4 2.36x, B8 2.82x
       (B16 2.74x from `g4ab-s1`). The stock floor + graph shape
       leave 2.2-2.8x headroom at EVERY width - the goal is never
       graph-bounded; the whole gap is mach1 compute.
       INSTRUMENT CAUTION EXTENDED: ABLATE=120 is clean ONLY at
       nt==16. At B1/B2 the mega-off path is SLOWER than the full
       control (129.5 vs ~163 at B1) - it falls to the generic
       expert path instead of removing work. Do not quote a120
       below nt=16.
       REQUIRED MACH1-COMPUTE CUT for 1.5x, from ratios (control
       R = 0.95/0.99/0.95/1.10/1.35, ceiling C as above; fraction
       = (1/R - 1/1.5)/(1/R - 1/C)):
         B1 63 percent, B2 58, B4 61, B8 44, B16 20.
       AGAINST THE SIX-KILL MEGA FLOOR THIS DECIDES THE BEAM:
       at B8 the mach1 share is ~7.7 ms/step of which the mega's
       P-flat floor is ~6.6 - zeroing EVERYTHING else (~1.1 ms)
       still needs the mega to shed ~35 percent, which six
       structural attacks say it will not. B1/B2/B4 need even
       larger cuts with the same dominant item. VERDICT: 1.5x at
       B1-B8 is INFEASIBLE in this kernel class on this hardware
       under the pure-runtime constraint - not for lack of
       ceiling, but because the dominant kernel is at a measured
       structural floor and the rest is too small to cover the
       difference. B16 remains ARITHMETICALLY OPEN: it needs ~20
       percent of mach1 compute (~1.8 ms/step) with non-mega at
       3.88 ms and the spine's MEM-65 headroom worth at most ~1.4
       ms even at a perfect bandwidth floor - tight, not excluded,
       and the only width where decode-side work can still move
       the goal needle. The campaign's honest position: hold the
       certified curve (0.95/0.99/0.95/1.10/1.33-1.36), spend
       remaining effort on the B16 spine beam, and record that the
       flat-1.5x target requires either a weight-format change
       (blocked: exporter constraint), a different device class,
       or a relaxation of the target.
       THE SPINE BEAM CLOSES THE STORY (`ncu-spine-s1`, imma8 at
       nt=16, stall ratios per issue-active): the plain
       rt_spine_imma8_cpasync kernels read wait 1.21-1.22 /
       long_scoreboard 0.56-1.31 / barrier 0.44-0.78 /
       short_scoreboard 0.55-0.69 / mio 0.01-0.20 - BALANCED, no
       dominant reclaimable stall; the kernel is at its structural
       balance point at MEM 65. The split4 forms' worst instances
       are long_scoreboard 5.54-5.72 (K-split shrinks per-block
       work below what the cp.async depth covers) - fixable with
       deeper staging, but those kernels are ~0.6 ms of the 3.88
       ms spine. Optimistic total spine recovery ~0.6-0.9 ms
       against the 1.8 ms B16 gap. FINAL VERDICT, ALL BEAMS
       MEASURED: the flat 1.5x-vs-Q4 target is NOT attainable at
       ANY of B1/2/4/8/16 with pure-runtime kernel work in this
       kernel class on L40S. The proof chain: ceilings 2.2-2.8x
       (never graph-bounded) + mega floor (six kills) + head
       optimal (hd race) + spine balanced (this capture) + knobs/
       maintenance neutral (rungs 3/5). What WOULD move it, all
       outside the current envelope: (a) the standard-output
       basis (+1.31 ms/step at B16, measured - needs the exporter
       and the Mac-only selector gate), (b) a device with lower
       latency floors relative to bandwidth (the H200 shape
       differs), (c) a relaxed target (the certified curve already
       delivers 1.33-1.36x at B16 and parity-to--5 percent at
       B1-B4 with 7.3 vs 20.2 GiB weights). The certified curve
       0.95-0.96 / 0.99-1.00 / 0.94-0.95 / 1.05-1.10 / 1.33-1.36
       (five healthy draws) is the campaign's deliverable.
       RUNG 10 DIED ON RECON (shexp/mega stream overlap): the idea
       was to hide the shexp region (~1 ms/step) inside the mega's
       63-percent-idle latency phase via a fork/join side stream.
       IT ALREADY EXISTS - mach1.cu ~13202 "FORK_DN: the walk ran
       on the fork stream ahead of the moe output; the fold (which
       reads moe) runs as mach1_shexp_out_kernel on main instead".
       The shexp gu/down walks already overlap the moe compute on
       the fork stream; only the moe-consuming fold serializes,
       which is inherent (it reads moe_out). The overlap surface
       is already banked in the certified curve. Do not rebuild.
       ALSO RE-CONFIRMED while sizing: pipeline depth is killed by
       its own predeclared gate ("do not revisit depth without a
       new representation") and the shexp K-split is withdrawn
       (the walk is decode-bound, not grid-starved). SHARED-FLOOR
       ARITHMETIC (one-sided stock optimization in the mach1 build
       only): at B1, S ~= 2.4 ms shared, M ~= 3.7 mach1 compute,
       Q ~= 3.0 q4 compute (cross-host rough) - 1.5x needs
       S' + M <= 3.63 ms, i.e. even S' = 0 fails while M > 3.63;
       realistic shared cuts still require the six-kill mega/spine
       floors to shed 8-15 percent first. The lever inventory is
       now EMPTY within the envelope: mega structure (six kills),
       head admission (raced), spine depth (predeclared kill),
       spine splits (materiality kill), table residency (both
       directions), knobs/maintenance (neutral), shexp overlap
       (already implemented), standard-output basis (blocked on
       exporter + Mac-local quality gate). Further same-envelope
       racing is measured waste; the campaign holds the certified
       curve and waits on an envelope decision.
       SAME-HOST BOUND TABLE (`bound-final-s1`, HOSTGATE ok q4km
       164.8 control 157.0; a soft host where the mach1 arms
       degrade more than q4, so these bounds are conservative).
       TG t/s q4/ctl/abl255: B1 164.8/157.0/344.8, B2 268.4/
       232.0/653.5, B4 399.2/314.7/939.1, B8 555.4/539.4/1512.1,
       B16 678.6/859.5/1735.4. In ms/step, budget for 1.5x =
       q4/1.5 - abl vs current mach1 compute = ctl - abl:
         B1   1.15 vs 3.47  -> cut 67 percent
         B2   1.91 vs 5.56  -> cut 66
         B4   2.42 vs 8.45  -> cut 71
         B8   4.31 vs 9.54  -> cut 55
         B16  6.50 vs 9.40  -> cut 31
       Ceilings abl/q4 2.09-2.72 - never graph-bounded, same
       conclusion single-draw clean. The cross-host estimate
       (44-63 B1-B8, 20 B16) was the OPTIMISTIC end; same-host
       says over half of all mach1 compute must vanish at B1-B8
       and ~31 percent at B16, against the six-kill mega floor,
       the balanced spine, and the already-banked shexp overlap.
       THE BOUND IS FINAL. The record is complete: goal
       infeasible in-envelope at every width, receipts at every
       branch, and the three envelope decisions (exporter/device/
       target) sized in the entries above.
       *** CAMPAIGN RE-SCOPED BY THE USER: >= 1.10x AT ALL OF
       B1/2/4/8/16 *** (envelope decision (c) taken - "okay, do at
       least 10% faster on these batch sizes"). NEW GAP TABLE from
       the certified curve and the measured ceilings: B16 1.33-1.36
       ALREADY MET. B8 1.05-1.10 AT THE LINE (needs margin, not a
       lever). The work is B1 +16 percent, B2 +11, B4 +16 - by the
       bound arithmetic that is a 17-24 percent mach1-compute cut
       at B1-B4 (vs the infeasible 55-71 for 1.5x): B1 ~0.8
       ms/step, B2 ~0.95, B4 ~1.9 (soft-host scale). FEASIBILITY
       CASE: rungs 4-9 attacked the nt>=2 QONCE mega and the B16
       spine; the nt 1-4 interior NEVER had its own instrumented
       beam - its only named numbers are the warp head (506 us at
       nt=2, ~9 percent of B2 compute) and "launch/node count" at
       B1. The TIME=3 census was sync-inflated and the fused
       regions were invisible to it; per-kernel hardware truth for
       the interior does not exist yet. INSTRUMENT FIRST (in
       flight): `ncu-int1-s1` and `ncu-int2-s1` - full mach1
       kernel census (kregex mach1, count 60, skip 10) at npl=1
       and npl=2. The kernel-duration ranking at those widths IS
       the lever list for this campaign; attack in measured order.
       Note: origin gained mach1/perf-scratch (the 35B additive
       line - single-token rt_apply, opt-in persistent banks,
       kernel-dispatch latency probe). Different model family, not
       direct assets, but the dispatch-latency probe and the
       single-token rt_apply shape are worth reading if B1's
       census points at launch overhead.
       THE INTERIOR CENSUS LANDED (`ncu-int1-s3`/`ncu-int2-s3`,
       skip=3000 past the nt=16 warmup - CAUTION: skip<720
       captures the warmup, `-s1/-s2` tags are warmup slices; the
       nt=16 slice in `ncu-int2-s2` remains valid as a B16-shape
       profile: mega 165.9 us avg exactly on the known floor,
       transforms ~15 percent). REAL DECODE SHARES OF MACH1 KERNEL
       TIME (ncu-inflated clocks, use ratios):
       npl=1 (per ~11-layer window + one head): head_mm_tg 478 us
       ONCE (~8 percent of step kernel time); mega nt=1 (non-QONCE)
       35.2 us avg ~27 pct; rt_qkv_batch 33.0/27.2 us avg ~25 pct;
       walk_rows_v 15.7 us ~13 pct; SHEXP TRIO (gu 10.5 + down 5.3
       + out 5.0) ~16 pct in THREE tiny launches; gdn_core 7.1 ~6
       pct; u/out tc singles ~6 pct.
       npl=2: mega QONCE P=16 53.4 us ~37 pct; walk_tt_pair 27.0 +
       walk_tt 14.4 ~27 pct; shexp trio ~16 pct; u/out ~9 pct.
       THE 1.10x RUNG LIST, sized against the gaps (B1 -0.82
       ms/step of mach1 compute, B2 -0.95, B4 ~-1.9):
       (A) SHEXP TRIO FUSION at nt 1-4: the fused out+glu exists
       but rides the sibling-batch path (nt >= ntlo_min); the trio
       (gu/down/out, 3 launches ~21 us/layer) is the unfused
       remainder. Fuse down+out via the sum4 last-block pattern
       (counter epilogue, fp32 order preserved) and gu+down via
       the mini-mega counter chain: 3 launches -> 1, ~0.4-0.5
       ms/step at nt 1-4. Value-identical class.
       (B) WALK_TT NT-SHAPE at nt=2-4: walk_tt_pair reads 27 us at
       nt=2 vs 16 us at nt=16 for the SAME op - the decode is
       once-per-op either way, so the delta is the apply/pad path.
       If the nt=2 form can reach the nt=16 shape: ~0.8 ms/step at
       B2 (27 pct family). INSTRUMENT FIRST: read the walk_tt tc
       apply at nt<5 before building.
       (C) HEAD FOLD TIER at B1: head_mm_tg 478 us; the in-tree
       fold bank (fp16, ~1 GiB VRAM) was measured ~0.25 ms - +3.6
       percent of the B1 step for one env if the bank story is
       acceptable. RACE IT (arm + memory note).
       (D) u/out single-sandwich fusion into spines (~0.2-0.3
       ms at nt 1-4). Build after A/B/C.
       Sum A+B+C+D covers the B1 and B2 gaps with margin and most
       of B4; B8 needs only its 1.05-1.10 wobble closed - A and B
       apply there too (trio runs at nt<5 only, but B is nt 2-8).
       B16 already clears 1.10. NEXT ACTION: build A (the trio
       fusion), gate np 1/2/4, then the 1.10x wall q4km/control/
       candidate at npl 1,2,4,8,16.
       (2) DONE AND
       KILLED: the extra-state maintenance skip (commit fb7e36e,
       GGML_MACH1_SKIP_EXTRA probe) is token-exact at np 2/8/16
       (`sx-nt-s1` - the extra mapping IS identity at steady state)
       but NEUTRAL at every width on a hostgated wall
       (`sx-sweep-s1`, q4km 173.6 control 166.3: B2 +0.1, B4 -0.1,
       B8 +0.3, B16 -1.2 percent - noise). Under graph replay the
       ~60 maintenance nodes cost single-digit us; the census's
       ~400 us/step estimate was TIME=3 sync-inflation. Keep the
       probe default off; do not build the fork-safe form.
       (3) the working instruments are the ncu lane
       (ncuprof_l40s, clock-control none), the rev-tarball uploader
       (benches/modal/upload_tree.sh - ALWAYS use it, never a fixed
       tarball path) and the enforced engagement markers - they
       caught two silent stale-build incidents this session; keep
       markers on every wall arm.
   (a2) PP lanes below nt = 256 - short-prompt prefill runs
       0.27-0.57x today and NOTHING has been tried there. Same
       shape of fix (admit smaller token counts, chunk or shrink the
       tiles); the GG lane additionally needs its host D2H sync
       reconsidered at small nt. Likely the single largest
       unexploited pool in the whole campaign.
   (b) B24/B32: the fixed ~30 ms/step B32 base (item 7 above) and
       decode-once-per-step forms.
   (c) B1 (0.993x): launch/node count per step - mach1 replays more
       nodes than q4, and B1 is the most common serving shape.
   (d) B16 1.50x: -1.98 ms/step, best odds in the spine's 3.88 ms
       (compute-bound, 165 GB/s) rather than the mega (bandwidth).
       NOTE this is now the LOWEST-value item on the list: it
       sharpens a spike the rest of the curve cannot use.
   DEAD LEVER, STOP QUOTING IT: the standard-output runtime
   (+1.31 ms/step) is blocked twice over - its selector quality gate
   exists only on the user's Mac at llm-compression@75a3b09b2 (not
   on the GitHub remote), AND landing it needs an exporter, i.e. a
   weight-format change, which is outside the pure-runtime
   constraint this campaign runs under. It is not the road to 1.50x.

7c. NTLOW RUNG EXECUTED - the fused decode stacks generalized DOWNWARD
   (commits `aad395a3f` cuda + `bb27594f5` knob/leak fix, env
   `GGML_MACH1_NTLOW=1` with `GGML_MACH1_NTLOW_MIN` (default 2), arms
   `p16ntlow` / `p16nl8` / `p16ntl32` / `p16nl832`, default OFF).
   WHAT CHANGED. The four audited 16-token spine kernels
   (`rt_spine_mma_p16`, `rt_spine_imma8_p16`, its cp.async form and
   the split-K4 form) and `head_mm_mma16` now take the token-tile
   height NTOK as a TEMPLATE parameter - 16 is the certified
   instantiation, 8/4/2 are new - and every host site serves a batch
   as a sequence of {16,8,4,2}-token chunks at plain pointer offsets
   (u/q8-slot records, scr_v rows, split-K partials and dst rows are
   all token-major, so a chunk is an offset). The m16n8k16 tile shape,
   the shared budget and every MMA/reduction line are IDENTICAL at all
   heights; only the global rows read and written shrink, and rows
   [NTOK,16) of the staged tile are zeroed once so the dead half of
   each A fragment contributes nothing. nt == 16 is ONE 16-chunk at
   zero offset, i.e. the certified launch, and `if constexpr` keeps
   its codegen unchanged (the NTOK static_asserts are the compile-time
   half of the identity proof). The ten `mach1_nt32_on()` admission
   sites became `mach1_nt_ok(nt)`; the mega QONCE took the downward
   admission directly (it is already (token,slot)-pair general and
   P = n_used*n_tok only SHRINKS, so the certified PCAP=128 counter
   layout carries it unchanged - no widening was needed going down).
   GDN SGF took `n_seqs` in [2,16) on the model side.
   ONE LEAK FOUND AND FIXED (`bb27594f5`): routing split4_storage
   through `mach1_nt_ok` had silently switched split-K storage ON at
   B24/B32 for the pre-existing NT32 arm. It is now
   `nt == 16 || (NTLOW && nt_ok)`, so the nt32 rung's own receipts
   still describe its behavior.
   TOKEN GATE PASS (`ntlow-nt-s1`, L40S, both arms, np 16/8/4/2):
   intra-batch AGREE everywhere and stream sha `34cdad2c7570` at EVERY
   np for BOTH p16goal4 and p16ntlow - the new rungs are TOKEN-EXACT
   against the goal4 fallback at the same np, and np16 still matches
   the certified sha, so no KLD lane was needed. ENGAGEMENT (from the
   saved np logs, all counts exactly 1): p16goal4 at np8 shows ONLY
   `GDN prep` + the two table repacks - EVERY fused decode family is
   dark, which is the 0.748x B8 interior in one line. p16ntlow at np8
   adds `QONCE mega NTLOW ENGAGED (nt=8 P=64 pcap=128)`,
   `chunked spine ENGAGED (nt=8 split4=1)`,
   `head_mma16 nt=8`, the qkvz/qkv sibling batches, the shexp gate/up
   fuse and the split-K4 line; np4 and np2 the same with nt=4/nt=2.
   p16ntlow at np16 shows EXACTLY the certified marker set (no NTLOW
   and no chunked line) - the nt == 16 path is untouched at runtime.
   CERTIFICATION WALL (`ntlow-cert-s2`, L40S, HOSTGATE ok q4km=167.0
   control=159.5, C-S-Q-S-C: p16j4a, p16nl8a, q4km, p16nl8b, p16j4b;
   candidate = NTLOW + NTLOW_MIN=8 + the nt32 knobs). TG tok/s,
   control mean / candidate mean / q4km, then candidate over q4:
     B1   161.47 /  163.03 / 167.03  = 0.976x  (same code both arms)
     B2   221.32 /  224.85 / 273.66  = 0.822x  (same code both arms)
     B4   344.81 /  349.10 / 408.00  = 0.856x  (same code both arms)
     B8   413.07 /  611.09 / 559.53  = 1.092x  <- was 0.738x
     B16  915.54 /  885.44 / 681.44  = 1.299x  (same code both arms)
     B24  449.90 / 1000.10 / 828.49  = 1.207x  <- was 0.543x
     B32  301.93 /  477.05 / 888.97  = 0.537x  <- was 0.340x
   CONFIRMING DRAW (`ntlow-wall-s3`, HOSTGATE ok q4km=172.0
   control=166.3, goal4 + its flank + three candidates): B8 goal4
   417.81 / flank 420.30 vs ntlow 630.01 / nl8 659.09 / ntl32 628.09
   against q4km 576.98 (1.09-1.14x); B24 ntl32 1034.11 vs goal4 459.52
   against q4km 852.64 (1.213x); B32 ntl32 381.05 vs goal4 249.80.
   `ntlow-wall-s1` (HOSTGATE ok q4km=174.3 control=159.3) is the third
   agreeing draw: B8 626.08 vs 424.12, B24 1035.60 vs 460.85.
   B16 NO-REGRESSION. The candidate and the control are the SAME CODE
   at B16 (and, at NTLOW_MIN=8, at B1/B2/B4 too), proven three ways:
   the np16 sha is unchanged, the np16 marker set is unchanged, and
   `mach1_nt_ok(16)` returns through the same branch into the same
   NTOK=16 instantiation. The batched lane nevertheless shows a
   POSITION-dependent U at B16 - in `ntlow-cert-s2` the five arms read
   919.80 / 878.00 / [q4] / 892.88 / 911.28 by position, and in
   `ntlow-wall-s3` 932.49 / 901.55 / 902.63 / [q4] / 913.08 / 931.98 -
   ends high, middle low, about 3 percent deep, while the SAME-code
   B2/B4 rows of the same draws agree to 1.2-1.6 percent. A
   candidate-flanked draw (`ntlow-b16-s3`, gate DEGRADED so shape
   only) brackets it the other way: 957.42 / [goal4] 941.67 / 940.63.
   Read B16 deltas of this size as lane position, not as arm.
   *** KILL, WITH THE ARITHMETIC: B2 AND B4 ARE NOT REACHABLE BY TILE
   GENERALIZATION. *** NTLOW_MIN=2 (admitting nt=2 and nt=4) measures
   B2 244.49 vs goal4 265.87 (`ntlow-wall-s1`) and 244.13 vs 266.90
   (`ntlow-wall-s3`) - a REPRODUCED -8.0/-8.5 percent, while B4 is a
   wash (+2.6 percent on s1, -0.1 percent on s3). Why: fit the fused
   candidate's own ms/step from `ntlow-cert-s2` (B8 13.09, B16 18.07,
   B24 24.00) and the stack splits into a FIXED 8.1 ms/step plus
   0.62-0.74 ms/token - the spine decodes the dense weights once per
   step however many tokens ride it. q4km's whole step is 7.31 ms at
   B2 and 9.80 ms at B4, so the fused stack's FIXED part alone already
   exceeds q4's entire step at both widths. No tile shape fixes that;
   B2/B4 need the 8.1 ms fixed cost cut (item (c)-class work: launch
   and node count, and the spine's ~3.9 ms trellis walk), not a
   narrower tile. NTLOW_MIN=8 is therefore the measured production
   setting and the certified arm sets it; the knob's default stays 2
   so the A/B remains available.
   BISECT INCONCLUSIVE, AND A HOST WARNING (`ntlow-bis-s2`, HOSTGATE
   ok q4km=165.3 control=159.2): at B8 the candidate flanks read
   381.40 / 402.05 and HEAD_MMA16=0 / GDN_SGF=0 / RT_MMA16=0 read
   393.44 / 390.82 / 397.41 - all inside the flank band, so no single
   family is carrying or hurting the B8 rung. But note the absolute
   level: this host passed the B1 gate at 159.2 yet gives the fused B8
   path 381-402 where `ntlow-cert-s2` gives 610-612, with q4km's B8
   stable at 584 vs 560. The two-sided B1 gate does NOT catch every
   m1-degraded host; when a B8 draw lands 35 percent low with q4km
   normal, redraw rather than believe it.
   RESULTING CROSS-BATCH POSITION vs q4km (best mach1 arm, HOSTGATE-ok
   draws): B1 0.976, B2 0.822, B4 0.856, B8 1.092, B16 1.299,
   B24 1.207, B32 0.537. The win is no longer a single spike: mach1
   now leads q4 across THREE CONSECUTIVE rungs, B8-B16-B24, and B8
   moved 0.738 -> 1.092 and B24 0.543 -> 1.207. Still behind at B1/B2/
   B4 (fixed per-step cost) and B32 (KV pressure at c=8192, item (b)).
   NEXT, RANKED, AFTER THIS RUNG: (a) the 8.1 ms/step FIXED cost -
   it is what caps B1/B2/B4 and it is the same pool as item (c);
   (b) B32; (c) B8 stretch to 1.2x needs -1.24 ms/step from 13.09.
   ADA RETUNE CHAIN (`ada-s1`, L40S bench lane ub4096/p4096): goal3
   2053 -> +RT_APPLY_TC 3305 (+61) -> +EXP_APPLY_MMA 3571 -> +TC_NT
   3581 -> +EXP_APPLY_FP16 3070 (ADA-NEGATIVE -14.3) -> +GG 4585
   (+49) -> +fork 4649. The fp16 env is moot inside the GG branch
   (cold kernel is fp16 regardless); the standalone fp16 form loses
   to s8 MMA on Ada. L40S position: pp8f 4649 vs q4-default 6770 =
   0.687x - the H200 parity-as-shipped receipt does NOT transfer to
   Ada (fp16 TC is ~5.5x weaker vs H200; q4's MMQ path is relatively
   stronger). Ada parity needs an int8/imma hot-lane GEMM or an
   s8-cold GG variant - queued.
   GG INT8 EXECUTED (`ggi8-*`, commits b402d8a/df9d7a6). Rung 1
   (GGML_MACH1_GG_COLD_S8): the s8 MMA fused kernel grew the fp16
   form's hotmask early-return so the GG cold slice can run the
   integer lattice class. STANDALONE KILL at ub4096: pp8fc8 4255.9
   vs pp8f 4298.5 (-1.0 percent, `ggi8-wall-s1`; -1.0 again on
   `-s2`) - the cold slice is small there and s8 buys nothing over
   fp16 on it. Rung 2 (GGML_MACH1_GG_I8): hot banks decode to INT8
   (gamma varies per k-tile so it folds into the i8 lattice against
   the row-tile bound - |w| <= 8 keeps codes within +-127; wsc =
   zs0*8*gmax/127 goes to the epilogue), B rows gather+quantize per
   row (absmax/127 -> bsc), the per-bucket batched GEMMs run
   8I/8I->32I, scatter applies wsc*bsc. PROBE RECEIPT: 8I TN
   cublasGemmBatchedEx is SUPPORTED on CUDA 12.8 (status 0, all-ones
   GEMM verified); cublasGemmGroupedBatchedEx REJECTS 8I with status
   15 - the ragged API stays fp32/64-only. i8 banks are 1 B/weight
   so the 200MB cap holds 2x the hot set, and with COLD_S8 also on
   the P*n eu16 fp16 convert is skipped entirely (both sides
   integer). WALL (bench lane ub4096/p4096, three containers):
   pp8fi8 +4.6/+4.5 same-container (4494.8/4481.5 vs pp8f
   4298.5/4289.6); COMPOSED p16pp8fb8 (I8+COLD_S8) +9.6/+9.5/+9.6
   (4713.2/4698.0/4651.7 vs 4298.5/4289.6/4245.0) - the b8-over-i8
   delta is dominated by the eu16 skip (the cold swap alone is -1).
   q4km flanks 6738.0/6685.7/6624.4 -> b8/q4 0.700/0.703/0.702 from
   pp8f's 0.638-0.642 on the same hosts (reference frame: 0.687 ->
   ~0.753). TG untouched. CAP SWEEP (`ggi8-wall-s3`): GG_MB=512 flat
   (+0.6, inside round noise), GG_MB=512+GG_MIN=32 LOSES -0.7 -
   banking the 32-64 middle still cannot win; defaults keep. H200
   (`ggi8-h200-s1`): NOT Ada-only - pp8fb8 +4.7 (7255.4 vs 6932.3,
   q4 6111.2 -> 1.19x q4-default same-container), TG 150.0 both,
   smoke MATCH all arms. KLD GATE PASS (`ggi8-kld-s1`, gates
   ub=512): m1 0.360835 +- 0.005286 / top-1 75.680, pp6c8
   0.360803/75.692, pp6i8 0.361598/75.717 (+0.14 sigma), pp6i8+c8
   0.360807/75.760 - the gamma-fold weight rounding and per-row B
   quant are quality-neutral at this precision. TOKEN (`ggi8-nt-s1`):
   np16 intra-batch AGREE all arms, stream sha 34cdad2c7570 ==
   goal3 - BUT the debug arm shows NO GG engagement markers in this
   lane: the ntcheck prompt is ~226 tokens, under the nt >= 256
   gate, so np16 shas are VACUOUS for the GG stage (this
   retroactively downgrades the gg-nt-s3/ct-nt-s8 GG token receipts
   to decode-path receipts; the ub512 KLD lane is where the PP
   stack actually gets gated). MEASUREMENT NOTE: the greedy smoke
   carries a pre-existing near-tie bimodality ("Here's" vs
   "HereNECT" at one token) that wandered across ARMS and CONTAINERS
   (s1: c8, s2: i8, s3: the pp8f REFERENCE itself) with a
   byte-identical alternative continuation - arm-independent, decode
   nt=1 where none of the new envs engage; b8 matched its parent on
   2 of 3 draws and the flip hit the parent on the third. VERDICT:
   p16pp8fb8 (GG_I8=1 + GG_COLD_S8=1 on the pp8f stack) is the new
   composed L40S PP candidate at ~+9.6 over pp8f, env-gated
   default-off, positive on BOTH Ada and H200.
   EXECUTED AS A KNOB (`pp-tcnt-s1`): the gate was GGML_MACH1_TC_NT
   (default 16) - the TC stage kernels already batch by grid.z, so
   TC_NT=512 turns them on at prefill with zero code. Same container:
   PP 2010.07 -> 2059.76 (+2.5 percent, ~6 ms/ubatch of the 61 the
   ablation attributed - the rest of that 61 is dependency/launch
   structure, not butterfly compute). TG untouched. TOKEN-EXACT
   (`pp-tcnt-nt-s1`): p16pp4 np16 stream sha 34cdad2c7570 == goal3
   even under the fp16 transform basis. p16pp4 (= pp2 + TC_NT=512) is
   the composed PP candidate going forward.
   (3) the non-mach1 129 ms floor.
   PARITY BUDGET (census, sync-inflated, per 512-token
   ubatch): mach1 keys ~146 ms of ~250 ms total; the non-mach1 rest
   (~100 ms: FA prefill, norms, router chain, launch tail) alone caps
   PP near ~5100 tok/s against q4's ~80 ms/ubatch - parity work must
   eventually cover the stock-op side too, though the graphs-off
   sync inflation overstates it; get the graphs-on nsys number first.
   SIZING ARM until a KLD gate runs. TOKEN-EXACT RECEIPT
   (`pp-tc-nt-s1`): p16pp1's np16 stream sha 34cdad2c7570 equals
   p16goal3's in the same container - the fp16-u rounding changes no
   generated token on this draw (token-level receipt, not logit-bitwise;
   a KLD pass is still the promotion gate). Remaining gap 2.74x; next
   rung is the expert apply (exp_zdp_apply ~380 ms) - rewrite its MAC
   phase as m16n8k32 int8 MMA over the swz z-nibble shared tiles and
   scr_q8 QONCE records; integer accumulation keeps it bit-exact vs
   dp4a (same certified numeric class). Ragged pair counts (scr_i) are
   already handled in-kernel, so no host sync is needed.

7d. PPLOW RUNG - SHORT-PROMPT PREFILL. THE 7b DIAGNOSIS IS WRONG IN
   BOTH HALVES AND THE ADMISSION GATE IS THE SMALLER OF THE TWO
   CAUSES. (commits `44a6d938c` cuda + `f0a116765` GG/fp16
   fallthrough, env `GGML_MACH1_PPLOW=1` with `GGML_MACH1_PPLOW_MIN`
   (default 128, clamped at 64) and `GGML_MACH1_PPLOW_GG_MIN`
   (default 256), arms `p16pp6all` / `p16pl128` / `p16pl2k`, default
   OFF.)
   *** CORRECT 7b BEFORE QUOTING IT. ***
   (a) ARM, NOT GATE. `p16goal4` sets NO PP env at all - grep it: no
       RT_APPLY_TC, no EXP_APPLY_MMA/FP16/GG, TC_NT=16. The `sweep-s3`
       S_PP column therefore measured the PP stack SWITCHED OFF at
       every batch width; the nt >= 256 gates were never reached. The
       `pplow-wall-s1` engagement dump confirms it - p16goal4's marker
       set contains no `PP ...ENGAGED` line at any npl.
   (b) THE UBATCH IS NOT 128. llama-batched-bench packs npl sequences
       of npp tokens into ONE batch and llama.cpp's `split_equal` cuts
       it at n_ubatch, so at `-npp 128 -ub 512` the prefill ubatch is
       min(128*npl, 512): 128 at B1, 256 at B2, 512 from B4 up. Only
       the B1 row was ever under the 256 floor. Short PROMPT does not
       mean short UBATCH - the ubatch is what mach1's fixed
       per-ubatch cost amortizes over, and it is a serving knob.
   COST MODEL, FITTED BEFORE BUILDING (`pplow-cost-s1`, L40S bench
   lane, -p 2048, two round-robin rounds, ms/ubatch = 1000*ub/S_PP):
     ub            64     128     256     512    2048
     q4km        25.0    41.3    53.3    74.6   226.3
     goal4          -   138.1       -   301.6       -
     pp6all     148.0   139.6   148.6   179.3   436.6
   Fits over each stack's admitted region:
     q4km     28.6 ms/ubatch fixed + 0.0965 ms/token
     goal4    83.5 ms/ubatch fixed + 0.4258 ms/token  (PP lanes OFF)
     pp6all  107.4 ms/ubatch fixed + 0.1607 ms/token  (PP lanes ON)
   The PP lanes cut mach1's PER-TOKEN cost 2.65x and buy that with
   ~24 ms of extra fixed cost, so their break-even is nt ~ 60 - the
   certified 256 floor was about 4x too conservative on Ada. The SAME
   fit is the ceiling argument: mach1/q4 = (107+0.161n)/(29+0.097n)
   is 1.91x at n = 2048 and tends to 1.66x, so NO admission widening
   reaches short-prompt parity. Below n ~ 1000 the 107 ms FIXED term
   is the whole gap, and it is the per-ubatch trellis decode of the
   dense spine into a transient fp16 bank - `GGML_MACH1_BANK` defaults
   off, so that decode is paid again every single ubatch.
   WALL (`pplow-wall-s1`, L40S batched lane, HOSTGATE ok q4km=168.1
   control=159.4 "results usable", one container, npl 1..32). S_PP
   tok/s; p16goal4 and q4km reproduce `sweep-s3` to within 2-4
   percent, so this draw is comparable to it:
     B    goal4     pp6all(ub512)  pl2k(ub2048)  q4km    q4u2048
     1    324.9     707.7          437.7         1132.8  1484.1
     2   1275.3     912.6         1626.6         3596.7  3837.9
     4   1811.3    3100.0         3037.7         6218.9  6273.6
     8   1810.8    3178.8         4063.3         6509.6  8118.7
     16  1733.3    3135.3         4584.3         6212.3  8476.4
     24  1804.7    3146.8         4464.2         6328.9  8616.8
     32  1696.3    3127.8         4610.9         6249.5  8870.5
   MATCHED-UBATCH RATIO vs q4 (this is the honest position):
     B          1      2      4      8     16     24     32
     goal4  0.287  0.355  0.291  0.278  0.279  0.285  0.271
     pp6all 0.625  0.254  0.498  0.488  0.505  0.497  0.500
     pl2k   0.295  0.424  0.484  0.500  0.541  0.518  0.520
   *** TURNING THE EXISTING PP LANES ON AT THE LANE'S OWN ub=512 IS
   +71 TO +84 PERCENT AT B4-B32 AND COSTS NOTHING - NO CODE, ONE ARM.
   0.27-0.29x becomes 0.49-0.51x. Raising the ubatch to 2048 (a
   serving config, not a kernel) adds another +28 to +47 percent at
   B8-B32; against q4 AT THE SAME UBATCH that is 0.50-0.54x. *** The
   B1/B2 rows of this lane are NOT steady state - each is a single
   ubatch measured once, and whichever row first touches a PP lane
   absorbs its one-time cuBLAS handle/workspace/i8-probe cost (it
   lands on B2 for pp6all and on B1 for the PPLOW arms, which is the
   whole of the 707.7-vs-457.6 and 912.6-vs-1638.1 swaps; B1+B2 total
   time is 0.436 s for pl128 against 0.461 s for pp6all). Read B1/B2
   from the bench lane, not from here.
   TG IS UNTOUCHED. Every mach1 arm in the wall runs identical decode
   code (every PP env gates at nt >= 64 at the loosest, decode tops
   out at 32) and the S_TG columns agree to within lane position:
   B16 826.9 / 900.5 / 911.5 / 906.9, B32 300.1 / 301.0 / 300.8 /
   301.3. Do not read the B16 spread as an arm effect.
   PPLOW ITSELF, MEASURED WHERE IT ENGAGES (`pplow-cost-s1`, bench
   lane, control = the same PP stack with the floor at 256):
     ub=128  ctrl 917.0 -> PPLOW 982.0 (+7.1 percent), +GG at 128
             964.0 (+5.1)
     ub=64   ctrl 432.4 -> PPLOW 578.0 (+33.7 percent), +GG at 64
             587.9 (+35.9)
   Engagement receipt (wall, DEBUG=1): `PPLOW rt apply tc ENGAGED at
   nt=128` and `PPLOW exp apply fp16 ENGAGED at nt=128` on the pl
   arms, absent on pp6all. The nt=64 rung is the larger number for a
   reason that is NOT the tile: at 64 tokens the pair count
   P = n_used*n_tok falls under `GGML_MACH1_DENSE_MIN` (1024) and the
   expert apply drops off the dense path entirely, so PPLOW's
   pair-floor relaxation is doing most of that +33.7. At 128 the
   dense path is already admitted and the +7.1 is the apply lanes
   alone.
   *** KILL: LOWERING THE GG FLOOR BUYS NOTHING. *** Measured, GG
   dropped to the PPLOW floor is -1.8 percent at nt=128 (964.0 vs
   982.0) and +1.7 at nt=64 (587.9 vs 578.0) - inside noise both ways,
   in neither direction worth a knob. `GGML_MACH1_PPLOW_GG_MIN`
   therefore keeps its own default of 256 and the production PPLOW arm
   leaves GG at the certified floor.
   MECHANISM (HYPOTHESIS, not measured here - the D2H sync is NOT the
   binding cost): GG banks a group only at `hcnt[g] >= GG_MIN` (64),
   and the pairs available per group average P/n_groups, which falls
   with the ubatch. Below some width no group clears the threshold,
   `n_hot` is 0, and the lane pays its sync and then runs exactly the
   cold fused kernel it exists to replace. Lowering GG_MIN cannot fix
   that: banking a group means decoding a whole expert to serve a
   handful of gathered rows - the same trade the ggi8 cap sweep
   already found losing at GG_MIN=32 even at ub=4096. To turn this
   into a receipt, print `n_hot` per call under DEBUG at ub 128 / 512
   / 2048; nothing in this rung needed it, since the wall verdict is
   the same either way.
   ONE STRUCTURAL FIX CAME OUT OF THAT: a GG request gated out by its
   OWN floor used to fall onto the standalone fp16 apply, which
   ada-s1 measured at -14.3 percent against s8 MMA on Ada. With PPLOW
   the two floors can differ, so `f0a116765` routes a GG-floor miss to
   the s8 lane instead. With PPLOW off both floors are 256 and the
   condition reads exactly as before.
   LONG-PROMPT NON-REGRESSION (`pplow-nonreg-s1`, L40S bench lane,
   cpu=16, two rounds, same container). PP tok/s per round:
     p16pp7f  (ub2048/p2048)  4039.1 / 3969.6
     p16pp8f  (ub4096/p4096)  4181.6 / 4188.3
     p16pp9f  (ub8192/p8192)  4144.6 / 4135.6
     p16pp10  (ub8192/p8192)  4526.0 / 4537.8
     q4km4k   (ub4096/p4096)  8078.4 / 8051.6
   GREEDY SMOKE: p16pp7f, p16pp8f and p16pp9f are BYTE-IDENTICAL to
   each other (md5 c46758ffb46919268635434964677d17 all three) - the
   certified long-prompt stack generates the same 64 tokens at all
   three ubatch widths under the changed binary. p16pp10 DIFFERS at
   one late near-tie ("about a capital city" vs "asking for the
   capital city", byte-identical up to it); that is the pre-existing
   i8/c8 numeric class the ggi8 block already recorded as flipping
   this smoke arm-independently, and it is decode nt=1 where no PP
   env and no PPLOW code can engage. The structural half of the proof
   is stronger than the wall: with PPLOW off `mach1_pp_min()` and
   `mach1_pp_gg_min()` both return the literal 256, `dense_min` equals
   `dense_min0`, the new `zgg && n_tok >= gg_min` term is unreachable
   (gg_ok would already have been true), and `mach1_pplow_mark()`
   returns on its first compare - every predicate on the certified
   path is textually what it was. Same-container structure also
   reproduces: p16pp10 over p16pp9f is +9.4 percent here against the
   certified +9.5/+9.6 for that composition. CAVEAT, stated plainly:
   the absolute levels are NOT comparable to the published band -
   this container runs q4km4k at 8078/8052 where the ggi8 wall's
   three containers read 6738.0/6685.7/6624.4, and p16pp8f 2.2
   percent under its published 4245-4299. Cross-container absolutes
   are not receipts; the byte-identical smoke and the predicate
   argument are.
   QUALITY GATE PASS (`pplow-kld-s2`, prefill-shaped KLD, 64 chunks,
   `--args "64@128"` - the ubatch where PPLOW actually engages, so the
   gate is not vacuous for it). Mean KLD +- sigma, sigma-distance from
   the certified m1 mean 0.360835 +- 0.005286, and same-top-1:
     m1          0.360617 +- 0.005280  (-0.04)  75.772
     p16pp6all   0.360742 +- 0.005285  (-0.02)  75.699
     p16pl128    0.360534 +- 0.005266  (-0.06)  75.766
     p16pls128   0.360362 +- 0.005267  (-0.09)  75.766
   Every arm is inside 0.1 sigma and every top-1 is inside 0.09 of the
   certified 75.680, against a +-0.3 allowance. Admitting the PP lanes
   at nt = 128 - on either the fp16 (pl) or the s8 MMA (pls) expert
   form - is quality-neutral at this precision; the lanes were already
   a certified numeric class and shrinking the ubatch does not change
   which arithmetic runs, only how much of it rides one launch.
   The mandated `--args "64@512"` gate ran too (`pplow-kld-s1`) and
   reproduces the reference EXACTLY: m1 0.360835 +- 0.005286 / 75.680,
   i.e. the published certified pair to six digits, with p16pl128 at
   0.360787 +- 0.005269 / 75.784 (-0.01 sigma, top-1 +0.10). PPLOW is
   a no-op at ub = 512, so that arm doubles as a same-container check
   that the changed binary leaves the certified PP numerics alone.
   RESULTING SHORT-PROMPT POSITION (npp=128, L40S, matched ubatch):
   B4-B32 goes 0.27-0.29x q4 to 0.49-0.51x at ub=512 and 0.48-0.54x
   at ub=2048; the ubatch-2048 arm is 0.62-0.74x of a q4 left at the
   default ub=512. mach1 is still BEHIND q4 at every short-prompt
   batch width - the rung roughly halves the deficit, it does not
   close it.
   BANK PROBE, AND IT ABORTS TODAY (`pplow-cost-s2`, arms
   `p16bku128` / `p16bklu128`). The fixed term is the dense spine's
   per-ubatch trellis decode, and `GGML_MACH1_BANK=1` exists to make
   it once-per-process. The probe confirms the sizing EXACTLY - the
   run banks 250 tensors totalling 2.62 GiB, which is the byte ledger
   spine census (1,405,091,840 weights at fp16) to the tensor - and
   then dies: `ggml-cuda.cu:108: CUDA error` inside
   `ggml_cuda_op_mach1_rt_mm` on the first decode after the last bank.
   BOTH bank arms failed identically at ub=128, and `p16bku512` /
   `p16bku2048` the same way, so all four bank arms are DEAD in this
   lane. *** THE DIAGNOSIS WRITTEN HERE - "`mach1_bank_get` sizes its
   budget as (free VRAM - 8 GiB) at FIRST USE, so the 8 GiB headroom
   is not enough" - IS WRONG, AND SO IS THE PROJECTION BELOW.
   SUPERSEDED BY 7e: the budget was 28.5 GiB against 2.62 GiB banked
   and nothing ever ran out of memory; the abort is a CUDA-graph
   capture violation. Once fixed, the lever is wall-neutral and the
   ~82 ms/ubatch projection is falsified. Read 7e, not this. ***
   The projection as it stood: removing the fixed term entirely would
   take ub=512 from 179 ms to ~82 ms/ubatch, i.e. ~0.9x q4. NOTE it
   is not free even once it runs - with banking on the nt == 1 rt op
   moves from the trellis walk to the bank apply, so it is a
   DECODE-class change and needs its own KLD gate; it cannot ride
   PPLOW's default-off argument.
   NEXT, RANKED. (a) THE FIXED ~107 ms/ubatch, via the bank lever
   above once its budget is fixed. (SUPERSEDED - 7e priced the bank
   and the fixed term is not the spine decode.)
   (b) The ubatch is a SERVING knob and nothing in the stack exposes
   it as one - a short-prompt deployment should run the largest
   n_ubatch its KV budget allows, and that is worth more than any
   kernel work at these widths.
   (c) PPLOW_MIN=64 is measured positive (+33.7) but most of it is the
   pair-floor relaxation; a cleaner rung is to make
   `GGML_MACH1_DENSE_MIN` scale with nt rather than be a flat pair
   count.
   SKINNY-MM AT SHORT UBATCHES: WALL-NEUTRAL, AS IT WAS AT LONG
   PROMPTS (`pplow-cost-s2`, one container, two round-robin rounds,
   means): at ub=128 p16sklu128 1202.4 against p16pl128u128 1194.7
   (+0.6 percent) and at ub=512 p16sku512 3168.7 against p16pau512
   3149.6 (+0.6) - the same +0.6 at both widths. The census sizes the
   lane at ~2.5 ms of a ~130 ms ubatch (under 2 percent), so this is
   the predicted result and the lane stays off.
   EXPERT-FORM RACE BELOW THE GG FLOOR (same run, ub=128): p16pau128
   (lanes gated out) 1056.0 -> p16pl128u128 (fp16 apply) 1194.7
   (+13.1) -> p16pls128u128 (s8 MMA apply) 1210.0 (+14.6). The s8 form
   is the better of the two by +1.3, reproduced in BOTH rounds
   (1211.1/1208.8 vs 1193.5/1195.9, spread 0.2 percent) - consistent
   with ada-s1's standalone-fp16 verdict and the reason `f0a116765`
   routes a GG-floor miss to s8. Same-container position at ub=128:
   0.348x q4 gated out, 0.394x with PPLOW, 0.399x with PPLOW on the s8
   form (q4u128 3034.7). NOTE this container's DECODE was unstable -
   TG wandered 84.4-117.1 on identical decode code - while its PP
   columns held to 0.2 percent across rounds, so read its TG for
   nothing and its PP as sound.

7e. BANK RUNG - THE PERSISTENT DECODED-WEIGHT BANK RUNS NOW, AND ONCE
   IT RUNS IT IS WALL-NEUTRAL. THE ~107 ms/UBATCH FIXED TERM IS NOT
   THE SPINE TRELLIS DECODE. (commits `342c5c35c` cuda +
   `7bb536013` BANK_NT + `8e6a92e38` / `bdba0549e` bench, env
   `GGML_MACH1_BANK=1` with `GGML_MACH1_BANK_GB` (default 4),
   `GGML_MACH1_BANK_RESERVE_GB` (default 8) and `GGML_MACH1_BANK_NT`
   (default 0 = the prefill floor), arms `p16bku128` / `p16bklu128` /
   `p16bku512` / `p16bku2048` / `p16bkd`, default OFF.)
   *** 7d's BANK-PROBE PARAGRAPH IS WRONG ABOUT THE ABORT. DO NOT
   QUOTE IT. ***
   (a) THE BUDGET WAS NEVER BINDING. All four dead arms banked 250
       tensors / 2.62 GiB against a budget of 28.5 GiB (27.0 GiB at
       ub=2048) on a 44.4 GiB L40S with 36.5 GiB free at first use.
       The numbers are in the `pplow-cost-s2` fail logs themselves -
       the bank spent 7 percent of the budget it was handed and no
       allocation ever failed. "Free VRAM minus 8 GiB at first use"
       IS a bad way to size a budget, but it is not what killed
       these arms.
   (b) THE ABORT IS A CUDA-GRAPH CAPTURE VIOLATION. `--mode san`
       (`bankdiag-s1`, app `ap-SbwuradV07ccDpBq4u3kS8`) runs the bank
       arm CLEAN with `GGML_CUDA_DISABLE_GRAPHS=1`, and compute-
       sanitizer memcheck returns rc=0 - so it is not a memory error.
       The message itself was being swallowed: llama-bench nulls the
       ggml log callback unless `-v`, which is why `pplow-cost-s2`
       captured a bare backtrace. With `-v` (`bankdiag-s2`, app
       `ap-3VyzskrTFSPpgu1PoQcfLw`), immediately after
       `ggml_backend_cuda_graph_compute: CUDA graph warmup complete`:
         CUDA error: operation not permitted when stream is capturing
           in function mach1_rt_z_table at mach1.cu:2405
           cudaStreamSynchronize(stream)
   (c) MECHANISM, EXACTLY. `mach1_bank_get` returned nullptr for the
       whole of a capture - including cache HITS - to keep cudaMalloc
       out of the capture. So the two uncaptured warmup executions ran
       the BANK APPLY branch and the captured one ran the WALK branch
       instead, and the walk's first-use `mach1_rt_z_table` does a D2H
       copy plus `cudaStreamSynchronize`. The bank is precisely what
       stopped the warmup from warming that lazy init. A cold
       `ggml_cuda_pool` block in the same fallback is the identical
       hazard one branch over. The general shape: A PERSISTENT CACHE
       THAT HIDES ITSELF DURING CAPTURE MAKES THE CAPTURED GRAPH TAKE
       A CODE PATH NOTHING WARMED.
   (d) THE FIX (`342c5c35c`). Serve a HIT even mid-capture and keep
       the capture guard on the MISS path only, so captured and
       uncaptured executions take the same branch. Budget: a FIXED
       cap `GGML_MACH1_BANK_GB` (default 4 GiB, ~1.5x the 2.62 GiB
       spine census) instead of a free-VRAM reading taken before the
       KV cache and compute buffers have finished growing, plus a
       per-bank live `cudaMemGetInfo` reserve
       `GGML_MACH1_BANK_RESERVE_GB` (default 8 GiB) that is re-read on
       every allocation so a context that grows later still wins.
       Every miss returns nullptr and the caller keeps its decode
       path - a bank that cannot fit degrades, it never aborts. The
       bank also gained a token floor at `eff_dense_min` (17 with
       WALK_TT=16), which is the same floor that already decides
       whether an [m, n] fp16 image gets materialized at all, so a
       banked process leaves the certified decode path textually
       alone; and `mach1_bank_enabled()` no longer disables the four
       decode fusions, which are all nt <= 16 and can therefore no
       longer meet the bank.
   WALL (`bank-wall-s1`, app `ap-js7XkjYI4Jc3y2xE8viMd7`, L40S bench
   lane cpu=16, `-p 2048 -n 128`, two round-robin rounds, ONE
   container). HOST SANITY FIRST: q4km reads 40.37 / 73.97 / 224.96
   ms/ubatch against `pplow-cost-s1`'s 41.3 / 74.6 / 226.3 and pp6all
   reads 143.41 / 180.30 / 441.70 against 139.6 / 179.3 / 436.6 -
   every rung inside 3 percent of the cost model this run reprices,
   so the draw is comparable to it and the new 4 Hz nvidia-smi
   sampler does not move the wall.
     ms/ubatch, mean of two rounds (S_PP tok/s in brackets):
       ub      q4km             ctrl              bank             d
       128     40.37 (3170.5)   143.41 (892.5)    146.50 (873.7)   -2.1
       512     73.97 (6922.2)   180.30 (2839.8)   181.63 (2819.0)  -0.7
       2048   224.96 (9103.8)   441.70 (4636.7)   438.11 (4674.6)  +0.8
   Round-to-round spread WITHIN one arm reaches 4.5 percent here
   (p16pau128 913.2/871.8, p16pau512 2891.3/2788.2), so all three
   bank deltas are inside the noise in both directions.
   *** KILL: THE BANK BUYS NOTHING ON PREFILL, AND WITH IT THE 7d
   ATTRIBUTION OF THE FIXED TERM. *** The projection was ub=512
   179 -> ~82 ms/ubatch, about 0.9x q4. With the spine decode paid
   ONCE PER PROCESS instead of once per ubatch - 250 tensors, all of
   it, confirmed banked in the run logs - the measurement is 181.63
   ms/ubatch. The bank recovers 0 of the 107 ms fixed term, so the
   fixed term is not the trellis decode and no lever built on that
   attribution is worth funding. At ub=128 the control re-decodes the
   whole spine SIXTEEN times over a 2048-token prompt and still wins
   by 2.1 percent.
   MECHANISM, MEASURED, AND IT NAILS THE FIXED TERM
   (`bank-stage-s2`, app `ap-91LhzHfSmUiTPEYzDwxuhk`, `--mode
   stagetime` at `-p 2048 -ub 512 -n 8`, GGML_MACH1_TIME=1 +
   GGML_CUDA_DISABLE_GRAPHS=1, 8 prefill ubatches per arm). The call
   counts alone confirm the lever does exactly what it says: the
   control runs `rt_dense_decode` 2,000 times (250 tensors x 8
   ubatches) and the banked arm runs it 250 times, once each.
     stage totals over the 8 ubatches, us:
       stage                    ctrl        bank      per-ubatch d
       rt_dense_decode       34,328.2     5,833.1     -3,561.9
       rt_apply_tc          212,369.3   222,479.9     +1,263.8
       TOTAL timed          1,643.1 ms  1,627.4 ms
   *** THE ENTIRE PER-UBATCH TRELLIS DECODE OF THE DENSE SPINE IS
   4.29 ms/ubatch AT ub=512. *** 7d called that same decode the whole
   of a 107 ms fixed term; it is 4 percent of it, a 25x error. The
   apply it feeds is 26.55 ms/ubatch, six times the decode, and it is
   a GEMM whose cost scales with nt - so it belongs to the per-token
   term, not the fixed one. Nothing in this census supports a ~107 ms
   fixed cost living in the spine decode-and-apply pair at all.
   AND THE BANK GIVES A THIRD OF ITS SAVING STRAIGHT BACK. Every
   apply shape is slower reading the persistent bank than reading the
   pool block the decode had just written (avg us/call, ctrl ->
   bank): m=8192 n=2048 497.26 -> 516.58 (+3.9 pct), m=2048 n=4096
   56.90 -> 61.81 (+8.6), m=4096 n=2048 51.70 -> 54.63 (+5.7),
   m=512 n=2048 20.63 -> 22.06 (+6.9), m=2048 n=512 19.15 -> 20.74
   (+8.3). That is the L2 story made explicit: the transient path is a
   producer/consumer pair - the largest spine tensor is 32 MiB against
   the L40S's 96 MB L2 - so the apply reads back what the decode just
   wrote, while the bank makes it read a 2.62 GiB working set cold
   from DRAM every ubatch. Net 3.56 saved minus 1.26 returned = 2.30
   ms/ubatch on a ~180 ms ubatch, i.e. 1.3 percent, which is why the
   wall reads neutral inside a 4.5 percent spread. READ THESE AS AN
   ATTRIBUTION, NOT A WALL: TIME=1 syncs every kernel and removes
   overlap, so the timed total (205 ms/ubatch) runs ~14 percent over
   the arm's real 180 ms.
   PPLOW COMPOSES, THE BANK STILL DOES NOT: p16bklu128 133.82
   ms/ubatch (956.5) against p16pau128's 143.41 - that is PPLOW's own
   +7 percent, arriving with the bank rather than because of it.
   MEASURED PEAK VRAM (nvidia-smi sampled at 4 Hz through every run;
   the bench lane now reports it per arm - `run_vram` in
   `benches/modal/bench_h200.py`). Same container as the wall above:
       ub      q4km        ctrl        bank        bank delta
       128     21169 MiB    8191 MiB   10851 MiB   +2660 MiB
       512     21543 MiB    8425 MiB   11105 MiB   +2680 MiB
       2048    23059 MiB   10273 MiB   12953 MiB   +2680 MiB
   +2680 MiB is 2.617 GiB, the byte-ledger spine census at fp16 to
   the MiB. THE TRADE, STATED PLAINLY: at ub=512 mach1 goes 8.23 GiB
   to 10.85 GiB against q4_K_M's 21.04 GiB, i.e. 0.39x of q4 becomes
   0.52x - still under half, and the headline 7.3 GiB model becomes
   ~9.9 GiB of weights. On this evidence that memory buys nothing on
   the wall, so the lever stays OFF and the memory story is unchanged
   by default.
   DECODE IS UNTOUCHED, BY CONSTRUCTION AND BY MEASUREMENT. tg128
   (mean of two rounds) for every arm whose bank sits at its prefill
   floor lands in one band: p16pau128 133.99, p16bku128 134.90,
   p16bklu128 135.08, p16pau512 138.44, p16bku512 135.90, p16pa2k
   134.75, p16bku2048 132.63, p16pp6all 138.37 (q4 193.9-194.3).
   DECODE GATE PASS (`bank-nt-s1`, app `ap-ugCEsWwdtT3Epkgf9aVHwR`):
   p16pp6all, p16bk AND p16bkd all reproduce the np16 stream sha
   `34cdad2c7570` with 16/16 streams in intra-batch agreement. That
   ntcheck prompt is long enough that the bank builds all 250 tensors
   during PROMPT PREFILL (the p16bk log carries the full banked
   census, the p16pp6all log carries none), so the identical
   continuation is also a prefill-numerics receipt at token level.
   *** NOT A KILL AT DECODE - THE OPPOSITE, AND IT IS UNCLAIMED. ***
   `GGML_MACH1_BANK_NT=1` (arm `p16bkd`) drops the floor so the
   nt == 1 rt op reads the fp16 bank instead of walking the
   compressed stream. It was predicted to LOSE on bytes (~2.8 GB of
   fp16 per step against ~0.70 GB compressed). MEASURED tg128
   153.63 / 152.58, mean 153.10, against p16pp6all's 139.81 / 136.92,
   mean 138.37: +10.6 percent, reproduced in BOTH rounds and clear of
   the 132.6-138.4 band every other mach1 arm in this container sits
   in. The reading: the trellis walk is ALU-bound at batch 1, not
   bandwidth-bound, so paying 4x the bytes to delete the arithmetic
   wins on Ada - which is also the honest reason the H200 receipt in
   the `mach1_bank_get` header (61.6 -> 62.5 tok/s) does not transfer.
   This is a DECODE-CLASS numeric change (the apply's fp32 summation
   order in place of the walk's) and it costs the same 2.62 GiB /
   +2680 MiB. It must clear a ub=1 KLD gate and a same-container
   confirmation before anything else. DO NOT PROMOTE IT ON THIS
   RECEIPT.
   *** SUPERSEDED BY 7f: THIS ARM DID NOT COMPUTE THE MODEL. THE
   nt == 1 BANK BRANCH CONSUMED AN UNWRITTEN ACTIVATION BUFFER, SO
   BOTH THESE NUMBERS AND THE GREEDY-SMOKE LINE BELOW ARE VOID. THE
   LEVER IS REAL BUT ITS SIZE HAD TO BE RE-MEASURED. ***
   GREEDY SMOKE, READ WITH CARE: 7 of 9 mach1 arms are byte-identical
   to p16pau128's 64-token stream; p16bku2048 and p16bkd DIFFER.
   p16bku2048's smoke config is IDENTICAL to p16bku512's (llama-cli
   takes no BENCH_UB and the 6-token smoke prompt is below every
   prefill floor), and p16bku512 MATCHED - so this smoke carries an
   arm-independent near-tie flip, the same class 7d recorded for
   p16pp10. Read the ntcheck sha, not this.
   BYTE-IDENTITY ARGUMENT (structural, and stronger than any wall):
   `decode_bank` is one launch of
   `mach1_rt_dense_decode_kernel(trellis, tlut, dst, m, n)` - a
   deterministic pure function of the weight tensor and the codebook,
   no atomics, no cross-block reduction. The transient path calls that
   lambda into a pool block; the bank path calls the SAME lambda once
   into a persistent block. The [m, n] fp16 image is therefore
   bit-identical and the apply consuming it is the same kernel with
   the same inputs, so banking cannot move a prefill logit. The only
   behavioural surface is WHICH nt gets an [m, n] image at all, and
   the floor pins that to the pre-existing `eff_dense_min`.
   QUALITY GATE PASS, AND IT IS A BYTE-IDENTITY RECEIPT
   (`bank-kld-s1`, app `ap-hZzjsvpczuRi4FHN9gpE1w`, prefill-shaped
   KLD, 64 chunks, `--args "64@512"`, one container). Mean KLD +-
   sigma, sigma-distance from the certified m1 mean 0.360835 +-
   0.005286, and same-top-1:
     m1          0.360835 +- 0.005286  ( 0.00)  75.680
     p16pp6all   0.360787 +- 0.005269  (-0.01)  75.784
     p16bk       0.360787 +- 0.005269  (-0.01)  75.784
   p16bk is inside 0.01 sigma and its top-1 is +0.104 against a +-0.3
   allowance, so the gate passes on its own terms. The stronger
   reading is that p16bk and p16pp6all agree in EVERY statistic the
   harness prints, not just the mean: Maximum KLD 10.605387, 99.9
   percent 7.001908, median 0.161085, minimum -0.000000, mean dp
   -5.032 +- 0.136, RMS dp 18.146 +- 0.238, same-top-1 75.784 +-
   0.335 - identical to the last digit across 32,768 tokens, where m1
   (a different arm, not a different bank) differs in all of them.
   That is the direct byte-identity proof the structural argument
   above predicts: banking the decoded spine moves no prefill logit
   at all, so this rung's numbers are a pure timing/VRAM question.
   NEXT, RANKED. (a) The ~107 ms/ubatch fixed term is now UNATTRIBUTED
   and the two stages this rung measured are ruled out: the spine
   decode is 4.29 ms/ubatch and the spine apply scales with nt. Find
   the fixed term in the stage census before funding another lever
   for it - `--mode stagetime` now takes BENCH_P / BENCH_UB / BENCH_N
   so it can be read at a PREFILL shape (arms `p16past` / `p16bkst`);
   its old fixed `-p 16` kept every prefill lane and the bank below
   their token floors, so no prefill stage key could ever appear in
   it, which is why this attribution went unchecked for a whole rung.
   (b) `GGML_MACH1_BANK_NT=1` at decode is the only
   positive number in this rung and is ungated. (c) The ubatch is
   still a serving knob and still the biggest short-prompt lever in
   the stack.
   RESULTING SHORT-PROMPT POSITION (bench lane, `-p 2048`, matched
   ubatch): 0.281x q4 at ub=128, 0.410x at ub=512, 0.509x at ub=2048
   without the bank; 0.276x / 0.407x / 0.513x with it. The bank moves
   the position nowhere and costs 2.62 GiB to do it.

7f. BANK_NT CERTIFICATION - 7e's B1 ARM WAS NOT COMPUTING THE MODEL.
   ONE MISSING LAUNCH. FIXED, RE-GATED, AND THE LEVER IS A REAL
   +10 TO +13 PERCENT AT B1 THAT WINS AT nt == 1 AND ONLY THERE.
   (commits `71c35884f` / `c96786583` / `3e0dfa81f` bench +
   `40671aef9` cuda; env `GGML_MACH1_BANK_NT` (default 0) and the new
   `GGML_MACH1_BANK_NT_MAX` (default 0 = uncapped); arms `p16nlbk0`
   = bank at its prefill floor, `p16nlbk` = bank at nt == 1 on the
   CERTIFIED DECODE stack (`p16nl832`), `p16nlbkc1` = the same lever
   capped to the nt == 1 window, `p16bkd` = 7e's PP-stack arm.
   EVERYTHING STAYS DEFAULT OFF - see (h).)
   (a) THE CORRECTNESS KILL, FOUND BY THE GATE 7e DEFERRED
       (`bankd-kld-s1`, gates lane = ub=1, the DECODE-shaped lane,
       64 chunks / 32,768 tokens). Mean KLD +- sigma and same-top-1:
         m1          0.359999 +- 0.005261   75.699
         p16nl832    0.359999 +- 0.005261   75.699
         p16pp6all   0.359999 +- 0.005261   75.699
         p16nlbk    12.650478 +- 0.017495    0.000   <- BANK_NT=1
         p16bkd     12.650478 +- 0.017495    0.000   <- BANK_NT=1
       ZERO same-top-1 over 32,768 tokens, mean dp -47.834 +- 0.310
       percent, RMS dp 62.068. Both banked arms agree with each other
       to the last digit and are wrong; all three unbanked arms agree
       with each other to the last digit and are right. This is not a
       quality regression, it is a kernel reading uninitialized
       memory - so every pre-fix timing number for a BANK_NT arm is
       void, including 7e's +10.6 percent (garbage activations also
       drive the MoE router, so the pre-fix step cost is not even a
       stable quantity).
   (b) MECHANISM, EXACTLY, AND IT IS ONE LINE. `ufuse`
       (GGML_MACH1_UFUSE_MAXB, default 256) folds the u transform
       INTO the rt walk kernel at nt == 1 for every m <= 4096 shape,
       and the standalone u launch is skipped. The deferred re-issue
       guarded only the nt >= 2 fold (`uf_tt`), so when BANK_NT sent
       nt == 1 down the bank branch the walk never ran, the u stage
       never ran, and `mach1_rt_apply_kernel<1>` took an unwritten
       pool block as its activations. The GGML_MACH1_ZBANK lane at
       nt == 1 sat behind the same guard and had the same latent bug.
       FIX (`40671aef9`): `if ((uf_tt || ufuse) && (zbank_served ||
       bank != nullptr)) launch_u_stage();`. BANK_NT/ZBANK are the
       only ways to reach a bank at nt == 1, so no certified path
       moves: p16nl832 reproduces `0.359999 +- 0.005261 / 75.699`
       after the fix, to the last digit.
       IT IS VISIBLE IN THE STAGE CENSUS TOO (`bankd-stage-s1`,
       `--mode stagetime` at `-p 16 -n 120`, TIME=1 + graphs off,
       pre-fix): the control's timed keys for m=2048 n=4096 nt=1 are
       `rt_walk` (4,840 calls, 98,331.2 us, avg 20.32) and NO u key -
       the fold - while the banked arm has `rt_apply` (4,840 calls,
       151,448.3 us, avg 31.29) plus `rt_dense_decode` 40 calls (once
       per tensor) and STILL no u key. A stage that has to exist and
       does not appear in either arm is the bug in one line.
   (c) WHAT ACTUALLY REACHES THE BANK AT DECODE. At nt == 1 on this
       stack exactly ONE spine family reaches `ggml_cuda_op_mach1_rt_mm`
       - m=2048 n=4096, 40 tensors, 640 MiB of fp16 - because the
       other four shapes are owned by the fused qkvz/qkvb/shexp
       families. The decode-only KLD process banks 40 tensors /
       0.62 GiB; a process that has prefilled banks all 250 / 2.62 GiB.
   (d) QUALITY GATE PASS, POST-FIX (`bankd-kld-s2`, same lane):
         m1          0.359999 +- 0.005261   75.699
         p16nl832    0.359999 +- 0.005261   75.699
         p16nlbk     0.360576 +- 0.005270   75.723
         p16bkd      0.360576 +- 0.005270   75.723
       +0.11 sigma against this run's m1 and -0.05 sigma against the
       certified 0.360835 +- 0.005286; same-top-1 +0.024 against
       75.699 and +0.043 against 75.680, both far inside the +-0.3
       allowance. NOT byte-identical, and that is expected and
       correct: max KLD 10.997368 -> 10.797287, 99.9 percent
       7.108890 -> 7.189011, median 0.161962 -> 0.162132. The prefill
       bank is byte-identical because it swaps WHERE a decoded [m, n]
       image lives; BANK_NT swaps the ARITHMETIC at nt == 1 - the ZDP
       walk's int8-quantized activation dot for an fp32 accumulation
       over fp16 weights - so a small, gated shift is the honest
       result, and the shift is if anything toward the teacher.
   (e) TOKEN GATE (`bankd-nt-s2`, L40S, post-fix). np16 stream sha
       `34cdad2c7570` with 16/16 intra-batch AGREE for p16nl832,
       p16nlbk, p16pp6all AND p16bkd - canonical. np1 240-token
       continuations (temp 0): p16nl832 `96840d3cffeb`, p16nlbk
       `ee58d9e5918d`, p16pp6all `3f7e18b1433d`, p16bkd
       `ee58d9e5918d`. The banked arms DIFFER - and so do the two
       UNBANKED stacks from each other - so np1 is a tolerance-class
       read, not a bitwise gate, and (d) is the decisive gate.
       TWO INSTRUMENT NOTES. (1) `--mode ntcheck` at np1 was VACUOUS
       until `c96786583`: n_predict 240 minus a 239-token prompt is
       ONE generated token, so the pre-fix np1 receipt certified a
       single token. It now runs 480 and strips the interleaved
       `mach1:` debug lines out of the hashed continuation. (2) The
       bench lane's 64-token greedy smoke is NOT a token gate on
       these draws: in `bankd-bench-s3` the IDENTICAL-CONFIG control
       flank p16nl8b DIFFERS from p16nl8a while both banked arms
       MATCH it, and in `bankd-bench-s1` p16pab DIFFERS from p16paa.
   (f) THE WALL, AND THE LANE THAT WAS LYING. Post-fix, at B1:
       BENCH LANE (`bankd-bench-s3`, app `ap-dnk8oQTnEvs9ctQNF3HTMP`,
       cpu=16, `-p 2048 -n 128`, two rounds, C-S-Q-S-C). tg128 tok/s
       per round then mean:
         p16nl8a  (ctrl) 132.79 132.72   132.76
         p16nlbka (bank) 144.82 146.15   145.48
         q4km            183.18 182.61   182.89
         p16nlbkb (bank) 147.77 145.30   146.53
         p16nl8b  (ctrl) 130.94 135.24   133.09
       +9.8 percent (146.00 vs 132.92). The control flanks agree to
       0.25 percent and the bank flanks to 0.7 percent, and all FOUR
       bank observations sit above all FOUR control observations.
       Prefill is untouched: pp2048 1687.3 / 1685.7 / 1666.8 / 1672.2.
       BATCH AXIS AT THE SAME CPU REQUEST (`bankd-cert-s3`,
       `--mode batched16`, the new cpu=16 batch lane, C-S-Q-S-C;
       HOSTGATE DEGRADED q4km=176.4 control=126.3, so RATIOS ONLY).
       Flank means, S_TG tok/s:
         B    ctrl      bank      bank/ctrl   q4km
         1    124.78    140.60    1.127       176.40
         2    241.66    232.41    0.962       281.74
         4    366.03    357.13    0.976       422.54
         8    653.42    651.73    0.997       580.73
         16   920.71    925.29    1.005       700.11
         24  1050.76   1051.47    1.001       850.46
         32   360.77    357.46    0.991       917.14
       *** THE CROSSOVER IS BETWEEN B1 AND B2, AND IT IS STRUCTURAL,
       NOT GRADUAL. *** nt == 1 is the only width where the bank is
       consulted against a walk that decodes the trellis for a SINGLE
       token; at nt in [2, 7] the TT walk amortizes one decode over
       the chunk's tokens while the bank's 640 MiB stays fixed per
       step, so it loses 2-4 percent (inside the 3-8 percent control
       flank spread, but negative in every draw); from nt = 8 the
       audited chunked MMA spine owns the op and the bank is never
       consulted, and at nt >= 17 both arms bank anyway - hence the
       0.99-1.01 band from B8 up, which is the no-op signature.
       *** AND THE DEFAULT-CPU BATCHED LANE PRICES THIS LEVER
       BACKWARDS. *** The same post-fix pair on `--mode batched`
       (Modal default CPU, `bankd-cert-s2`, HOSTGATE DEGRADED) reads
       B1 69.2 / 110.7 for the bank against 139.3 / 145.4 for the
       control - a 40-50 percent LOSS. BANK_NT trades ONE fused walk
       launch for TWO (u + apply), so a host-starved container prices
       the launch, not the kernel. `bench_l40s` was given cpu=16 for
       exactly this reason and the batched lane never was; every
       cross-batch position number in this ledger was taken at the
       default request. `--mode batched16` (`3e0dfa81f`) is the same
       lane at cpu=16 and should be preferred for any lever that
       changes launch counts.
   (g) MEASURED PEAK VRAM (`run_vram`, 4 Hz nvidia-smi, per arm):
         lane / config                ctrl        bank        delta
         bench -p 2048 ub 512      8162 MiB   10842 MiB    +2680 MiB
         batched c=8192 npl<=32   10296 MiB   12974 MiB    +2678 MiB
         (7e, PP stack, ub 512)    8425 MiB   11105 MiB    +2680 MiB
       q4_K_M on the same containers: 21542 MiB (bench) and
       23654 MiB (batched). +2680 MiB is the 2.617 GiB fp16 spine
       census to the MiB, reproduced on three containers. THE TRADE,
       STATED PLAINLY: in the bench config mach1 goes 7.97 GiB ->
       10.59 GiB against q4's 21.04 GiB, i.e. 0.379x of q4 becomes
       0.503x. Still barely half of q4, and it now BUYS something -
       about +10 percent at batch 1 - which 7e's version did not.
   (h) ADMISSION FLOOR, AND WHY IT IS STILL DEFAULT OFF. The measured
       window is EXACTLY nt == 1, which is what `GGML_MACH1_BANK_NT=1
       GGML_MACH1_BANK_NT_MAX=1` expresses (arm `p16nlbkc1`, np16 sha
       `34cdad2c7570`); uncapped BANK_NT=1 also hands the bank nt in
       [2, 16], where the curve above says it loses. NOT PROMOTED,
       for three reasons and none of them is the gate: the B1 ratio
       is certified on two cpu=16 containers but neither draw was
       HOSTGATE-ok at absolute level, so the ratio is a receipt and
       the resulting POSITION is not; it costs 2.617 GiB of a memory
       story that is the product's headline; and it has no
       production-server lane yet. The next rung is one HOSTGATE-ok
       cpu=16 draw of `p16nl8a,p16nlbkc1,q4km,p16nlbkc1b,p16nl8b`
       plus an F-Q-F server confirmation - then it is promotable as
       an opt-in serving switch for batch-1-dominated deployments.
   (i) TWO FREE FINDINGS. (1) AT nt == 1 THE ENTIRE PROMOTED p16
       STACK IS NUMERICALLY IDENTICAL TO PLAIN `m1`: m1, p16nl832 and
       p16pp6all all read `0.359999 +- 0.005261 / 75.699` to the last
       digit in the ub=1 lane. Every fused decode family is either
       nt >= 2 gated or bit-exact, so batch 1 has never received an
       arithmetic-changing optimization - which is the one-line
       explanation of why B1 sits at 0.976x while B16 sits at 1.30x,
       and it says the B1 rung is open ground rather than a tuned
       surface. (2) THE BANK IS A SHORT-PROMPT PREFILL WIN, which
       7e's `-p 2048` ladder could not see: at npp=128 (one ubatch,
       the worst amortization case) S_PP goes 491.8 -> 659.1/677.7
       tok/s (+34 percent) and at npp=256 1026.6 -> 1250.5/1257.4
       (+22), converging by B4. That is the per-ubatch spine decode
       being paid once per process instead of once per ubatch, and it
       is a property of `GGML_MACH1_BANK=1` alone (`p16nlbk0`, decode
       textually untouched), not of BANK_NT.
   RESULTING CROSS-BATCH POSITION. UNCHANGED, BY CONSTRUCTION: the
   certified table (B1 0.976, B2 0.822, B4 0.856, B8 1.092, B16 1.299,
   B24 1.207, B32 0.537) was re-confirmed on the one HOSTGATE-ok draw
   of this rung (`bankd-cert-s1`, q4km=170.5 control=165.1): 0.968 /
   0.853 / 0.830 / 1.058 / 1.324 / 1.229 / 0.532. Applying the
   certified +9.8-to-+12.7 percent B1 ratio PROJECTS B1 to
   1.06-1.09x, i.e. it would flip the most common serving shape from
   a loss to a win - but that is a projection from a ratio, not a
   measured position, and it needs the HOSTGATE-ok cpu=16 draw in (h)
   before it goes in the table.

## Replacement-only native-S4 spine

This family is now closed for the current schedule. The earlier F16-parent
proxy looked promising, but the exact same-source promoted-parent screen on
`ap-1ZjUmrWgdp4jRNu5yg4TVC` measured current q8/cp.async+TT at 3.161429 ms and
S4X2 at 4.166997 ms. All resource, independent mapping, pair, tail-canary,
160-producer/200-core census, and footprint gates passed, so the 1.008640-ms
regression is structural rather than a spill or ABI artifact. Do not spend on
an S4 re-encode or live route without a materially different producer/core
schedule and a new ceiling.

### Quantitative contract

The verified `qgem/byte_ledger.py` selected-spine census is 1,405,091,840
weights over 250 tensors:

```
40 x 8192x2048
30 x 4096x2048
40 x 2048x4096
40 x 2048x512
100 x 512x2048
```

Direct S4 therefore occupies exactly 702,545,920 bytes, the same 4 bpw as the
trellis code stream it replaces. One FP16 scale per output-row K256 group adds
exactly 10,977,280 bytes (10.977280 MB, 10.46875 MiB). The historical
"0.735 GB spine" release receipt is rounded and includes container/side
information; 11.5 MB is retained only as a conservative release-level scale
allowance. The exact tensor calculation is the authoritative runtime delta:
0.0625 bpw, or 1.5625 percent over the replacement S4 payload.
The current 250 scalar F32 `Wscale` values occupy 1,000 bytes and the
tier-shared F32 `[512,2]` TLUT occupies 4,096 bytes; both disappear. Thus the
raw payload delta is +10,972,184 bytes before container alignment. The
existing I8 `SU`/`SV` streams total 1,198,080 bytes and remain byte-for-byte,
so they are neither a new cost nor a removable timing shadow.

Shipping rules are non-negotiable:

- direct S4 replaces the already-4-bpw spine payload; it is never a shadow;
- no current trellis stream, decoded BF16/S8 bank, or alternate S4 bank remains
  resident after a successful load;
- current `SU`/`SV` rotations are reused; scalar `Wscale` is absorbed into the
  K256 scales and the trellis TLUT is absent;
- exporter plus runtime admission rejects any persistent shadow or a measured
  shipping delta above 11,500,000 bytes. The raw delta leaves only 527,816
  bytes for container alignment and metadata under that conservative cap;
- routed experts stay in their native approximately 1.5-bpw representation;
- the transient activation planes overlay/replace the current FP16 `u`
  scratch. At maximum n=4096 they use 65,536 bytes plus 512 bytes of scales,
  versus 131,072 bytes for the FP16 input tile;
- no expert or whole-model int4 conversion is authorized.

The isolated stage must save at least 2.316140 ms/step to advance to a
full-model tie attempt and 3.862295 ms/step to support a conservative 5 percent
lead. Relative to the 5.830 ms affected tier, those correspond to candidate
times no greater than 3.513860 ms and 1.967705 ms. For reference, the exact
wall-only lines are 4.197121 and 3.107082 ms. Do not add the measured
cp.async saving to a direct-S4 projection: both replace work in the same spine
tier. A future full-model S4 arm must use the promoted cp.async model as its
control or clear the frozen singleton threshold independently.

The first fixed-p16 measurement made the structural tradeoff concrete. On
Modal app `ap-RzCP6M3SETdCxLAiPaqMHk`, the affected spine baseline was 5.830
ms/step. The native core with prepacked activation planes averaged 4,422.03
us/step (reps 4,423.68, 4,420.83, and 4,421.57), saving 1.408 ms/step. The
standalone pack plus the same core averaged 6,053.16 us/step (reps 6,053.89,
6,052.86, and 6,052.74), losing 0.223 ms/step. Thus the standalone producer
cost 1,631.13 us/step. It is rejected as a schedule, not as a format: the core
is a real near-miss and the pack cost is the next target. Resource receipts
were pack 27 registers/zero spill and core 64 registers/zero spill with 4,192
bytes shared; the core read exactly 702,545,920 weight bytes/model step.

### Format ABI

Each tensor is a distinct format-v5 candidate, not a relabel of current codes:

| field | type and logical shape | contract |
| --- | --- | --- |
| `s4_codes` | packed I4 `[m,n]` | signed two's-complement; serialized in native B-fragment order |
| `s4_scale` | F16 `[m,n/256]` | symmetric per-output-row K256 scale; physical order output-tile-32, K256 group, row; no zero point |
| `SU`, `SV` | existing I8 `[n]`, `[m]` | retained deterministic input/output RHT signs; no duplicate stream |
| tensor metadata | integers/strings | exact manifest row below |

Every tensor manifest row binds `format=mach1_spine_direct_s4_v1`, logical
`m,n`, `k_group=256`, `weight_order=ot32_k64_orow_kbyte`,
`scale_order=ot32_g256_orow`, `nibble_order=even_input_low`,
`symbol_domain=s4_twos_complement`, `scale_dtype=f16_rne`, source tensor and
Hessian SHA256 values, encoder commit/config SHA256, and the SHA256/byte count
of codes, scales, `SU`, and `SV`. `Wscale`, trellis, and TLUT fields are
illegal. The runtime accepts the full native symbol domain [-8,7]; the first
quality arm preserves the canonical spine encoder's symmetric policy by
restricting emitted weights to [-7,7]. A separate [-8,7] arm is quality
evidence, not a silent format change.

The physical B order is output-tile-32, K-tile-64, output row, then 32 packed
bytes. Even input columns occupy low nibbles. One CTA reads 1024 contiguous
bytes for a K64 tile as aligned 16-byte `cp.async.ca` operations. Shared memory
holds only the packed S4 image; `ldmatrix.x2` maps each warp's eight rows into
the two native B registers. Runtime has no state hash, LUT gather, decoded-B
expansion, or decoded-weight write/read round trip.

Native m16n8k64 S4 requires both operands to be 4-bit. Activations therefore
use two signed-S4 planes with `q = 16*hi + lo`, qmax 119. This deliberately
narrows the symmetric source range to [-119,119]; -127..-120 and 120..127 are
not silently carried. An exhaustive 255-value startup check requires exact
recomposition for all 239 admitted values and explicit rejection of all 16
endpoints/carry cases. Relative to qmax 127, this increases the activation
step by 1.067227x and the ideal uniform-noise variance by 1.138973x; it adds no
correction instruction or side metadata. That accuracy cost must survive the
fresh-artifact KLD gate. The transient A order is K-tile-64, token row, then 32
packed bytes. `ldmatrix.x4` maps it to the four native A registers. Four K64
instructions accumulate one K256 group; the kernel combines the planes and
applies one activation and one weight scale per output element/group. The
first timing probe's standalone producer was uncompetitive. The follow-up
`PACKED_S4X2` arm performs K256 maxima, inverse scales, and q119 high/low
packing before the TC-FWHT CTA exits. Pass 2 writes its exact FP32 values into
the now-dead `sHa+sHb` shared image; after the max reduction, the same CTA
emits both native A planes and FP16 scales. It does not materialize half U,
launch a pack kernel, or allocate a second global scratch bank.

### Probe and admission contract

`llama-mach1-chainbench --price-spine-s4` is fail-closed unless
`GGML_MACH1_TIME=1`. It first exhaustively validates the narrowed activation
plane contract, then compares one exact 16x8x64 integer product with a CPU
oracle, which validates `cp.async`, `ldmatrix.x4/x2`, MMA, and C scatter. It
then reports
registers, local bytes, shared bytes, blocks/SM, the exact format/scale census,
full-stream bytes read, and alternating core-only versus standalone-pack-plus-
native samples. Its first receipt says:

```
PRICE_SPINE_S4 ENGAGED TIMING_ONLY SYNTHETIC_DIRECT_FORMAT NOT_QUALITY NEVER_LIVE_ROUTER REENCODE_REQUIRED
```

The harness rejects a missing exact-dot, replacement-not-shadow, scale census,
activation-pack, or promotion receipt and saves the result under a first-line
`TIMING_ONLY NOT_QUALITY REENCODE_REQUIRED REPLACEMENT_NOT_SHADOW` marker.
No model route, loader, exporter, or production kernel selects this format.
That first gate has now run; its exact result is the near-miss receipt above.

The active follow-up is
`llama-mach1-chainbench --price-spine-s4x2`, also fail-closed unless
`GGML_MACH1_TIME=1`. It is an exact 250-shape synthetic pair:

- control: current fixed-p16 `rt_u_tc(PACKED_F16)` plus the same native-S4
  core, with preseeded activation planes;
- candidate: `rt_u_tc(PACKED_S4X2)` plus that core;
- each arm launches 250 U producers and 250 cores per model step over exactly
  40 n512, 170 n2048, and 40 n4096 shapes;
- the candidate launches zero standalone pack kernels;
- both views fit one existing 262,144-byte FP32 U scratch. The control F16
  view occupies its first 131,072 bytes; the largest packed view occupies
  66,048 bytes after that. Candidate global allocation delta is zero;
- before timing, every specialization must report zero local bytes and at
  least one resident block/SM. A one-shot oracle requires the F16 control to
  match direct RNE conversion and the fused high plane, low plane, and scales
  to match a separate FP32 group pack bit-for-bit for n=512/2048/4096;
- the harness refuses missing mapping, resource, census, no-shadow, no-extra-
  launch, or promotion receipts, and labels the artifact `TIMING_ONLY
  NOT_QUALITY NEVER_LIVE_ROUTER`.

The hard composition gate is incremental fused overhead at most 0.410000
ms/step. It is intentionally slightly stricter than the arithmetic remainder
`1.408 + 0.635 - 1.632879 = 0.410121` ms after the two exact sibling-transform
screens. This gate is directional only: even a pass still requires fresh S4
re-encode/KLD evidence and a same-container full-model pair.

Frozen shortest command after coordination releases the Modal source slot:

```sh
bash benches/modal/run_bench.sh --mode codec-s4x2-price --gpu L40S --tag codec-s4x2-fused-p16-s1r3 --args 1,3
```

### Re-encode and quality dependency

The real encoder belongs in upstream `llm-compression`, not Mach, an experiment
copy, or the desktop vendor snapshot. The minimum input artifact is the exact
dense BF16 source tensor plus its calibration Hessian, source SHA256, tensor
name/shape, and current v3 quality-control manifest.

The shortest owned seam is `qgem/spine_encode.py::encode_tensor`: it already
owns seeded `SU`/`SV`, both RHTs, Hessian preparation, block-LDL feedback, and
the injected quantizer. Factor its pre-feedback K-group scale schedule and the
existing int5-head group quantization pattern into a qgem-owned
`DirectS4GroupQuantizer`; do not copy the LDLQ loop into a Modal experiment.
The quantizer receives the existing block position, chooses an fp16-snapped
per-row K256 scale before/during search, returns reconstructed values plus I8
symbols, and records clamp/inflation receipts. The final packer folds the
current scalar `Wscale` into those group scales, retains `SU`/`SV`, emits the
two native physical orders above, reloads them, and requires exact
symbol/scale/reference reconstruction. This is the spine counterpart of the
injected `qgem/yaqa_encode.py::TileQuantizer` boundary; the format-v4 expert
path continues to use that YAQA boundary directly.

The first quality job is one tensor of each of the five geometries, with
[-7,7] versus explicitly labeled [-8,7] and a tiny pre-feedback scale-override
sweep sharing the same cached dense/Hessian inputs, then layers 0, 20, and 39.
It advances only if packed round-trip is exact, weighted NMSE is
at most 1.05x v3 in aggregate and 1.10x per tensor, mean KLD is at most 0.3660,
same-top-1 is at least 75.43 percent, and median/p99 per-token KLD regress by no
more than 5 percent. A full freshly serialized/reloaded artifact and paired
model KLD remain mandatory. The timing probe is never quality evidence.

## Expert L1 direct-symbol V8

The candidate alphabet is `{z in Z^8: ||z||1 <= 4}`, exactly 3649 vectors.
Its wire is a contiguous MSB-first 12-bit stream. At stream bit `12*i`, the
current 16-bit direct window resolves `rank=(window>>4)&0xfff`; ranks 3649 to
4095 are reserved and decode to zero only in the timing probe. Production must
reject them. Ranking is shell, then lexicographic signed tuple.

Payload remains exactly 1.5 bpw. Replacing the current image removes 524,288
bytes of persisted F16 TLUT plus 131,136 bytes of runtime packed/levels cache
and adds one 16,384-byte derived image: net -639,040 bytes/model, with no
shadow or runtime scratch bank.

The first L2/L3 resource gate compiled at 64 registers but reported a real
144-byte stack frame, 48-byte local stores, and 68-byte local loads, so timing
was correctly forbidden. The retuned exact production-symbol arm cleared that
gate at 64 registers, zero stack/spill, and 2,064 bytes shared. In its healthy
same-allocation L40S draw, B1 host control was 152.96 tok/s and Q4 was 173.23,
so the host gate passed. B16 moved p16final 649.23 to direct-L1 673.97 tok/s
(+3.81 percent, about 0.904 ms/step); Q4 was 694.01, leaving 2.89 percent.
This was the strongest measured expert timing ingredient, but the timing probe
reused old payload bytes and therefore made quality intentionally invalid.

The real encoder and reload path was then implemented in upstream
`llm-compression` at commit `2aa9102927bf283610ba41a1987a914ffb0b2987`, with
ABI SHA256
`357495f45edc135eaac2f4339bdd9256b93060e9681dfeb39d22bef000fa02e1` and 47
tests passing. The selector-disjoint paired quality read on Modal app
`ap-qD7QGIkja8Fmmi8yuvOlYq` covered 2,655 tokens across 16 sequences and
failed every decision-bearing gate:

- reverse KL rose from 0.016893474850803614 to 0.026720270048826933
  (+58.17 percent), versus an allowed maximum of 0.019427496078424156;
- high-confidence flips rose from 15 to 30, versus an allowed maximum of 17;
- absolute residual-tail mass error rose from 0.0007184819762091627 to
  0.0009659760415890796 (+34.45 percent), versus an allowed maximum of
  0.000827254272640537.

Direct-L1 is therefore quality-rejected and archived as a timing near-miss. It
must not return to the live ranking on the strength of its old-byte timing
receipt.

## Closed structural screens

### Scaled QONCE q8 producer family

The QONCE producer scaled every FWHT output through a shared write/read pass
plus one 1024-thread barrier per pack site (2P = 256 gate/up producers at
d = 2048 and P = 128 down-u packs at d = 512 per MoE layer), then packed 16
values per q8 record. Both replacement schedules kept the records bitwise by
applying the identical per-element `__fdiv_rn(u, sqrt(d))` elsewhere:

- record-load fold (`9ab7f53`): the pack helper divides at load. Bitwise,
  CPU-oracle exact, 56 registers/zero spill - and 0.437248 -> 0.574576
  ms/step on the faithful replica census. Concentrating the d/16-thread
  division chains while 28 warps wait at the barrier costs more than the
  deleted pass. Killed on app `ap-5bdP5ywvyCsLBImwrcuO0Y`.
- FWHT last-pass fold (`e7d1080`): `mach1_fwht_block<true>` divides in the
  last butterfly store, keeping the divisions distributed. Strictly better
  than control - 0.437248 -> 0.428788 ms/step - but the saving is 0.008460
  ms/step versus the 0.150 kill line. Killed on app
  `ap-XHjK6TAibaYpQgA1SmWMpy` without spending a production TIME or wall run.

The decisive number: the producer tail is dominated by the FWHT itself and
the su*x global loads, so the entire deletable pass+barrier surface prices at
about 8.5 us/step, falsifying the 0.3-0.8 ms/step ceiling this beam was
opened under. Any future producer beam must attack the FWHT or the load
stream, not the normalization.

The probe is `GGML_MACH1_TIME=1 llama-mach1-chainbench --qonce-ipack
[reps] [layers]` (three arms, bitwise+CPU oracles, forced du = 0 records,
resource gates, alternating graph timing). The FWHT-fold runtime path remains
compiled behind `GGML_MACH1_QONCE_FWSCALE=1` (arm `p16qfs`, occupancy slot
13, `q1=2` TIME label) as a validated exact ingredient only; it must not be
promoted to default on this evidence.

### Early format-v4 pricing singleton

The first output-standard timing proxy saved 0.69149 ms/step and synthetic
tied gate/up saved 0.10546 ms/step, totaling 0.79695 ms. That was insufficient
as a standalone answer, so the tied-GU family remains archived. A later exact
production-register output census independently passed at 0.432226 ms raw and
is active above as a split-K composition beam; it still requires a real
standard-basis re-encode and selector-quality gate. The metadata, YAQA seam,
and quality contract remain in `benches/l40s-codec-format-v4.md`.

### Pairwise-sparse S4 expert rank

A fresh advisor checked a replacement-only 12-bit/8-weight sparse-S4 rank.
The proposed alphabet is combinatorially valid: the signed four-vector L1 ball
of radius five has 681 elements, and six pairwise 4:8 patterns produce 4,086
codes with ten reserved ranks. It would keep the expert payload at 1.5 bpw.
However, Ada sparse integer MMA sparsifies operand A, forcing a transposed
expert schedule whose N=8 dimension is routed tokens. The measured real-route
census has 81 duplicate routes in 27 chunks (mean live N=3) plus 47
singletons. At 3/8 useful columns, even a perfect nominal 2x sparse tensor-core
rate is only 0.75x useful work before rank decode, scaling, and scatter. This
fails the predeclared route gate of weighted N greater than four (preferably
six). Do not implement the encoder/runtime unless a materially new legal route
packer first proves N at least six without mixing experts.

### Signed-Q6 family

Scalar direct-symbol Q6 (`509a81675`, Modal app
`ap-0HZ1KcTEnIEvql60OUtw2J`) had exact 6.5-bpw arithmetic, 72 registers, zero
spill, and a 511,180,800-byte (487.50 MiB), 110-tensor shadow receipt. Its
same-host B16 pair moved p16final 643.89 to q6 598.02 tok/s (-7.12 percent,
+1.898 ms/step); Q4 was 693.92. The 4096 scalar global byte loads plus the
decoded-B shared write/read round trip dominated.

Vector staging (`38aa86920`) passed the static gate at 80 registers, zero
stack/spill, and 14,080 bytes shared, with the same exact shadow receipt and a
valid host. It improved the deficit but still moved p16final 637.26 to q6vec
608.58 tok/s (-4.50 percent, +1.184 ms/step); Q4 was 695.81. Exporting it as a
replacement would add 196,608,000 bytes over selected 4-bpw codes. Under the
tightened native-footprint rule, kill the Q6 family: no bank-conflict retune or
continuation hash.

### Expert collision reuse and phase fanout

The collision census measured 128 routed pairs touching 45.7 experts. PC2 and
PC4 ideal decode-instance reductions were 37.5 and 54.4 percent, with gross
ceilings of 1.72-2.30 and 2.50-3.33 ms/step. The first implementation spilled:
control 436.92, PC2 437.39 with 88 B/thread, and PC4 426.25 with 216 B/thread.

The spill-free follow-up isolated the scheduling cost: control 632.41 versus
PC2/WG512 527.43, PC4/WG512 500.11, and PC4/WG256 359.69 tok/s, all at zero
stack/spill. A separate five-phase fanout also had zero stack/spill (phase
register counts 48/94/31/85/48) but moved same-build control 593.42 to 549.47
tok/s (-7.40 percent). The host gate failed, so the absolute rates are not
publishable; the within-run loss is decisive. Do not reintroduce global phase
boundaries.

### Recurrent-spine implementation shapes

- Native same-warp cp.async double buffering (`c9bcc3400`) aliases the
  otherwise-dead `s_red` buffer and keeps the native 4.0-bpw source with zero
  persistent/global delta. It passed resources at 80 registers, zero
  stack/spill, and 14,848 bytes shared. Two invalid-host directional draws
  (1.825 and 0.979 ms saved) are excluded. The healthy third draw passed its
  host gate (B1 p16rti8 161.44, Q4 172.96) and moved B16 p16rti8 644.30 to
  p16rti8cp 673.80 tok/s (+4.578 percent, 1.0876 ms/step saved), with Q4 at
  694.19. Promote as a composable ingredient only after the pending
  non-vacuous np16 bitwise/continuation hash.
- Direct B fragments: 645.86 to 647.15 tok/s, about +0.049 ms/step. Retain as
  a minor exact ingredient only.
- First IMMA8 screen: 637.46 to 645.90, about +0.329 ms/step, zero global
  allocation. Retain as a minor ingredient.
- Paired CTA reuse: 582.97 to 562.59 (-3.50 percent). Halving grid width cost
  more than shared activation/scale/LUT reuse saved.
- Named-barrier IMMA pipeline: 62 registers and zero spill statically, then
  NVIDIA XID 13 followed by XID 43 on the engaged screen. No throughput number
  exists. Do not rerun without a standalone legality/SASS proof.

### Exact transform/head ingredients

The batched GDN screen moved 644.05 to 653.96 tok/s, about 0.377 ms/step. Its
sibling extension moved a GDN-enabled control 643.51 to 650.24, about 0.258
ms/step. The separate screens suggest about 0.64 ms composable, pending a
counterbalanced combined confirmation. The entire head tier is only 0.76 ms,
so head/output work cannot close the gap alone.

## Ada implementation research boundary

Only the following findings currently map to a quantified codec action:

- NVIDIA PTX defines dense
  `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` for sm80 and newer. Its
  per-thread fragments are four A u32, two B u32, and four S32 accumulator
  registers. The direct-S4 ABI above follows those published coordinates.
- NVIDIA's Ada guide gives 64K 32-bit registers/SM, 48 warps/SM, 100 KB
  shared/SM, and 99 KB/block. The probe reports resources and requires zero
  local spill; occupancy alone is not a promotion result.
- Ada sm89 has ordinary `cp.async`, but PTX bulk/TMA operations require sm90.
  TMA is not an available optimization. The S4 B tile already maps to two
  coalesced sectors, so an extra shared round trip has no assumed benefit.
- Dense 2:4 sparse MMA can offer a theoretical 2x arithmetic rate only after
  the encoder enforces the sparsity and carries metadata. It is not part of
  the baseline S4 format and has no ceiling credited here.
- The L4 study arXiv:2608.10103 reports hand-written native INT4 PTX at
  2.9-4.3x its same-precision WMMA baseline and attributes the win to native
  m16n8k64 plus coalesced loads. Its MIT reference kernel provides the exact
  `cp.async.ca`, `ldmatrix.x4/x2`, MMA, and scatter sequence used by this probe.
  This supports the instruction choice, not a model-speed prediction; our
  full-census gate remains decisive.

Primary references:

- https://docs.nvidia.com/cuda/parallel-thread-execution/
- https://docs.nvidia.com/cuda/ada-tuning-guide/
- https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
- https://arxiv.org/abs/2608.10103
- https://github.com/MattJBorowski1991/TensorCorePTX/blob/master/kernels/int4_ptx_mma_k64_x4_x2nontrans_ca.cu

## LANE AUTHORITY: the cross-batch curve, re-measured at cpu=16

The BANK_NT rung showed the default-CPU `--mode batched` lane prices at
least one lever BACKWARDS (40-50 percent inverted), because that lever
trades one fused launch for two and the lane is CPU-starved. `bench_l40s`
was given cpu=16 for exactly this class of distortion; the batched lane
never was, and every cross-batch position number written before this
section came from it. Re-measured on `--mode batched16`
(`batched_l40s_c16`, same code, cpu=16) as `c16-curve-s1`, HOSTGATE ok
(q4km=173.8, control=158.6), C-S-Q-S-C flanks, against the same-arm
default-lane draw `ntlow-cert-s2`. Candidate is NTLOW_MIN=8 (+NT32),
control p16goal4; ratio is candidate/q4km, flank means:

      width   cpu=16   default   delta
      B1      0.893    0.976     -0.083
      B2      0.906    0.822     +0.084
      B4      0.888    0.856     +0.033
      B8      1.132    1.092     +0.040
      B16     1.232    1.299     -0.067
      B24     1.167    1.207     -0.040
      B32     0.396    0.537     -0.141

READINGS. (1) The lane moves a ratio by up to 0.14 and it moves it in
BOTH directions, so it is not a uniform bias that cancels in a
comparison - it has to be pinned per width. (2) THE WINS SURVIVE: B8,
B16 and B24 are still the three consecutive winning rungs, at
1.132/1.232/1.167 rather than 1.092/1.299/1.207. The NTLOW rung's
headline claim stands on the authoritative lane. (3) The two rungs that
move most against us are B1 (0.976 -> 0.893, because q4km gains more
from the CPU allocation than mach1 does: 167.0 -> 173.8) and B32
(0.537 -> 0.396). Any B32 number from the old lane was flattering. (4)
Flank spread on this draw: candidate 0.8 percent at B32 and 4.9 percent
at B1, control 4 percent at B32 - so the B1 and B32 shifts are larger
than the flank noise, and the B4/B24 shifts are near it.
CONFOUND NOW SEPARATED - AND IT WAS THE THREAD COUNT, NOT THE CPU
REQUEST. cpu=16 also moves the core count llama.cpp sees (32 vs 17).
Within-container A/B via per-arm BENCH_T (`thr-ab-s2`, ONE cpu=16
container, duplicate arms at each setting):

      arm            B1       B8       B16
      mach1 t17    166.01   413.02   934.14
      mach1 t32    154.77   398.33   893.19   -6.8 / -3.6 / -4.4 pct
      q4km  t17    175.50   610.22   698.53
      q4km  t32    175.44   608.69   698.98   -0.03 / -0.25 / +0.06

MORE THREADS HURT MACH1 AND DO NOTHING TO Q4. That is a serving-config
finding as much as a bench one: pin threads for mach1.
MECHANISM - AND MY FIRST EXPLANATION WAS WRONG. I wrote that mach1
"replays far more nodes per step" than q4. The node-count census
(`nodes-s1`, graphstat counters at TIME>=4, B16) says otherwise:
  mach1  capture:n3966 x1, capture:n4267 x1, direct:n3966/4266/4267 x5,
         REPLAY:n3966 x126
  q4km   DIRECT:n3746 x133, and NO capture or replay row at all
Node counts are within 6 percent (3966 vs 3746) - not "far more". The
real asymmetry is the opposite of what I claimed: MACH1 DECODE RUNS AS A
CAPTURED CUDA GRAPH AND Q4 DECODE DOES NOT RUN AS A GRAPH AT ALL. q4
executes all 133 decode steps directly, almost certainly because stock
mul_mat_id for the MoE experts blocks capture (see
[TAG_MUL_MAT_ID_CUDA_GRAPHS] in ggml-cuda.cu), while mach1's custom
MACH1_EXP_MM does not block it.
So the thread sensitivity is best explained as threadpool
OVERSUBSCRIPTION DURING REPLAY: while mach1 replays one captured graph
the CPU threadpool has nothing to dispatch and spin-waits, and 32
spinning threads on a 16-CPU request contend with the one thread doing
the launch. q4, running direct, keeps those threads busy and is
indifferent. The node counts and the graph/direct split are RECEIPTS;
the spin-wait attribution is a HYPOTHESIS, not yet directly measured.
CONSEQUENCE FOR THE 1.50x ROAD: mach1 already holds the graph-replay
advantage and q4 does not, so "cut mach1's launch count" is NOT an
available lever - that asymmetry is already banked and is part of why
B8 through B24 win. The remaining B16 gap is GPU work, not host
dispatch.

BANK_NT IS KILLED ON THE AUTHORITATIVE LANE. Its certified +9.8 percent
at B1 was drawn at 32 threads. Re-drawn at cpu=16 WITH THREADS PINNED AT
17 (`bkt17-s1`, HOSTGATE ok q4km=173.3 control=163.2, flanked):

      width   control    bank    delta
      B1       162.75   148.98   -8.5 pct   (capped arm -9.5)
      B8       643.76   648.09   +0.7
      B16      926.51   928.74   +0.2
      B24     1056.78  1056.52   -0.0

The sign REVERSES: +9.8 at 32 threads, -8.5 at 17. The mechanism is
consistent with the earlier note that BANK_NT trades one fused launch for
two - the control gains much more from dropping to 17 threads (154.8 ->
162.8) than the bank arm does (which lands at 149.0), so the bank's
apparent win was the control being handicapped, not the bank being fast.
VERDICT: the bank lever is dead in every configuration measured - neutral
at prefill (recovers 0 of the fixed term), neutral at B8/B16/B24, and
negative at B1 on the authoritative lane. It stays default OFF and should
not be re-opened without a new mechanism. The 2.617 GiB it costs buys
nothing.
LESSON FOR THIS LEDGER: two levers in a row (the cross-batch curve, then
BANK_NT) changed sign under the thread count. Any A/B whose effect is
comparable to the thread effect (3-9 pct) MUST be drawn at cpu=16 with
threads pinned, or it is measuring the lane.

B16 DECODE BUDGET RE-PRICED ON THE AUTHORITATIVE LANE (`abl17-s1`,
cpu=16 @17 threads, ms/step = T_TG/128). The original budget was drawn on
the contended lane and, per the lesson above, had to be redone:

      arm              authoritative   contended-lane
      goal4 control        16.81           17.77
      ABLATE=255 floor      7.12  (42%)     8.65  (49%)
      ABLATE=120           11.63           12.53
      q4km                 23.18           23.70
      => expert mega        4.52            5.24
      => rest-of-mach1      5.18            3.88

THE TARGET IS SMALLER AND BETTER-PLACED THAN I CLAIMED. 1.50x needs
15.45 ms/step against q4's 23.18, i.e. a cut of 1.36 ms - not the 1.98 I
reported off the contended lane. And it comes out of 9.70 ms of
mach1-side work, so it is a 14 percent cut of code we own, not a raid on
a shared floor. The floor is also smaller than stated: 42 percent of the
step, not 49.
WHERE IT SHOULD COME FROM: rest-of-mach1 (spine + head) is now the
LARGEST mach1 term at 5.18 ms, ahead of the expert mega's 4.52. The
expert phase is near a bandwidth limit (~510 GB/s of compressed bytes);
the spine moves ~640 MB/step (~165 GB/s) and is ALU-bound, which the
BANK_NT rung independently confirmed by winning at nt==1 purely by
deleting walk arithmetic. 1.36 of 5.18 ms is a 26 percent cut of an
ALU-bound kernel family - a real target, not a raid on shared code.
ALSO: at B16 the PLAIN goal4 control reads 951.82 vs q4's 690.24 =
1.379x, better than the NTLOW candidate's 1.326. NTLOW costs about 3
percent at B16 and buys B8 (0.748 -> 1.132) and B24 (0.903 -> 1.238).
So the campaign's best B16 ratio is 1.379 and the best CURVE is NTLOW;
they are different arms and the ledger should quote both.
AUTHORITATIVE LANE = cpu=16 WITH THREADS PINNED AT 17. Not the default
lane (CPU-starved) and not cpu=16 at its natural 32 threads (mach1
handicapped). Measured as `c16t17-s1`, HOSTGATE ok (q4km=169.8,
control=163.2), C-S-Q-S-C flanks. Candidate/q4km, all three lanes:

      width   cpu16+t17   cpu16+t32   default
      B1        0.968       0.893      0.976
      B2        0.914       0.906      0.822
      B4        0.890       0.888      0.856
      B8        1.132       1.132      1.092
      B16       1.326       1.232      1.299
      B24       1.238       1.167      1.207
      B32       0.537       0.396      0.537

The authoritative column is best-of-both, and that is the point: the
default lane was right at B1/B16/B32 and wrong at B2/B4/B8 (starvation);
cpu16+t32 was right at B2/B4/B8 and wrong everywhere else (thread
contention). EITHER LANE ALONE MISPRICES ABOUT HALF THE CURVE.
CERTIFIED CROSS-BATCH POSITION: mach1 wins at B8 (1.132), B16 (1.326)
and B24 (1.238); loses at B1 (0.968), B2 (0.914), B4 (0.890), B32
(0.537). B16 1.326 is the highest certified decode ratio in the campaign.
EVERY cross-batch position number earlier in this document is
LANE-SUSPECT; the cpu16+t17 column supersedes all of them, including the
cpu16+t32 table above, which is retained only to show the thread effect.


## SPINE OPT rung: the spine is NOT ALU-bound, and the kill has a mechanism

THE PREMISE THIS RUNG WAS GIVEN IS FALSE. The B16 budget above sent me at
"rest-of-mach1" (5.18 ms) on the reading that the spine "moves ~640 MB/step
(~165 GB/s) and is ALU-bound". Three independent deletions of spine walk work
now measure ZERO at B16, and two positive controls in the same containers move
the wall by 0.30-0.36 ms. The spine walk kernels are grid/throughput bound, not
arithmetic bound. The 165 GB/s figure divides spine bytes by the WHOLE step; the
walk kernels only occupy a fraction of it, and inside that fraction they already
run at 430-978 GB/s.

WHERE THE 5.18 ms ACTUALLY SITS (`so-stage-s1`, TIME=1 + graphs off, BENCH_NPL
=16, per-step us over 129 decode steps, sync-inflated, 551 mach1 launches/step):

      exp_mega                 6700      qkvzb_u_tcb          491
      qkvzb_imma8              1065      qkvzb_out_tcb        462
      shexp_gub16_walk_tt       819      shexp_gub16_oglu     416
      head_mm_mma16             768      qkvb_mixed_imma8     368
      rt_imma8_split4 n4096     737      shexp_gub16_u_tcb    365
      rt_u_tc n4096             481      rt_imma8_split4 n512 367
      rt_out_tc_split4 n4096    363      rt_u_tc n512         339
      rt_out_tc_split4 n512     359      gdn_prep             235
      qkvb_out 152, qkvb_u 102        TOTAL 14587 us/step

Shares: exp_mega 46 pct, spine family (rt + qkvzb + qkvb + shexp) 47 pct, head
5.3 pct, gdn_prep 1.6 pct. Scaled onto the ablation budget's 9.70 ms of mach1
work those shares give mega 4.52 (which reproduces the ABLATE=120 receipt
EXACTLY), spine 4.65, head 0.52, gdn_prep 0.16. So the ledger's guess was right:
THE SPINE IS ~4.4-4.7 OF THE 5.18, THE HEAD IS ~0.5-0.8.

TWO WARNINGS ABOUT THE INSTRUMENTS THIS PRODUCED.
(1) SINGLE-FAMILY ABLATION IS NOT A SUBTRACTION ON THIS STACK. In `so-wall-s1`,
ABLATE=7 (rt spine) reads 20.87 ms/step and ABLATE=128 (head) reads 28.22
against a 16.67 control - both SLOWER than the arm they are supposed to be the
arm-minus-a-family of. Removing one family bails the fused route into a
fallback instead of deleting work. "control minus ABLATE=X" is only readable
where the delta comes out NEGATIVE (255 and 120 do; 7 and 128 do not).
(2) THE TIME=1 TABLE CANNOT RESOLVE THIS RUNG. Between the two census arms the
whole DECODE table moved up 3-10 pct including kernels the candidate cannot
touch (shexp_walk_tt +10.5, rt_u_tc +8.8, exp_mega +3.8) while every PREFILL key
moved +0.1-0.3 pct. The per-launch sync dominates an 8-40 us kernel and drifts
within a container. Census is structure only; the wall is the receipt.

THE CANDIDATE. `GGML_MACH1_RT_SPINE_OPT`, two exact restructurings of the native
int8 spine (`mach1_rt_spine_direct_b_i8` and the three imma8 kernels), each on
its own bit, both value-identical:
  bit 0 WALK8 - this fragment's state index is s = ri*8 + ti*2 + h, so s and h
    always share parity and j0 = 8*s gives j0>>4 == ri*4 + ti for BOTH h and
    j0&15 == 8*h. The window is therefore word-aligned at h == 0 and byte-
    shifted at h == 1: the funnel shift, the offset arithmetic, the wb_i select
    and ONE OF THE TWO shared code loads per state are dead. About half the
    walk's ALU and a third of its shared loads disappear.
  bit 1 QPAD - the staged q8 rows sit KW = 128 B apart, so the A-fragment load
    `sq + ar*KW + kb*16 + ac4` puts all eight ar rows of a warp in the same four
    banks: an 8-WAY shared bank conflict on both a.r[0] and a.r[1], 16 wavefronts
    per mma_block where 2 would do. A 16 B row pad makes the bank
    (ar*4 + kb*4 + (lane&3)) mod 32, all 32 distinct, and keeps every cp.async
    copy 16 B aligned. Costs 256 B/stage of shared, still 6 CTAs/SM.

WALL, `so-wall-s1`, cpu=16 with threads pinned at 17, HOSTGATE ok (q4km=173.4,
control=162.4), C-S-S-Q-S-S-C, B16 S_TG t/s:

      p16j4a  959.38     control            control mean 960.01 = 16.666 ms/step
      p16so1  962.17     WALK8              +0.22 pct
      p16so2  962.60     QPAD               +0.27 pct
      q4km    695.19                        = 23.015 ms/step
      p16so3a 961.63     WALK8+QPAD
      p16so3b 956.12     WALK8+QPAD         mean 958.88, -0.12 pct
      p16j4b  960.64     control
  control flank spread 0.13 pct, candidate flank spread 0.57 pct.

REDRAWN ON A SECOND HOST, `so-mech-s1`, HOSTGATE ok (q4km=173.2, control=163.2):
p16j4a 962.32 / p16j4b 957.88 (mean 960.10), p16so3 957.58 = -0.26 pct. TWO
INDEPENDENT DRAWS, BOTH FLAT, both inside the control flank spread.

VERDICT: WALK8 KILLED, QPAD KILLED, THE COMPOSITION KILLED. Together with the
BANK_NT row above (+0.2 pct at B16, and that lever deletes the ENTIRE walk),
that is FOUR independent ways of removing spine walk arithmetic or spine shared-
memory traffic, all of them worth nothing at B16.

THE TWO POSITIVE CONTROLS THAT SAY WHY (same container, `so-mech-s1`):

      arm                      B16 t/s   delta vs control   ms/step
      p16sk0 (split-K OFF)      940.10      -2.08 pct        +0.355
      p16lp1 (u stage x2)       943.37      -1.74 pct        +0.296

  p16sk0 drops the m2048 spine grids from 256 CTAs to 64 AT IDENTICAL TOTAL
  WORK and costs 0.355 ms/step. Quadrupling the grid is worth 0.36 ms; halving
  the walk's ALU is worth 0.00. That is the whole finding in one line, and it
  agrees with the in-tree stream probe already quoted at mach1.cu:5807 (64-CTA
  grids cap at ~320 GB/s, 256-CTA grids reach ~978).
  p16lp1 re-issues the rt u stage once per spine op. The stage is a pure
  function of x and su into scr_u, so the repeat is IDEMPOTENT - values are
  unchanged and this is a clean price, not an ablation. 80 extra decode
  launches/step cost 0.296 ms, i.e. 3.69 us per small spine launch.

WHERE THE NEXT RUNG SHOULD GO. Of the 551 mach1 decode launches per step, 120
are walks and 320 are u/out transforms on tiny data (rt_u_tc 80, rt_out_tc_split4
80, qkvzb u/out 60, qkvb u/out 20, shexp u/oglu 80). At the measured 3.69 us
those 320 launches price at ~1.18 ms/step - the 1.36 ms the 1.50x bar needs is
almost exactly the size of the spine's TRANSFORM-LAUNCH TAIL, not of its
arithmetic. Separately, the spine's own byte traffic is only ~694 MB/step, which
is 0.80 ms at DRAM peak against a ~4.6 ms spine term, so the walk kernels are
not where the spine's time is either. The available lever is launch STRUCTURE -
batching or fusing the u/out transforms the way QKVZ_MMA16_BATCH already batches
the qkvz siblings - and NOT another per-weight-cost rung. Note this does not
contradict the "cut mach1's launch count is not a lever" line above: that was
about HOST dispatch, which graph replay already banked. This is the GPU-side
issue cost of 320 small kernels inside the replayed graph.
Do not spend a rung on more split-K either: the in-tree probe puts 256 CTAs at
the ~978 GB/s plateau, so 4 -> 8 splits has little left to recover.

GATES. Token (`so-nt-s1`, `so-nt-s2`, `--mode ntcheck --gpu L40S`): OPT 0, 1, 2
and 3 all give intra-batch AGREE and stream sha 34cdad2c7570 at np16; at np24
(NTLOW+NT32, tiled 16+8, the only path where the OPT dispatch's q8-slot pointer
arithmetic differs from a pass-through) both the NTLOW control and NTLOW+OPT=3
give AGREE and 34cdad2c7570; at np1 all four arms give continuation sha256
96840d3cffeb. BIT-IDENTICAL AT EVERY SETTING, so the KLD gate is not owed.
Build gate ok. Peak VRAM unchanged at 9292 MiB in every candidate arm.

POSITION. B16 control/q4km reads 1.3809 (`so-wall-s1`) and 1.3818
(`so-mech-s1`), reproducing the 1.379 baseline on two hosts to within 0.07 pct.
THE RUNG ADVANCES NOTHING: best B16 ratio stays 1.379-1.382 against the 1.50x
bar, which still needs 15.34 ms/step against the measured 16.67.
`GGML_MACH1_RT_SPINE_OPT` and `GGML_MACH1_LAUNCH_PROBE` stay DEFAULT OFF. The
OPT code is retained only because it is the receipt for the kill; it should not
be promoted and should not be re-opened without a new mechanism.

## TRANSFORM TAIL rung: 3.69 us is the u stage RUNNING, not a launch costing

THE REDIRECT'S PRICE TAG IS WRONG, AND THIS RUNG MEASURED THE ERROR DIRECTLY.
The spine OPT rung priced one small mach1 decode launch at 3.69 us from
`p16lp1` (the rt u stage re-issued, 80 extra launches, +0.296 ms/step) and sent
this rung at the 320 u/out transform launches on the reading that 320 x 3.69 us
= ~1.18 ms is LAUNCH cost recoverable by batching. It is not. Removing 80 real
launches whose kernels do trivial work costs 0.6 us apiece, so 3.69 us is
overwhelmingly the u kernel EXECUTING - 16 CTAs of 512 threads on a 142-SM part
running a 4096-point tensor-core FWHT - and fusing transform launches together
cannot recover it.

CANDIDATE 1, `GGML_MACH1_RT_TAIL`. The two standalone rt projections per layer
(attention wo / GDN ssm_out, and the shared-expert down) are each followed by
elementwise nodes that read the whole [m, nt] rt output and nothing else. The
split-K TC out stage already holds every element in a register at store time,
so the fold writes `((y*sv)*gate) + add + add2` there instead: the residual
ADD, and the shexp sigmoid-gate MUL plus the moe and ffn-residual ADDs.
`__fmul_rn`/`__fadd_rn` pin the roundings the separate MUL/ADD kernels had.

WHAT IT ACTUALLY REMOVED (`rtt-cen-s1`, TIME=2 per-OP census, graphs off,
BENCH_NPL=16, 129 decode evals) - the matcher engages at EVERY site, and the
launch accounting is not the naive one:

      counter/row            control      candidate    per step
      rttail:residual              -           5160     40 (all 40 layers)
      rttail:shexp                 -           5160     40 (all 40 layers)
      ADD launches              5470            310     40 -> 0
      MUL launches              5790            630     40 -> 0

  So 80 launches/step leave the chain, not the 120 the fold consumes. The 40
  residual ADDs were ALREADY FREE: stock ggml-cuda fuses [ADD, RMS_NORM, MUL]
  and that region never reaches the per-op timer, which is why the control has
  exactly one standalone ADD launch per layer (the shexp moe/residual pair,
  itself already multi-add fused). Folding the residual ADD out of the norm
  fusion and into the rt store trades one launch for one launch.

WALL, cpu=16 threads pinned at 17, C-S-Q-S-C in s1/s2 and four flanks each in
s3, B16 S_TG t/s, HOSTGATE ok in all three:

      draw          control mean   candidate mean   delta    ctrl spread
      rtt-wall-s1        942.47          944.12    +0.175 pct   2.22 pct
      rtt-wall-s2        964.41          971.38    +0.723 pct   0.48 pct
      rtt-wall-s3        945.31          945.27    -0.005 pct   2.26 pct
      pooled                                       +0.30 pct

  In ms/step: -0.030, -0.119, +0.001, pooled -0.049 ms/step. Against 80 removed
  launches that is 0.6 us each. At the 3.69 us the redirect assumed, the same
  80 launches would have been worth 0.295 ms - SIX TIMES what the wall gives,
  and two of the three draws put the candidate inside the control flank spread.
  s1 and s3 also show the c16 containers drifting 2.2 pct across one pass
  (first arm always fastest), which is why s3 ran four flanks per arm.

CANDIDATE 2, `GGML_MACH1_TC_WG=1024`. The consequence of the diagnosis: if the
transform's cost is the block's own latency, the lever is threads per block,
not launches. The TC transform factorizes N = A*B and each warp owns
(mi, nj); when A/16 >= (WG/32)/(B/8), i.e. A*B >= 4096, doubling the threads
halves MIW, the per-warp MMA row count. That is exactly the A = 64 family: the
standalone n = 4096 u stage (40/step) and the qkvz/qkv sibling out batch at
m = 4096/8192 (40/step). Bit-identical at either width - the block guard is a
max-reduction and every output element is still one warp accumulating the same
k range in the same order; only the (mi, nj) -> warp map moves.

      `tcw-wall-s1`, four flanks each, HOSTGATE ok (q4km=173.6, control=166.3)
      control  925.26 923.01 926.48 924.01   mean 924.69  spread 0.375 pct
      tcw      925.13 918.85 921.90 914.93   mean 920.20  spread 1.11 pct
                                             -0.485 pct = +0.084 ms/step

  KILLED, and it costs more than it saves. 1024 threads need <= 64 regs and the
  first build died on "too many resources requested for launch"; the
  `__launch_bounds__(WG)` that admits it makes ptxas spill, and the extra warps
  do not pay for the spill. Note the direction agrees with the split-K control
  from the previous rung only in sign of interest: MORE CTAs helped there (64
  -> 256 was worth 0.36 ms), MORE THREADS PER CTA does not.

GATES. Token (`rtt-nt-s1`, `tcw-nt-s3`, `--mode ntcheck --gpu L40S --args
"16"`): control, RT_TAIL and TC_WG all give intra-batch AGREE and stream sha
34cdad2c7570. BIT-IDENTICAL, so the KLD gate is not owed. Build gate ok. Peak
VRAM unchanged at 9292 MiB. Engagement receipts enforced in every wall arm:
both tail markers present exactly once, plus the seven goal4 markers.

POSITION. Within-container control/q4km read 1.3590 (s1), 1.3891 (s2) and
1.3577 (s3); the candidate read 1.3614, 1.3992 and 1.3576. Pooled the rung is
worth +0.30 pct, which moves the 1.379-1.382 baseline to about 1.383-1.386
against the 1.50x bar. THE RUNG ADVANCES ESSENTIALLY NOTHING.
`GGML_MACH1_RT_TAIL` and `GGML_MACH1_TC_WG` stay DEFAULT OFF; both are retained
only as the receipts for the two kills.

WHERE THE NEXT RUNG SHOULD GO. The transform tail is an EXECUTION tail, so the
three ways left to spend it are (a) make one transform finish faster without
more threads per CTA - the FWHT is two dependent MMA passes through shared, and
the second pass cannot start until the first is stored, so a software-pipelined
or single-pass factorization is the only in-kernel move; (b) OVERLAP it - the
routed expert chain (exp_mega, 46 pct, bandwidth bound) and the shared-expert
chain are INDEPENDENT from the post-attention norm until the final add, and
`mach1_fork_state` already runs a second stream with capture-safe events at
nt <= 4 (`GGML_MACH1_FORK_NT`); a shexp chain hidden under exp_mega is ~6
serial launches per layer of latency that costs nothing extra in bandwidth;
(c) stop paying it at all in the ops where the transform is on the critical
path for one token only. Do NOT spend another rung on launch COUNT: this rung
removed 80 launches for 0.05 ms, and the previous one removed walk arithmetic
for 0.00.

## Promotion rule

Resource/static, directional timing, quality, full-model, and production
serving gates are separate. Never report a timing-only arm as quality, never
report a failed-host absolute rate as publishable, and never retain a timing
shadow in a shipping byte ledger. Promote only after a receipt-gated compile,
focused correctness or exact pack oracle, same-container directional screen,
fresh-artifact quality suite when arithmetic changes, counterbalanced full
confirmation, and production server validation.
   RUNG A PHASE 1 KILLED (`nfd-wall-s1`, HOSTGATE ok q4km 168.2
   control 161.0; control interior soft again - pool wobble):
   p16nfd (FORK_DN=0, fold the shexp out into the down kernel)
   LOSES within-draw at B1 -22.5 / B2 -9.8 / B4 -3.6 / B8 -6.6
   percent (B16 +2.5, wobble). The fork runs the whole down WALK
   on s2 overlapped with the mega output path - the separate out
   launch is the price of that overlap, not waste. The census
   misread it. KILL the trio-fusion premise entirely (phase 2
   gu+down would break the same overlap). Keep FORK_DN=1 exactly
   as shipped. Remaining 1.10x levers: rung C (head fold bank at
   B1, +3.6 pct sized), rung B (walk_tt, clock-caveated), rung D
   (u/out singles). B1 needs +16 - the identified inventory now
   sums well short of it; the next instrument is a B1 kernel-gap
   profile (sum of ncu durations vs step wall) to size the
   frontend/idle share the fork family hides.
   RUNG C WIN AT B1 (`hb-wall-s1`, HOSTGATE ok q4km 166.5 control
   161.3): p16hb (GGML_MACH1_HEAD_BANK=1, scoped fp16 head bank,
   commit 1b4a2e9df) reads B1 167.0 vs control 161.3 = +3.5
   percent - the census sizing (+3.6) confirmed. B1 = 1.003x vs
   q4 on this draw: PARITY CROSSED at B1 for the first time.
   Other widths wobble-neutral (bank engages at nt==1 only).
   Cost: ~1.0 GiB VRAM bank, tolerance-class logits (mmvf fp32
   order; np1/np2 coherence gate passed, np2 sha certified since
   the bank is inert there). PROMOTE p16hb to the candidate
   stack. B1 standing: ~1.00x, needs ~+10 percent more for the
   1.10x goal. Next instruments: the B1 kernel-gap profile (sum
   of ncu durations vs step wall - sizes the frontend/idle share
   the fork family hides) and rung B's same-container walk_tt
   check.
   WM2 NEUTRAL (`wm2-wall-s1`, gate DEGRADED on the control B1
   leg only - the rest of that control curve is the healthiest
   ever: B2 279.8 / B4 401.0 / B8 636.8 / B16 983.0): walk_tt
   MINB=2 within-draw B2-16 deltas are +-1 percent = NEUTRAL,
   killed without redraw (the wg512 walk barely runs at nt=1, so
   B1 loses nothing). BONUS RECEIPT: on this healthy-interior
   draw the control reads B8 1.106x and B16 1.41x vs q4 - B8
   CLEARS 1.10 on healthy hosts; its "margin" question is draw
   variance, not a missing lever. The 1.10x scoreboard with the
   head bank promoted: B1 ~1.00 (bank +3.5), B2 ~1.00, B4
   0.95-0.96, B8 1.10 healthy, B16 1.33-1.41. Remaining gaps: B2
   ~+10, B4 ~+15, B1 ~+10.
   RUNG B CLOSED (`wtt-ab-s1`, same-container): the 27-vs-16 us
   walk_tt gap is real at equal clocks BUT resolves to different
   ops - at nt >= 5 the big walks run the MMA16 batch forms, so
   the nt=2 pair walk is the fallback for exactly the ops whose
   padded-MMA admission rung 2a killed at nt {2,4}. ttc=4 at nt=2
   keeps z=1 and only pads slots. No new lever. 1.10x standing:
   B1 ~1.00 (head bank), B2 ~1.00, B4 0.95-0.96, B8 healthy-1.10,
   B16 1.33-1.41. Open gaps B1/B2 ~+10, B4 ~+15; remaining sized
   ideas: rung D (u/out singles fusion ~0.2-0.3 ms at nt 1-4),
   the B1 kernel-gap profile, and re-examining the B4 shape
   (0.95 at nt=4 - which families grow vs nt=2?).
   NT=4 CENSUS (`ncu-int4-s1`): mega 80.0us avg (30 pct of window),
   HEAD_MM_WARP 670.8us ONCE = ~6.7 percent of the whole B4 step
   in one launch (nt 2-4 run the warp head - tg is nt==1, mma16 is
   nt>4), walk_tt_pair 38.3us x ~0.8/layer, shexp trio ~27us/layer,
   transforms ~12us/layer. THE SUCCESSOR RUNG (D-prime): extend
   HEAD_BANK to nt<=4 with ONE batched fp16 GEMM over the existing
   bank (cublas HGEMM, n=nt; one ~1 GiB bank read serves all
   tokens; per-token mmvf loops would re-read the bank and lose).
   Sized: B4 saves ~370us (+3.7 pct step), B2 ~250us (+3.5 pct),
   composing with the banked B1 win. After D-prime the standings
   would be ~B1 1.00 / B2 1.035 / B4 0.99 - still short of 1.10 at
   those widths; the remaining gap stays mega-floor-shaped. But it
   is the strongest sized lever left; build it, AGREE-gate np 2/4,
   wall.
   MERGE CERTIFIED (`merge-nt-s1`, tree = the mach1/main merge at
   4f37e1ada): np 2/8/16 all AGREE with the certified continuation
   in every stream (26/26) and the np16 stream sha reads
   34cdad2c7570. The composed stack - main's NTLOW chunked forms,
   TC widening, gfork gate/up overlap + this line's NTLO, HEAD_BANK
   (mmvf + GemmEx), NOSTAGE, qsplit pflags, OR'd admissions - is
   byte-identical to the certified stream on the goal envs. Both
   branches carry it; every commit now pushes to mach1/main too.
   NEXT: the D-prime wall on the merged tree (q4km / p16ntlo /
   p16hb at npl 1,2,4,8,16) - the head-bank GemmEx at nt 2-4 rides
   a stack that now also carries main's gfork overlap, so the B2/B4
   rows measure the composition.
   DP-WALL-S2 (merged tree, HOSTGATE ok q4 168.6 control 161.2):
   control 161.2/235.5/380.4/623.7/944.1 - B2 in the established
   wobble band (same tree hit 280 on wm2-wall), B16 healthy. The
   D-PRIME PREMISE VANISHED IN THE MERGE: main's chunked mma16
   head serves nt 2-4 (the 671us warp head is gone there), so the
   bank GEMM raced the MMA head and read neutral (-0.3/-0.4 at
   B2/B4). Reverted HEAD_BANK to nt==1 scope. B1 head bank read
   -10 percent THIS draw vs +3.5 on hb-wall-s1 - suspect the 1
   GiB bank fill landed in the timed window; needs a clean
   re-race with a warm fill before it stays in the candidate
   stack. The B2/B4 1.10x question now rests on the merged
   chunked forms - best observed same-draw ratios 1.005 at B2
   and 0.96 at B4 (wm2 draw). Next: a multi-draw healthy-host
   baseline of the merged control before further rungs.
   MBASE-WALL-S1 DISCARDED (interior-degraded host, worst yet): q4
   healthy 170/275/407/561/680 while control read 164/203/321/493/
   785 - B16 785 is far below the 900-983 band and the tree delta
   from dp-wall-s2 is one env-off line, so this is the pool's
   mach1-interior degradation, not code. The B1-only host gate
   passed it (163.8 >= 150): GATE HARDENED with a control B16
   floor (>= 850) so interior-degraded draws self-discard when the
   sweep includes B16. Redraw queued.
   MBASE-WALL-S2: q4 healthy (173/279/419/580/699), control B1
   127.4 FAILS the gate while B16 930 passes the new floor - the
   degradation moved legs. SIX-DRAW CONTROL TABLE (same-tree
   family, q4 healthy on all):
     nfd   161/222/323/568/882   interior soft
     hb    161/232/343/580/904   interior soft
     wm2   133/280/401/637/983   B1 degraded, interior HEALTHY
     dp2   161/235/380/624/944   B2 soft
     mb1   164/203/321/493/785   interior collapsed (discarded)
     mb2   127/232/380/624/930   B1 degraded
   B2 IS BIMODAL (~233 vs ~278 clusters) and some leg degrades on
   nearly every draw while q4 never does. HYPOTHESIS: mach1-induced
   thermal throttling - the spin-heavy mega pulls near-100 percent
   issue power; host cooling decides which legs throttle. If so the
   wobble is intrinsic to the stack's power profile on this pool
   and single draws cannot certify; the scoreboard should use
   PER-WIDTH MEDIANS across gate-passing draws, and the honest
   instrument is nvidia-smi clock/power logging inside the lane
   (add sm clock sampling per leg). Redraw queued at a different
   hour; a clocks-logged draw decides the hypothesis.
   POOL PAUSE: three consecutive mach1-degraded draws (control B1
   68.3 / 115.5 / 127.4 with q4 healthy on each; the chunk-floor
   A/B never got a usable p16nlm8 leg). Applying the recorded
   lesson - redraw at a different hour, do not hammer the pool.
   Resume armed for +3h: chunkab-wall-s3 answers whether the
   pre-merge ntlo paths at nt 2-4 (NTLOW_MIN=8) restore the
   278-class B2 vs the chunked ~236; a yes converts B2's gap into
   a dispatch-priority line. Clock sampler note: nvidia-smi
   produces no output inside the bench containers (files absent
   with both samplers) - the thermal hypothesis stays open on the
   draw-distribution evidence alone.
   PREDICTION BEFORE THE A/B (static dispatch read): pre-merge
   nt=2-4 NEVER ran fused spines (ntlo_min=5 filtered them; the
   census showed walk_tt_pair 27us at nt=2) - the 278-cluster IS
   the walk-path B2. The merged tree's chunked admission starts at
   nt=2, so B2 now rides the chunk-2 MMA form - and chunk-2 MMA
   losing to the walk is rung 2a's padded-MMA kill re-manifesting
   at chunk granularity. PREDICTION: p16nlm8 (NTLOW_MIN=8 -> walk
   below 8) reads ~278-class at B2 and ~395-class at B4; if
   confirmed, the fix is NTLOW_MIN default 8 on this pool's
   hardware class (chunks pay at 8+, walk wins below) - worth ~+18
   percent at B2 and ~+4 at B4 vs the current merged default.
   TWO DEGRADATION MODES SEPARATED (`min8-wall-s2`, npl 1,1,...):
   (1) COLD FIRST LEG - real for BOTH stacks: q4 B1 168.9 -> 183.0
   warm (+8.4 pct), control 161.9 -> 176.1 (+8.7). The npl 1,1
   double-leg absorbs it; score the second row (the parsers and
   hostgate already keep the last B1). Warm-vs-warm B1 = 0.962.
   (2) PROGRESSIVE OVER THE RUN - this draw's control interior
   went soft (233/352/557/774) and the THIRD arm (hb) fully
   collapsed (113/101/223/337/535/816) while first-position q4
   stayed healthy - consistent with every prior "degraded" draw:
   q4 always runs FIRST. Order-reversal draw in flight
   (`order-wall-s1`, mach1 first, q4 last): if degradation follows
   POSITION, the methodology fix is one arm per container (fresh
   host per arm) with cross-container ratios, or order-alternated
   medians. min8 default second-draw receipts meanwhile: B2 0.970,
   B4 0.947, B8 1.127, B16 1.391 (min8-wall-s1, healthy legs).
   ORDER-REVERSAL KILLS THE POSITION THEORY (`order-wall-s1`,
   mach1 first, q4 LAST): p16ntlo in FIRST position read degraded
   (145 warm B1 / 201 / 321 / 580 / 863) and q4 in LAST position
   read perfectly healthy (182.7 warm / 275 / 408 / 562 / 679).
   Degradation follows THE MACH1 ARMS, not run position, and
   varies by host draw - q4 is immune on every draw ever recorded.
   Best remaining fit: the stack's power profile (spin-heavy mega
   at near-100 percent issue) meeting per-host power/cooling
   limits; unprovable here (nvidia-smi produces nothing in these
   containers). METHODOLOGY SETTLED: certify by medians over
   healthy draws (hostgate + band check), warm-B1 scoring
   (npl 1,1,...), order irrelevant.
   THE MIN=8 SCOREBOARD ON HEALTHY DRAWS (two draws each):
     B1 0.962 (warm, one healthy reading)
     B2 0.970 / 0.977
     B4 0.947 / 0.944
     B8 1.127 / 1.135   PASS
     B16 1.391 / 1.339  PASS
   The 1.10x condition needs B1 +14, B2 +13, B4 +16 percent
   against the six-kill mega floor, the balanced spine, and a
   lever inventory measured empty (walk owns nt 2-4 as of min=8;
   the padded and chunked MMA alternatives are both raced kills;
   head bank banked; trio fusion killed by the fork overlap).
   This is the same infeasibility class the 1.5x analysis mapped,
   at a lower bar: closing B1-B4 needs an envelope move (the
   standard-output exporter road remains the sized one) or the
   target scoped to the widths the codec's physics favor - the
   batched region B8/B16 passes with margin.
   GF/TCW RACE, ROUND 1 (`gf-nt-s1` + `gf-wall-s1`): the token
   gate PASSED for both p16gf (GGML_MACH1_FORK_P16=1 - main's
   shexp gate/up region on s2 under the mega, never engaged in
   the goal stack until now) and p16tcw (GGML_MACH1_TC_WG=1024).
   The wall drew the mach1-degradation lottery: control
   160/213/326/560/835, gf and tcw deeper still with a shared B8
   collapse (366/378) that tracks degradation depth, not the
   knobs; q4 healthy as always (183/274/409/564/682).
   INCONCLUSIVE - REDRAW on the next healthy hour: q4km, p16ntlo,
   p16gf, p16tcw, npl 1,1,2,4,8,16. The gfork remains the top
   open interior lever (hides ~13us/layer of shexp under the
   mega, the same overlap physics that killed rung A when removed).
   GF/TCW ROUND 2 (`gf-wall-s2`, q4 healthy, mach1 arms mid-band
   degraded): control 162.6w/235.9/366.9/576.9/913.0; p16gf
   167.2w/228.0/317.6/595.3/938.3 (B1 +2.8, B8 +3.2, B16 +2.8
   pct; B2 -3.4, B4 -13.4 - degraded-leg noise vs effect not
   separable); p16tcw drew another collapsed B8 (416.6) -
   unreadable both rounds. Verdict deferred to a healthy draw
   (the 02:05Z monitor). The gfork's B1/B8/B16 direction is
   promising and consistent with its overlap mechanism; B4 needs
   the clean read before any default flip.
   GFORK BANKED, TCW KILLED (`gf-wall-s3`, healthiest interior
   draw of the campaign; q4 185.6w/280.0/418.6/578.3/694.9):
   p16gf same-draw B8 685.1 (+4.7 pct, 1.185x vs q4) and B16
   1027.3 (+5.5 pct, 1.478x - the best B16 ever, brushing the old
   1.5x goal), neutral B2/B4. FORK_P16 default ON (3eb6893ae,
   both branches). TC_WG=1024 KILLED: three draws, three
   collapsed B8 legs (~370-434) - a real regression, default
   stays off. The control's healthy interior also improved the
   band: B2 0.986, B4 0.967. BEST-KNOWN STACK (ntlo + min8 +
   head bank + gfork), healthy legs: B1 ~0.96 (warm; healthy
   B1+gfork read still missing - the B1 host lottery persists
   across both legs on some draws), B2 0.98-0.99, B4 0.96-0.97,
   B8 1.13-1.185 PASS, B16 1.40-1.478 PASS. Remaining to 1.10x:
   B1/B2 ~+11 pct, B4 ~+14. NEXT: re-race the other fork knobs
   (p16zf4, p16g3f) on the NEW composition - the gfork proved
   overlap wins compose, and those raced neutral only on the old
   stack; then accumulate B1 healthy-leg draws for the median.
   GFORK DEFAULT VINDICATED (`nogf-wall-s1`, soft host, same-draw
   discriminator): default-on control 159.2w/214.4/324.2/582.6/
   907.4 vs FORK_P16=0 161.4w/216.3/329.7/559.1/867.7 - the
   default beats off at B8 +4.2 and B16 +4.6 percent with the
   interior inside noise: the overlap delivers even on degraded
   hosts, and `refork-wall-s1`'s 55-class collapse was the worst
   host lottery on record, not the tree (second occurrence of
   that class; chunkab-s1 was the first). The zf4/g3f re-race
   verdict stays unresolved after two burned draws - PARKED (old
   stack read them neutral; expected value low). CAMPAIGN
   POSITION, all defaults banked on both branches (min8 + head
   bank + gfork): B8 1.13-1.185 PASS, B16 1.40-1.478 PASS, B2
   0.98-0.99, B4 0.96-0.97, B1 ~0.96 warm-healthy. The interior
   gap to 1.10x-everywhere is +11-14 percent with the measured
   lever inventory empty - the envelope decision (standard-output
   exporter or target scope) is the remaining road.
   RUNGS G AND H BUILT AND GATED, VERDICT PENDING A USABLE HOST
   (user cleared the goal hook; the campaign continues on
   judgment). The "empty inventory" claim was WRONG and is
   retracted with receipts: (G) the trio fork (FORK_NT, default
   1) never engaged at nt 2-4 - p16fn4 races it at 4, hiding the
   gu+down walks under the mega at exactly the interior widths
   (sized +6-8 pct; schedule-only, values exact); (H) min8
   brought the 500-670us warp head BACK at nt 2-4, so the bank
   GemmEx was restored (8d897c903) - sized +3.5-3.7 pct at B2/B4;
   (I, reserve) the u/out single-sandwich fusion was deprioritized,
   never raced (~+3). The composed projection puts B2 at ~1.08-
   1.10 and B1/B4 at ~1.04-1.07. p16all token gate PASSED at np
   1/2/4/8 (`gh-nt-s1`). THE WALL (`gh-wall-s1`) drew the third
   catastrophic-class host of the night (control 23.8 at B1, 10x
   down, q4 healthy - too deep for clock capping; offload state
   unverifiable from the saved slice). Do not conclude anything
   from catastrophic-class draws; redraw `gh-wall` (q4km, p16ntlo,
   p16fn4, p16all, npl 1,1,2,4,8,16) on the next healthy window
   and score with the median scoreboard. If G/H price at their
   sizing, flip FORK_NT default to 4 and HEAD_BANK default on,
   then build rung I as the closer.
