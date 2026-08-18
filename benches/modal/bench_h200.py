"""Paired decode bench on one H200, following benches/h200-mach1-vs-bf16.md.

One container, alternating arms, N rounds, llama-bench -fa 1 -p 2048 -n 128.
Source ships as a git-archive tarball on the volume (put there by run_bench.sh);
build is ccache-backed so warm rebuilds are minutes.

    bash benches/modal/run_bench.sh            # tar source, put, run default arms
    python -m modal run benches/modal/bench_h200.py --arms m1,bf16 --rounds 3

Arms (comma-separated): m1, fold, foldq, bf16, q8, q4km - see ARMS below.
Results land in /vol/bench/results/<tag>/ and are summarized on stdout.
"""

import hashlib
import json
import os
import re
import subprocess
import threading
import time

import modal

APP = modal.App("mach1-h200-bench")
VOL = modal.Volume.from_name("mach1-build-cache")

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "ninja-build", "ccache", "git", "curl", "libcurl4-openssl-dev")
    .env({"CCACHE_DIR": "/vol/ccache", "CCACHE_MAXSIZE": "20G",
          "CCACHE_TEMPDIR": "/tmp/ccache-tmp", "CCACHE_NOCOMPRESS": "1"})
)

# nsys for the graphs-on kernel profile lane only - a separate image so the
# main lanes' layer cache is untouched. Package name varies by repo snapshot,
# so try the candidates in order.
NSYS_IMAGE = IMAGE.run_commands(
    "apt-get update && (apt-get install -y cuda-nsight-systems-12-8 "
    "|| apt-get install -y nsight-systems-cli || apt-get install -y nsight-systems)"
)

# ncu for per-kernel stall analysis - the nsys builds on these hosts record no
# kernel data, so the decode-ALU beam needs Nsight Compute instead
NCU_IMAGE = IMAGE.run_commands(
    "apt-get update && (apt-get install -y cuda-nsight-compute-12-8 "
    "|| apt-get install -y nsight-compute)"
)

MODELS = {
    "mach1": "/vol/models/Mach-1-Additive-35B.mach1.gguf",
    "bf16":  "/vol/models/Qwen3.6-35B-A3B-bf16.gguf",
    "q8":    "/vol/models/Qwen3.6-35B-A3B-Q8_0.gguf",
    "q4km":  "/vol/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
}

# arm -> (model key, extra env)
ARMS = {
    "m1":      ("mach1", {}),
    "m1b":     ("mach1", {}),   # duplicate of m1: calibrates smoke determinism
    "ofuse":   ("mach1", {"GGML_MACH1_OFUSE": "1"}),
    "persist": ("mach1", {"GGML_MACH1_EXP_PERSIST": "1"}),
    "expffn":  ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_FFN": "1"}),
    "expmega": ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_MEGA": "1"}),
    # expert megakernel down phase as 4 row groups of 32 column tiles
    "expdn":   ("mach1", {"GGML_MACH1_EXP_DNROWS": "1"}),
    "noffn":   ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_FFN": "0"}),
    "gopt":    ("mach1", {"GGML_CUDA_GRAPH_OPT": "1"}),
    # shexp fusion alone (expffn pinned off) and both fused regions together
    "shexp":    ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_SHEXP_FFN": "1", "GGML_MACH1_EXP_FFN": "0"}),
    "shexpffn": ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_SHEXP_FFN": "1", "GGML_MACH1_EXP_FFN": "1"}),
    "vtiled":  ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_VTILED": "1"}),
    "qkvb":    ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_QKV_BATCH": "1"}),
    "gdnf":    ("mach1", {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_GDN_FUSE": "1"}),
    # full GDN glue region: 1 folds the state gather, 2 also folds the conv+silu.
    # TC_FWHT is NOT pinned here - it is default ON since round 40, so pinning
    # it off (as the older fusion arms do) would confound the pair against m1
    "gdnfull": ("mach1", {"GGML_MACH1_GDN_FULL": "1"}),
    "gdnfull2":("mach1", {"GGML_MACH1_GDN_FULL": "2"}),
    # round-45 batch-scaling probe: the expert ops only take the dense
    # decode-then-GEMM path at P = n_used*n_tok >= DENSE_MIN (1024), i.e.
    # never below B = 128, while the teacher's MUL_MAT_ID batches from B = 2.
    # Round 44 measured our ratio DEGRADING 0.544 -> 0.430 from B = 1 to 16.
    "dmin32":  ("mach1", {"GGML_MACH1_DENSE_MIN": "32"}),
    "dmin64":  ("mach1", {"GGML_MACH1_DENSE_MIN": "64"}),
    "dmin128": ("mach1", {"GGML_MACH1_DENSE_MIN": "128"}),
    # glue floor: every mach1 weight op skipped, so what is left is the glue.
    # OUTPUT IS GARBAGE - never run these with smoke. The GDN region is the
    # only fused region live under ABLATE (it carries no mach1 stage), so
    # these three price the same graph with 0, part and all of it fused.
    "ablate":   ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_CUDA_DISABLE_GRAPHS": "1",
                           "GGML_MACH1_GDN_FUSE": "0"}),
    "ablatef":  ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    "ablatex":  ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_CUDA_DISABLE_GRAPHS": "1",
                           "GGML_MACH1_GDN_FULL": "1"}),
    "ablatex2": ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_CUDA_DISABLE_GRAPHS": "1",
                           "GGML_MACH1_GDN_FULL": "2"}),
    # the same floors with CUDA graphs left ON - the production replay mode
    "ablateg":  ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_MACH1_GDN_FUSE": "0"}),
    "ablategf": ("mach1", {"GGML_MACH1_ABLATE": "255"}),
    "ablategx": ("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_MACH1_GDN_FULL": "1"}),
    "ablategx2":("mach1", {"GGML_MACH1_ABLATE": "255", "GGML_MACH1_GDN_FULL": "2"}),
    # ---- the graph-replay 2x2 (round 133) -------------------------------
    # Two honest measurements of the SAME H200 decode disagree about what
    # the codec costs, and the disagreement is the finding:
    #
    #                       graphs ON            graphs OFF
    #   full model          m1      95.3         m1ng      ?
    #   glue only (ABLATE)  ablategf 429.2       ablatef   ?
    #
    # graphs ON puts the codec at 10.49 - 2.33 = 8.16 ms of a 10.49 ms
    # token (78%).  The TIME=3 census, which runs graphs OFF, puts the
    # SAME codec ops at ~3.9 ms of an ~8.3 ms token (47%) - and that
    # token is FASTER than the graphs-ON one.  So replay is not paying
    # for itself here: ggml_cuda_graph_check_compability already records
    # fold 80.8 vs fold+DISABLE_GRAPHS 101.0 tok/s for the same reason.
    #
    # Every existing DISABLE_GRAPHS arm also ablates the codec, so the
    # shipping config has NEVER been measured with replay off.  m1ng is
    # that measurement; ablatef completes the square so the codec's cost
    # can be differenced within each column instead of across them.
    # bf16ng is the control: if the teacher speeds up too the flag is
    # just globally good on this box and says nothing about mach1.
    "m1ng":    ("mach1", {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
    "bf16ng":  ("bf16",  {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # replay off stacked on the best glue fusion (ablategx2 was the top
    # floor at 433.7).  TC_FWHT is left at its default ON, unlike the
    # older fusion arms, so this pairs cleanly against m1ng.
    "m1ngx2":  ("mach1", {"GGML_CUDA_DISABLE_GRAPHS": "1",
                          "GGML_MACH1_GDN_FULL": "2"}),
    # GGML_MACH1_EXP_MEGA already DEFAULTS to 1, so the "expmega" arm above
    # differs from m1 only by its TC_FWHT=0 pin - it has never tested the
    # fusion.  This turns the fusion OFF, which is the arm that does.  The
    # r133 census shows all three EXP_MM nodes per layer being timed
    # individually, and a fused region is skipped before it is ever timed,
    # so the matcher appears not to be engaging on this model.  If that is
    # right, expmega0 measures IDENTICAL to m1 and the dominant tier is
    # paying ~80 launches/token it was supposed to have stopped paying.
    "expmega0": ("mach1", {"GGML_MACH1_EXP_MEGA": "0"}),
    # what whole-layer expert fusion is WORTH, priced before it is built.
    # The prime suspect for the rejection is the demoted-tier guard, which
    # the executor does not need (it reads src[0]/2/3/8, never src[1]), so
    # forcing past it fuses the region and omits the demoted correction.
    # OUTPUT IS NUMERICALLY WRONG - never run this arm with smoke.  If it
    # measures the same as m1 the blocker was NOT the demoted tier, and
    # graphstat's expffn:nofuse:line<N> says which guard it actually is.
    "expfuseforce": ("mach1", {"GGML_MACH1_EXPFFN_FORCE": "1"}),
    # ROUND 150: llama.cpp flag audit - 18 GGML_MACH1_ flags exist in the
    # source that NO arm has ever set.  The same audit on the vLLM side
    # just found MACH1_SPINE_NC=32 worth +8.6/9.2/10.4% on Ada.  These
    # are the ones that are OFF by default and touch a measured tier
    # (EXP_MM 27.8%, RT_MM 8.8% of the decode census).
    "fusert":  ("mach1", {"GGML_MACH1_FUSE_RT": "1"}),
    "rttc4":   ("mach1", {"GGML_MACH1_RT_TC": "4"}),
    "coop":    ("mach1", {"GGML_MACH1_COOP": "1"}),
    "exppc4":  ("mach1", {"GGML_MACH1_EXP_PC": "4"}),
    "exppc16": ("mach1", {"GGML_MACH1_EXP_PC": "16"}),
    "walkrg2": ("mach1", {"GGML_MACH1_WALK_RG": "2"}),
    "walkrg8": ("mach1", {"GGML_MACH1_WALK_RG": "8"}),
    "tcfwht":  ("mach1", {"GGML_MACH1_TC_FWHT": "1"}),
    "tcwalk":  ("mach1", {"GGML_MACH1_TC_FWHT": "1", "GGML_MACH1_TC_WALK": "1"}),
    # knobs measured neutral/negative BEFORE the fusion rungs landed - the
    # occupancy landscape changed, so re-A/B them against current defaults
    "exprows": ("mach1", {"GGML_MACH1_EXP_ROWS": "1"}),
    "walkwg":  ("mach1", {"GGML_MACH1_WALK_WG": "1024"}),
    # The rt walk's own comment records it as 4.1 ms of a 15.4 ms token,
    # blames the shapes that miss the staged row-split, and ships the fix
    # for them behind WALK_WIDE - which defaults to 0.  n = 4096
    # (tiles_y 256, the out_proj shape) falls through to the original
    # one-thread-per-column-tile kernel: 32768 threads where the part
    # wants ~270000, plus the ~12x read amplification the staging exists
    # to remove (a warp touches 32 distinct 128 B lines, ~4 KB moved to
    # deliver 320 B, measured ~168 GB/s effective).
    # These were left off from Hopper measurements.  Read amplification
    # costs BANDWIDTH, which is what Ada is short of and Hopper is not,
    # so the Ada verdict does not follow from the Hopper one.
    "walkwide": ("mach1", {"GGML_MACH1_WALK_WIDE": "1"}),
    # WIDE covers tiles_y == 256 and WG covers tiles_y <= 128 - disjoint
    # shape sets, so together they cover both spine widths in the model
    "walkboth": ("mach1", {"GGML_MACH1_WALK_WIDE": "1",
                           "GGML_MACH1_WALK_WG": "1024"}),
    # LUT1K only exists inside the WIDE/WG branches (they pick the _s_
    # kernel), so on its own it is dead for every shape in this model -
    # it has to be stacked.  Bit-exact by construction: one LDS.64 of a
    # float2 replaces two random-row LDS.32, with the sign folded into
    # the table, and the accumulation order is unchanged.
    "walkall":  ("mach1", {"GGML_MACH1_WALK_WIDE": "1",
                           "GGML_MACH1_WALK_WG": "1024",
                           "GGML_MACH1_LUT1K": "1"}),
    # "MEASURED NEUTRAL (H200, paired, 3 rounds) ... the head stopped
    # being memory-instruction bound once the 32-way bank conflict was
    # fixed."  That premise is a Hopper premise; the head tier is 3.5x
    # off roofline on Ada, where the 5x sector reduction still buys
    # bytes.  Re-tested on the card, not inherited from the other one.
    "headstage": ("mach1", {"GGML_MACH1_HEAD_STAGE": "1"}),
    "expfast": ("mach1", {"GGML_MACH1_EXPFAST": "1"}),
    "ufuse512":("mach1", {"GGML_MACH1_UFUSE_MAXB": "512"}),
    "fold":    ("mach1", {"GGML_MACH1_FOLD": "1"}),
    "foldq":   ("mach1", {"GGML_MACH1_FOLD": "1", "GGML_MACH1_FOLDQ": "1"}),
    # round-44 defect: expert ops only take the dense decode+GEMM path at
    # n_used*n_tok >= 1024, i.e. never below B = 128, so at B = 4..16 they
    # run per-slot walks with no reuse across the batch
    "dm256":   ("mach1", {"GGML_MACH1_DENSE_MIN": "256"}),
    "dm64":    ("mach1", {"GGML_MACH1_DENSE_MIN": "64"}),
    "dm16":    ("mach1", {"GGML_MACH1_DENSE_MIN": "16"}),
    # ROUND 204: the dense bank path decodes the whole matrix ONCE and
    # runs a real GEMM against it - the amortize-decode-across-rows
    # kernel - and it is gated at 4 tokens.  At B=2 the gate is shut, so
    # two tokens go through the per-weight walk, and Mach-1 at B=2
    # (134.38) is SLOWER than at B=1 (139.42).  These move the gate.
    # ROUND 205: the z-bank tier - packs decoded weights to q8_0 (1 B/weight
    # against the fp16 bank's 2) and hands them to llama.cpp's OWN mmvq
    # kernel instead of a hand-written walk.  Built, marked UNCERTIFIED,
    # defaulting off, and only wired for nt == 1.
    #
    # ROUND 207 - ZBANK ALONE CANNOT ENGAGE IT.  The tier reaches its bank
    # through mach1_bank_get, which returns nullptr unless
    # mach1_bank_enabled(), and that reads GGML_MACH1_BANK (or FOLD) - never
    # ZBANK.  So zb is null, zbank_served stays false, and the walk runs as
    # usual.  What ZBANK=1 does on its own is trip the four
    # mach1_zbank_enabled() guards that switch OFF the shexp, v-tiled, QKV
    # and cooperative fusions.  The round-205 arm therefore priced three
    # lost fusions, not this tier; both flags are required.
    "zbank":   ("mach1", {"GGML_MACH1_ZBANK": "1", "GGML_MACH1_BANK": "1"}),
    # The control zbank needs: BANK=1 alone pays the SAME fusion loss (the
    # guards test mach1_bank_enabled too) and serves nt == 1 from an fp16
    # bank at 2 B/weight.  bank -> zbank is then a clean one-variable step,
    # 2 B/weight through our walk against 1 B/weight through llama.cpp's
    # own dp4a mmvq, and m1 -> bank prices banking by itself.
    "bank":    ("mach1", {"GGML_MACH1_BANK": "1"}),
    # ROUND 208: the lane-addressed codebook, taken from the canonical
    # kernels.  Ours indexes shared memory by the decoded row, which is
    # pseudorandom per lane, so every lookup takes a data-dependent bank
    # conflict - and twice, because a0 and a1 live in separate fp32 arrays.
    # Theirs folds the lane into the ADDRESS (row*REP + lane) over a table
    # replicated REP times, which makes the bank a function of the lane
    # rather than the data: conflict-free by construction, one packed half2
    # load instead of two fp32 loads, and the sign an XOR of bit 15 rather
    # than a branch.  REP is the replication factor; 32 is fully
    # conflict-free but does not fit in static shared here.
    "lutx4":   ("mach1", {"GGML_MACH1_LUTX": "4"}),
    "lutx8":   ("mach1", {"GGML_MACH1_LUTX": "8"}),
    "lutx16":  ("mach1", {"GGML_MACH1_LUTX": "16"}),
    "dense1":  ("mach1", {"GGML_MACH1_DENSE_MIN_TOK": "1"}),
    "dense2":  ("mach1", {"GGML_MACH1_DENSE_MIN_TOK": "2"}),
    # ROUND 213: the integer-lattice walk. The V8 alphabet is s0*z (z4 doc:
    # z in {-4..4}), so the mega walk's per-state gather-and-MAC collapses to
    # two dp4a against the tile's activations quantized to q8 once per pair.
    # Tolerance-class numerics (activation q8 + s0*z reconstruction), like
    # TC_FWHT: smoke DIFFERS vs m1 is expected, certification is the KLD gate.
    "zdp":     ("mach1", {"GGML_MACH1_ZDP": "1"}),
    "zdpng":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # ROUND 46 (chainbench lane), arm P7: the same integer walk with the
    # activations carried at 15 bits as two int8 planes (q = 128*qh + ql, 4
    # dp4a). rel-RMS 9.2e-5 on the probe against q8's 7.9e-3. EXP_DP4A implies
    # the integer walk, so dp4a vs m1 prices the whole rung and dp4a vs zdp
    # prices the second plane.
    "dp4a":    ("mach1", {"GGML_MACH1_EXP_DP4A": "1"}),
    # EXP_DP4A converts the EXPERT walk only; GGML_MACH1_ZDP also converts the
    # rt spine (rows_v, qkv batch, both shexp walks - round 218), so zdp vs
    # dp4a is not like-for-like. zdpx is zdp's expert half alone, which is what
    # dp4a pairs against; zdpx4 is the stack, spine q8 + the expert two-plane.
    "zdpx":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_ZDP_RT": "0"}),
    "zdpx4":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_EXP_DP4A": "1"}),
    # the split expert path (mega off), where the nibble table is in GLOBAL:
    # expmega0 is its control, spdp4a its integer walk
    "spdp4a":  ("mach1", {"GGML_MACH1_EXP_MEGA": "0", "GGML_MACH1_EXP_DP4A": "1"}),
    # ROUND 213: each mega tile task recomputes its item's activation FWHT -
    # 8x redundant in gate|up, up to 32x in down. MEGA_U1 computes each u
    # once behind the existing counter-spin pattern. BIT-EXACT: same input,
    # same block-wide FWHT, one writer per vector - smoke must MATCH.
    "megau1":  ("mach1", {"GGML_MACH1_MEGA_U1": "1"}),
    "zdpu1":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_MEGA_U1": "1"}),
    # ROUND 215: the qkv-batch walk allocates mmax*4 = 32 KB dynamic shared on
    # EVERY block for an out fold only the last block per op runs - 2 blk/SM
    # on Ada, 2.7 waves. TC_WALK moves the fold to the fp16 TC workspace
    # (smaller smem); OFUSE folds out_proj's separate rt_out launch into its
    # walk. Both shipped, neither measured on Ada.
    "zdptcw":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1"}),
    "zdpof":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_OFUSE": "1"}),
    "zdptwof": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_OFUSE": "1"}),
    # ROUND 216: per-slot h release in the mega - down tasks wait on their own
    # slot's h instead of the global CNT_H barrier, so slot chains overlap the
    # gate|up tail. BIT-EXACT (scheduling only): smoke must MATCH.
    "zdpsr":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_MEGA_SLOTREL": "1"}),
    # ROUND 217: race the expert-tier FORMS on Ada under the current stack.
    # ffn2: the two-launch fused form instead of the mega's ~8 grid-wide
    # phases (never raced since the 64 KB compaction; TC_FWHT left default).
    # g2: GDN_FULL=2 folds the state gather + conv/silu glue (default off,
    # best ablation floor on H200, never a serving arm on Ada).
    "zdptcwsr":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_MEGA_SLOTREL": "1"}),
    # ROUND 218: ZDP now also covers the rt spine (rows_v, qkv-batch, shexp
    # walks decode s0*z through dp4a against q8 activations). zdpexp pins the
    # spine part off for attribution.
    "zdpexp":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_ZDP_RT": "0", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2"}),
    # ROUND 219: unmeasured-on-Ada knobs stacked on the round-218 best.
    # dn: EXP_DNROWS=1 - the mega down phase at mff 512 leaves 96 of 128
    # threads idle at DNCT 128; the poc measured -29% on that phase.
    "zdpdn":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_EXP_DNROWS": "1"}),
    "zdpfw":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FWHT_WG": "1024"}),
    "zdphs":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_HEAD_STAGE": "1"}),
    # ROUND 225: out_proj (wg 1024, no tcw) recomputes its n=4096 FWHT in
    # every block under UFUSE; =0 moves it to one rt_u_tc launch per op
    "zdpnouf":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_UFUSE_MAXB": "0"}),
    # ROUND 226: the per-slot expert megakernel - one plain block per routed
    # slot, the whole gate|up|GLU|down chain in shared, no fences or spins.
    # BIT-IDENTICAL to the mega by construction: smoke must MATCH the mega
    # arm's stream (both DIFFER vs plain m1 through zdp).
    "zdpslot":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_EXP_SLOT": "1"}),
    # ROUND 228: the cross-region fork - shexp gate|up on a second stream,
    # concurrent with the routed expert region. Same kernels, same values,
    # different schedule: smoke must MATCH the unforked arm's stream.
    "zdpfork":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1"}),
    # ROUND 230: mega gathers straight from L1/L2, no smem staging (r229:
    # the chain is 25 of 36 us and the staging is on every block's path)
    "zdpforkns": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NOSTAGE": "1"}),
    # ROUND 233: stack re-tune under the fork
    "zdpfk0tcw": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1"}),
    "zdpfk0g2":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_FORK": "1"}),
    "zdpfkhs":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_HEAD_STAGE": "1"}),
    # ROUND 238: the r237 census priced the batch walk at 5.4 waves - the TC
    # fold's 77 KB workspace caps residency at 1 block/SM. OSPLIT: u and out
    # each as ONE batched TC launch (block b serves op b), the walk between
    # them with no dynamic shared at 3 blocks/SM. Same TC transforms on the
    # same values, so the smoke stream must MATCH the fold path's.
    # GDF: the mega's GLU block continues into the down-u FWHT while h is
    # still in shared and publishes dbuf; down tasks load it instead of each
    # recomputing su*h + FWHT(512). Same fp32 expressions: smoke must MATCH.
    "zdpfosp":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKVB_OSPLIT": "1"}),
    "zdpfgdf":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_GDF": "1"}),
    "zdpfog":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKVB_OSPLIT": "1", "GGML_MACH1_MEGA_GDF": "1"}),
    # ROUND 239: cross-region forks - the class that has actually paid.
    # fz: the GDN z-gate walk rides s2 under the conv/state chain that never
    # reads it; the gdn_full exec joins main to ez ahead of the gated norm.
    # fdn: the shexp down WALK rides s2 (it only needs s2's own gu product);
    # the fold that reads moe_out runs as a small epilogue on main.
    # Both are schedule-only: same kernels for fdn (smoke MATCH expected);
    # fz swaps the batched pair for the single-op walks (same math, same
    # values expected - any diff falls to the KLD gate).
    "zdpfz":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_FORK_Z": "1"}),
    "zdpfdn2":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_FORK_DN": "1"}),
    "zdpfzdn":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_FORK_Z": "1", "GGML_MACH1_FORK_DN": "1"}),
    # ROUND 240: XPERM - the tiled->grouped input reorder ahead of ssm_out
    # (the census's 136 us/token m1-only CONT) becomes the walk's u-prologue
    # gather. Address-only: smoke must MATCH.
    "zdpfxp":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_XPERM": "1"}),
    "zdpfxpdn":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_XPERM": "1", "GGML_MACH1_FORK_DN": "1"}),
    # ROUND 241: MEGA_HALF - the mega at WG 512 with the 8192-row staged
    # table (44 KB): two blocks per SM, each phase's tile rows split twice as
    # wide, staging halved. High table rows read L2; identical words, so
    # smoke must MATCH.
    "zdpfmh":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "1"}),
    # the r241 hang bisection: 2 = WG 512 alone, 3 = half table alone
    "zdpfmh2":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "2"}),
    "zdpfmh3":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "3"}),
    "zdpfmh4":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "4"}),
    # ROUND 245: MEGA_SPLIT - the widened mega as two plain launches cut at
    # the H boundary. No cooperative launch, so no residency promise to
    # betray; the kernel boundary is the join. Values identical to the mega
    # (same tiles, same counters, same folds): smoke must MATCH.
    "zdpfms":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_SPLIT": "1"}),
    "zdpfmsg":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_SPLIT": "1", "GGML_MACH1_MEGA_GDF": "1"}),
    # ROUND 248: pricing the encode-side ask - per-stage transform ablations
    # (OUTPUT IS GARBAGE, stopwatch only, never smoke). Ablation disables the
    # batching matchers, so the control is the same stack UNBATCHED and the
    # DELTAS against it price each transform class's wall cost:
    # u = input FWHTs (the shared-su ask), out = output FWHTs (the one-sided
    # rotation ask), uo = both spine sides, allt = + the expert tier's.
    # ROUND 249: GDN_FULL level 3 - the region swallows the beta/alpha
    # projections and the conv-state round trip (2 mul_mats, get_rows, concat,
    # cpy per GDN layer). Projection dots are ulp-class: KLD gate, not smoke.
    "zdpfg3":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "3", "GGML_MACH1_FORK": "1"}),
    # ROUND 252: TC u/out transforms at nt <= 16 (grid.z = token) - the
    # butterfly per-op stages were 8.8/17 us at nt=2. TC numerics: KLD class.
    "zdpftc":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_TC_NT": "16"}),
    # ROUND 254: the mega at nt <= 4 - (token, slot) pairs ride the task loop,
    # so a B=2-4 decode step amortizes the serial phase chain over the batch.
    # nt == 1 path is index-identical (smoke MATCH); ntr adds per-pair release
    # so one token's down phase overlaps another's gate|up tail.
    "zdpfnt":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16"}),
    "zdpfntr":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_MEGA_SLOTREL": "1"}),
    # nt-mega debug: node window + routed ids/weights dump at the first nt>1
    # region (graphs stay on; the san lane forces them off itself)
    "dbgnt":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_DEBUG": "3"}),
    # nt census: the stack + nt-mega with per-kernel timers, graphs off -
    # names the remaining nt>1 costs (the unfused spine triples)
    "dbgtiment": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # ROUND 259: the TT walk - the spine decodes once per token chunk at
    # nt in [2, 16] instead of once per token (and instead of the dense
    # decode-and-apply materialization at nt >= 4)
    "zdpfw":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16"}),
    # ROUND 260: + the out fold - rt_out runs in the TT walk's last block at
    # nt <= 4, cutting the per-op serial chain (walk -> out -> next u).
    # REJECTED (-5.5% B=2, -12% B=4): rt_out pipelines under SIBLING walks.
    "zdpfwo":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_WALK_TT_OF": "1"}),
    # ROUND 261: fused dense prefill - the expert apply decodes weight slices
    # to shared instead of round-tripping the 0.5 GB fp16 bank through global.
    # r262: +18% on down (n=512), -16% on gate/up (k-slicing) - gated n==512.
    "zdpfd":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1"}),
    # ROUND 263: the integer prefill - z-nibble tiles + dp4a for gate/up
    # (n=2048, no k-slicing) stacked on the n=512 fp16 fused down
    "zdpfz":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "1"}),
    "dbgtimez":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "1", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # ROUND 264: + quant-once (level 2) - each (pair, tile) quantized in one
    # pass instead of once per row block; apply values bit-identical
    "zdpfz2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2"}),
    "dbgtimez2": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # ROUND 266: the pair-chunk width - acc[PC][16] register pressure probe
    "zdpfzp4":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_EXP_ZPC": "4"}),
    "zdpfzp2":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_EXP_ZPC": "2"}),
    # ROUND 276: the wide TT walk A/B - zdpfw8 adds WG1024/RG8 on small m
    "zdpfw8":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_WALK_TTW": "1"}),
    # ROUND 279: the GDN full region at batched decode - zdpfg2 lifts the
    # fusion's seq cap to 4 (state still through the graph's gather)
    "zdpfg2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4"}),
    # ROUND 280: the tc FWHT u/out stages at batched decode - the r277 census
    # shows plain rt_u+rt_out cost 2.7 ms/step at nt=2 (one BLOCK per token)
    # where the tc forms cost 0.6 at nt=1 and are already grid.z-sliced; the
    # only gate was GGML_MACH1_TC_NT=1. ufuse/tcw/ofuse stay nt==1 by their
    # own gates, so this flips exactly {rt_u, rt_out}. fp16 transform: np>=2
    # stream hashes legitimately shift - certify by AGREE + tails + np=1 hash.
    "zdpft2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_TC_NT": "16"}),
    # ROUND 281: the composed serving stack - GDN-at-nt (r279) + tc u/out at
    # nt (r280) on top of zdpfz2
    "zdpfgt2":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16"}),
    # census arm for the composed stack
    "dbgtimegt": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # ROUND 282: was the r272 shexp-at-nt win an artifact of plain u/out?
    # zdpfgs1 pins the fused shexp back to nt==1 on the composed stack, so
    # nt>=2 shexp ops ride rt_u_tc + TT walk + rt_out_tc instead
    "zdpfgs1":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_SHEXP_NT": "1"}),
    # ROUND 283: UFUSE_TT - the u stage folds into the TT walk for m <= 4096
    # ops at nt >= 2 (one serial stage removed from 90 of 130 rt ops/step)
    "zdpfu2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_UFUSE_TT": "1"}),
    # ROUND 284: the qkv/qkvz batch regions at nt >= 2 in the bnt_split form -
    # pair-merged u_tcb/out_tcb launches + decode-once TT walks (r275's loss
    # was the sliced walk; r283's rule says merged launches must not
    # replicate per-token work, and these do not)
    "zdpfq2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4"}),
    # ROUND 291: free nt=1 fork arms on the production stack
    "zdpfy1":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_FORK_Z": "1"}),
    # ROUND 290: the residency pin - two blocks/SM for the wg-512 TT walks
    # (free at TT=2 where regs already sit at 64; forced spill at TT=4)
    "zdpfr2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_WALK_MINB": "2"}),
    # ROUND 289: FORK_DN at nt on top of the z fork - the down walk rides s2
    # after gu, only the moe-reading fold stays on main
    "zdpfd4":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1"}),
    # L4 serving campaign's final p16 kernel stack, isolated here from the
    # HTTP/server-only scheduler and sampling flags. The RT, head, and QONCE
    # branches are exact-nt16 gates; lower-B rows remain a useful receipt that
    # this arm does not silently improve unrelated batch shapes.
    "p16final":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_RT_MMA16": "1", "GGML_MACH1_RT_MMA16_M2": "1", "GGML_MACH1_HEAD_MMA16": "1", "GGML_MACH1_MEGA_QONCE": "1", "GGML_MACH1_DEBUG": "1"}),
    # ROUND 293: prefill env sweep on the production stack. The apply kernel
    # is register-saturated (r266: 255 regs, 2 blocks/SM), so the pair-chunk
    # width trades registers for reuse; RT_TC does the same for the spine
    # apply. Neither has been raced since the stack changed under them.
    "ppzpc2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_EXP_ZPC": "2"}),
    "ppzpc8":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_EXP_ZPC": "8"}),
    "pprttc4":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_RT_TC": "4"}),
    # ROUND 294: the spine apply as a cuBLAS GEMM at prefill (tensor cores;
    # the bank is already fp16 so there is no staging pass). ppcub0 pins it
    # off as the control.
    "ppcub":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1"}),
    "ppcub0":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_DN": "1", "GGML_MACH1_APPLY_CUBLAS": "0"}),
    # ROUND 288: FORK_Z at nt - the qkvz z op rides s2 under the conv/core
    # chain (greenlit by lane B's overlap result), on the FORK_NT stack
    "zdpfz4":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4", "GGML_MACH1_FORK_Z_NT": "4"}),
    # ROUND 287 lane C: the mega as two plain launches at nt >= 2 (the
    # kernel-boundary join vs the DN spin waves, which grow with P)
    "zdpfm2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_MEGA_SPLIT": "1"}),
    # ROUND 287 lane B: the mega/shexp fork at batched decode - the s2 gu
    # overlaps the mega at nt <= 4 (FORK_NT; machinery unchanged, gate lifted)
    "zdpff2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_MACH1_FORK_NT": "4"}),
    # ROUND 287 lane A: cap-lift for the nt 8-16 regime (SHEXP_NT stays 4 -
    # the shexp counter tail has 4 slots; GDN_NT clamps at 8 in-code)
    "zdpfq16":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "8", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "16", "GGML_MACH1_QKVB_NT": "16", "GGML_MACH1_QKVB_WMERGE": "1"}),
    # ROUND 287 lane A: llama.cpp's own graph-opt pass on the final stack
    "zdpfgo":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1", "GGML_CUDA_GRAPH_OPT": "1"}),
    # ROUND 285: the region's walks as ONE pair-select grid (WMERGE)
    "zdpfq3":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16", "GGML_MACH1_QKVZ_NT": "4", "GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVB_WMERGE": "1"}),
    # ROUND 275: the qkvz-at-nt A/B - zdpfv1 caps the GDN in-proj pair at nt=1
    "zdpfv1":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_QKVZ_NT": "1"}),
    # ROUND 274: the qkv-batch-at-nt A/B - zdpfq1 caps it at nt=1
    "zdpfq1":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_QKVB_NT": "1"}),
    # ROUND 272: the shexp-at-nt A/B - zdpfsx1 caps the fused shexp region
    # at nt=1 (the old behavior); the stack default admits nt <= 4
    "zdpfsx1":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_SHEXP_NT": "1"}),
    # ROUND 269: the warp head A/B - zdpfhw0 keeps the old row-per-thread
    # nt>=2 head; the stack default is the warp form
    "zdpfhw0":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_HEAD_WARP": "0"}),
    # ROUND 267: forced residency - launch_bounds min-blocks floor, acc
    # spills to L1 (MMQ-style); the latency-vs-bandwidth discriminator
    "zdpflb4":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_EXP_ZLB": "4"}),
    "zdpflb6":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_EXP_ZDP_APPLY": "2", "GGML_MACH1_EXP_ZLB": "6"}),
    "dbgtimew":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    "dbgtimed":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16", "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # graphstat at nt>1: does replay engage for the B=2 decode graph?
    "dbgstat":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_TIME": "4"}),
    "dbgstatq":  ("q4km",  {"GGML_MACH1_TIME": "4"}),
    # nt-regime instrumentation: the stack with per-kernel timers, graphs off
    "dbgtime":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # graph recon: one-shot node dump around the gdn_full entry (stderr)
    "dbg3":      ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_DEBUG": "3"}),
    "abctrl":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_VTILED": "0"}),
    "abu":       ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_VTILED": "0", "GGML_MACH1_ABLATE": "1"}),
    "abo":       ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_VTILED": "0", "GGML_MACH1_ABLATE": "4"}),
    "abuo":      ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_VTILED": "0", "GGML_MACH1_ABLATE": "5"}),
    "aballt":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_VTILED": "0", "GGML_MACH1_ABLATE": "85"}),
    # ROUND 244: the consolidation stacks of the bit-certified levers
    "zdpfxd":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_XPERM": "1", "GGML_MACH1_SHEXP_DN16": "1"}),
    "zdpfall":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_XPERM": "1", "GGML_MACH1_SHEXP_DN16": "1", "GGML_MACH1_FORK_DN": "1"}),
    "zdpfmhxp":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "1", "GGML_MACH1_XPERM": "1"}),
    # ROUND 242: SHEXP_DN16 - the shexp down walk at RGT 16 (CT 32): the
    # n = 512 shape has 32 column tiles, so RGT 4 idles 96 of 128 threads per
    # vblock. Same partials, same warp-0 shuffle tree: smoke must MATCH.
    "zdpfd16":   ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_SHEXP_DN16": "1"}),
    # MEGA_HALF + EXP_DNROWS: the mega's down phase has the same 96-of-128
    # idle shape SHEXP_DN16 fixes; DNROWS measured dead pre-fork, re-raced
    # under the additive-model lens with the widened mega.
    "zdpfmhdn":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_HALF": "1", "GGML_MACH1_EXP_DNROWS": "1"}),
    # ROUND 243: OSPLIT level 2 - the batch walk pinned at 2 blocks/SM via
    # launch bounds (the register file was the residency cap, so tell ptxas
    # the budget) with the batched TC u/out launches. Same transforms:
    "zdpfo2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKVB_OSPLIT": "2"}),
    "zdpfo2d16": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_FORK": "1", "GGML_MACH1_QKVB_OSPLIT": "2", "GGML_MACH1_SHEXP_DN16": "1"}),
    # ROUND 227: mega phase probes (stopwatch only - output is garbage)
    "zdpmp1":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_MEGA_PROBE": "1"}),
    "zdpmp2":    ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_MEGA_PROBE": "2"}),
    # ROUND 220: qkv-batch walk probes (stagetime only - output is garbage)
    "zdpp1":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_QKVB_PROBE": "1"}),
    "zdpp2":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_QKVB_PROBE": "2"}),
    "zdpp3":     ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_QKVB_PROBE": "3"}),
    "zdptcwffn": ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_EXP_MEGA": "0", "GGML_MACH1_EXP_FFN": "1"}),
    "zdptcwg2":  ("mach1", {"GGML_MACH1_ZDP": "1", "GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2"}),
    "bf16":    ("bf16",  {}),
    "q8":      ("q8",    {}),
    "q4km":    ("q4km",  {}),
    # CPU reference walk: same binary, scalar path. The only controlled way to
    # measure a CPU change - container variance dwarfs the effect (see cpucheck)
    "m1cref":  ("mach1", {"GGML_MACH1_CPU_SCALAR": "1"}),
    # CPU integer-lattice walk: z is an int8 (a nibble in practice), so the
    # lattice table is 256 KB instead of the widened fp32 1 MB, and that table
    # is read at random offsets - residency is the thing being bought
    "m1cint":  ("mach1", {"GGML_MACH1_CPU_INT": "1"}),
    # nibble-packed lattice: 128 KB for the whole expert tier, as the CUDA ZDP
    # repack stores it (8 x 4-bit (z+8) per row)
    "m1cnib":  ("mach1", {"GGML_MACH1_CPU_INT": "2"}),
    # per-OP stream occupancy, the one decomposition comparable across models
    "optm":    ("mach1", {"GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    "optg3":   ("mach1", {"GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1",
                          "GGML_MACH1_GDN_FULL": "3"}),
    "optg2":   ("mach1", {"GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1",
                          "GGML_MACH1_GDN_FULL": "2"}),
    # zdpfgt2 MINUS the two approximating flags (ZDP, EXP_ZDP_APPLY). Everything
    # left is exact restructuring - fusion and scheduling - so this is the
    # configuration that could become the default without an accuracy decision.
    # GDN_FULL=3: the proj-fuse rung - folds the beta/alpha mul_mats AND the
    # 21-node glue window into the PROJ core kernel. ulp-class on the dots.
    "gdn3":    ("mach1", {"GGML_MACH1_GDN_FULL": "3"}),
    "gdn3dbg": ("mach1", {"GGML_MACH1_GDN_FULL": "3", "GGML_MACH1_DEBUG": "1"}),
    # FORK arms a concurrent stream event, and the GDN/shexp fusion matchers
    # are guarded on concurrent_events.empty() - so FORK SUPPRESSES them.
    "nofork":  ("mach1", {"GGML_MACH1_FORK": "0"}),
    # ABLATE=255 skips EVERY mach1 weight stage (rt u/walk/out, expert
    # grp/u/walk/out, head). Output is garbage; the TIMING is the glue floor -
    # what the step costs with a free codec. Decides whether the remaining gap
    # is in the weight path or in the glue.
    "ablall":  ("mach1", {"GGML_MACH1_ABLATE": "255"}),
    "ablwalk": ("mach1", {"GGML_MACH1_ABLATE": "34"}),
    # per-stage ablation: cost of a stage = m1 time - that arm's time.
    # bits: RT_U 1, RT_WALK 2, RT_OUT 4, EXP_GRP 8, EXP_U 16, EXP_WALK 32,
    #       EXP_OUT 64, HEAD 128
    # VALID attribution: every arm here has a NONZERO mask, so all of them sit
    # in the same fusion state (nine sites test ABLATE == 0 and switch fusion
    # off wholesale). Cost of a stage = its leave-one-in arm MINUS ablall.
    # Comparing any of these against m1 (mask 0) is invalid - that was round
    # 328's mistake.
    "abRt":    ("mach1", {"GGML_MACH1_ABLATE": "248"}),   # rt u/walk/out only
    "abExp":   ("mach1", {"GGML_MACH1_ABLATE": "135"}),   # expert grp/u/walk/out only
    "abHd":    ("mach1", {"GGML_MACH1_ABLATE": "127"}),   # head only
    "abWk":    ("mach1", {"GGML_MACH1_ABLATE": "221"}),   # the two walks only
    # spine-walk shape. RG=8/WG=1024 is documented in mach1.cu as the
    # latency-bound answer for small m; note six fusion sites require rg==4,
    # so RG=8 must beat the change AND the fusion it costs.
    # untested knobs aimed at the two dominant terms: the FWHT transforms
    # (~49% of the unfused weight path) and the expert walk's staging
    "fw256":   ("mach1", {"GGML_MACH1_FWHT_WG": "256"}),
    "fw1024":  ("mach1", {"GGML_MACH1_FWHT_WG": "1024"}),
    "zlb4":    ("mach1", {"GGML_MACH1_EXP_ZLB": "4"}),
    "pc16":    ("mach1", {"GGML_MACH1_EXP_PC": "16"}),
    "qbnt4":   ("mach1", {"GGML_MACH1_QKVB_NT": "4", "GGML_MACH1_QKVZ_NT": "4"}),
    "rg8":     ("mach1", {"GGML_MACH1_WALK_RG": "8", "GGML_MACH1_WALK_WG": "1024"}),
    "wg1024":  ("mach1", {"GGML_MACH1_WALK_WG": "1024"}),
    "abRtWk":  ("mach1", {"GGML_MACH1_ABLATE": "253"}),   # rt walk only
    "abExpWk": ("mach1", {"GGML_MACH1_ABLATE": "223"}),   # expert walk only
    "abRtU":   ("mach1", {"GGML_MACH1_ABLATE": "254"}),   # rt u only
    "abRtO":   ("mach1", {"GGML_MACH1_ABLATE": "251"}),   # rt out only
    "abHead":  ("mach1", {"GGML_MACH1_ABLATE": "128"}),
    "abRtUO":  ("mach1", {"GGML_MACH1_ABLATE": "5"}),
    "abExpUO": ("mach1", {"GGML_MACH1_ABLATE": "80"}),
    "abExpG":  ("mach1", {"GGML_MACH1_ABLATE": "8"}),
    # COOP: one cooperative launch replaces rt_u + walk + rt_out - the
    # structural fix the H200 analysis named. Cooperative launch and stream
    # forks may not coexist, so test with FORK on and off.
    "coop":    ("mach1", {"GGML_MACH1_COOP": "1"}),
    "coopnf":  ("mach1", {"GGML_MACH1_COOP": "1", "GGML_MACH1_FORK": "0"}),
    "g3nf":    ("mach1", {"GGML_MACH1_GDN_FULL": "3", "GGML_MACH1_FORK": "0"}),
    "optg3nf": ("mach1", {"GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1",
                          "GGML_MACH1_GDN_FULL": "3", "GGML_MACH1_FORK": "0"}),
    # single-knob sweep on top of the promoted defaults - clean attribution
    "sdn16":   ("mach1", {"GGML_MACH1_SHEXP_DN16": "1"}),
    "sxperm":  ("mach1", {"GGML_MACH1_XPERM": "1"}),
    "szpc2":   ("mach1", {"GGML_MACH1_EXP_ZPC": "2"}),
    "szpc8":   ("mach1", {"GGML_MACH1_EXP_ZPC": "8"}),
    "srttc4":  ("mach1", {"GGML_MACH1_RT_TC": "4"}),
    "sfdn":    ("mach1", {"GGML_MACH1_FORK_DN": "1"}),
    "swm":     ("mach1", {"GGML_MACH1_QKVB_WMERGE": "1"}),
    "shw0":    ("mach1", {"GGML_MACH1_HEAD_WARP": "0"}),
    "gdn4":    ("mach1", {"GGML_MACH1_GDN_FULL": "4"}),
    "exactfg": ("mach1", {"GGML_MACH1_TC_WALK": "1", "GGML_MACH1_GDN_FULL": "2",
                          "GGML_MACH1_FORK": "1", "GGML_MACH1_MEGA_NT": "16",
                          "GGML_MACH1_WALK_TT": "16", "GGML_MACH1_EXP_FUSED_DENSE": "1",
                          "GGML_MACH1_GDN_NT": "4", "GGML_MACH1_TC_NT": "16"}),
    "optq":    ("q4km",  {"GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1"}),
    # bar honesty: the GDN glue fold is weight-format-agnostic, so the bar
    # gets it too if it helps
    "q4kmg2":  ("q4km",  {"GGML_MACH1_GDN_FULL": "2"}),
}

# Profiling-only mirror of the final p16 arm. Per-kernel synchronization makes
# its end-to-end row non-production; consume only the stage census.
ARMS["p16time"] = (
    "mach1",
    {
        **ARMS["p16final"][1],
        "GGML_MACH1_TIME": "1",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
    },
)

# Higher-upside zero-VRAM spine candidate: the TC u transform writes q8 K16
# records into its existing scratch, and the compressed trellis feeds signed
# int8 m16n8k16 fragments directly. No decoded weight bank is allocated.
ARMS["p16rti8"] = (
    "mach1",
    {
        **ARMS["p16final"][1],
        "GGML_MACH1_RT_IMMA8": "1",
    },
)

# Native 4-bpw, zero-persistent-VRAM overlap rung. The second 4.5-KiB stage
# aliases the reduction tile, so the measured 14,848-byte shared footprint is
# unchanged from p16rti8.
ARMS["p16rti8cp"] = (
    "mach1",
    {
        **ARMS["p16rti8"][1],
        "GGML_MACH1_RT_IMMA8_CPASYNC": "1",
    },
)

# Exact aliases for the two validated sibling-batching rungs. Keeping the
# control separate makes the one-knob graph screens and engagement receipts
# explicit even after they are composed with the IMMA8 spine below.
ARMS["p16gdn0"] = ("mach1", dict(ARMS["p16final"][1]))
ARMS["p16gdnb"] = (
    "mach1",
    {
        **ARMS["p16gdn0"][1],
        "GGML_MACH1_QKVZ_MMA16_BATCH": "1",
    },
)
ARMS["p16sib0"] = ("mach1", dict(ARMS["p16gdnb"][1]))
ARMS["p16sib"] = (
    "mach1",
    {
        **ARMS["p16sib0"][1],
        "GGML_MACH1_P16_SIBLING_BATCH": "1",
    },
)

# Same sibling schedule with the native IMMA8 handoff held on and cp.async
# held off. This is the sibling-only cell of the orthogonal 2x2 below.
ARMS["p16i8sib"] = (
    "mach1",
    {
        **ARMS["p16rti8"][1],
        "GGML_MACH1_QKVZ_MMA16_BATCH": "1",
        "GGML_MACH1_P16_SIBLING_BATCH": "1",
    },
)

# Exploit-beam composition: the validated native IMMA8/cp.async spine plus
# both independently validated graph sibling-batching rungs. All tensors keep
# their native Trellis payloads and every added buffer aliases existing scratch.
ARMS["p16combo"] = (
    "mach1",
    {
        **ARMS["p16rti8cp"][1],
        "GGML_MACH1_QKVZ_MMA16_BATCH": "1",
        "GGML_MACH1_P16_SIBLING_BATCH": "1",
    },
)

# Exact-shape work-centric spine candidate. It adds only three transient
# [nt, m] FP32 partial rows for m2048/n4096/nt16 and folds them in the existing
# out launch; the native trellis payload and every other shape stay unchanged.
ARMS["p16ks4"] = (
    "mach1",
    {
        **ARMS["p16combo"][1],
        "GGML_MACH1_RT_IMMA8_SPLITK": "1",
    },
)
# Distinct labels prevent the two control/candidate draws in the interleaved
# C-S-Q-S-C wall from overwriting each other's logs.
ARMS["p16kc0"] = ("mach1", dict(ARMS["p16combo"][1]))
ARMS["p16kc1"] = ("mach1", dict(ARMS["p16combo"][1]))
ARMS["p16ks0"] = ("mach1", dict(ARMS["p16ks4"][1]))
ARMS["p16ks1"] = ("mach1", dict(ARMS["p16ks4"][1]))

# QONCE producer FWHT scale fold on the promoted composition parent: the
# mega's producer normalization rides the FWHT's last butterfly store (bitwise
# records, zero payload/persistent/transient delta). The pack-side fold was
# priced and killed in the replica probe.
ARMS["p16qfs"] = (
    "mach1",
    {
        **ARMS["p16combo"][1],
        "GGML_MACH1_QONCE_FWSCALE": "1",
    },
)
ARMS["p16qf0"] = ("mach1", dict(ARMS["p16qfs"][1]))
ARMS["p16qf1"] = ("mach1", dict(ARMS["p16qfs"][1]))

# Standard-output-basis timing composition on the split-K parent: row-scale
# epilogues replace the TC output FWHTs (spine + expert mega). Wrong-basis
# payloads, so the runtime refuses to engage without GGML_MACH1_TIME=1 and the
# arm is TIMING_ONLY - never a quality or wall claim.
ARMS["p16std"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_RT_STDOUT": "1",
    },
)

# Mega stopwatch probe on the split-K parent (output garbage, stage timing
# only): probe=1 skips the walk tiles, isolating the walk share of exp_mega.
ARMS["p16mp1"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_MEGA_PROBE": "1",
    },
)
# QONCE mega with the DNROWS down phase (DNCT=32): n = mff = 512 gives the
# down walk 32 column tiles, so CT=32 hands the 96 otherwise-idle threads of
# each vblock a row group. Bit-exact by the mach1_exp_mega_tile argument.
ARMS["p16qd32"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_EXP_DNROWS": "1",
    },
)
# QONCE mega with per-slot release: down tasks wait on their own slot's h
# instead of the global P-count, overlapping slot chains with the gate/up
# tail. Runtime flag on the same audited kernel.
ARMS["p16qsr"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_MEGA_SLOTREL": "1",
    },
)
ARMS["p16qdsr"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_EXP_DNROWS": "1",
        "GGML_MACH1_MEGA_SLOTREL": "1",
    },
)
# fused shared-expert out+glu epilogue (bit-identical values, two launches
# and one round trip fewer per MoE layer)
ARMS["p16soglu"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_SHEXP_OGLU": "1",
    },
)
# task widening: each mega task covers tw row chunks, amortizing the q8
# record staging, fences, and counter traffic of the ~3k tasks/layer
ARMS["p16qtw2"] = ("mach1", {**ARMS["p16ks4"][1], "GGML_MACH1_MEGA_TW": "2"})
ARMS["p16qtw4"] = ("mach1", {**ARMS["p16ks4"][1], "GGML_MACH1_MEGA_TW": "4"})
# composition of the scheduling arms
ARMS["p16qall"] = (
    "mach1",
    {
        **ARMS["p16ks4"][1],
        "GGML_MACH1_MEGA_TW": "4",
        "GGML_MACH1_EXP_DNROWS": "1",
        "GGML_MACH1_MEGA_SLOTREL": "1",
    },
)
# every racing scheduling/fusion winner on the split-K parent
ARMS["p16qace"] = (
    "mach1",
    {
        **ARMS["p16qall"][1],
        "GGML_MACH1_SHEXP_OGLU": "1",
    },
)
ARMS["p16qace8"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_MEGA_TW": "8"})
# walk-share probe on the new exploit parent (output garbage, stopwatch only)
ARMS["p16qmp"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_MEGA_PROBE": "1"})
# work-centric split for the 64-CTA m2048 n512 shexp-down spine walk
ARMS["p16k512"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_RT_IMMA8_SPLITK_N512": "1"})
# whole-graph per-OP census (TIME=2) on the new parent: sizes the ~5 ms of
# stock GDN/attention/glue the stage keys never cover
ARMS["p16qt2"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_TIME": "2"})
# GDN state-gather fusion: the delta-net op gathers state rows through s_copy
# at load, deleting the 33.5 MB per-layer get_rows copy (identity-mapping
# aliasing contract; steady-state decode only until server admission)
ARMS["p16sgf"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_GDN_SGF": "1"})
ARMS["p16sg0"] = ("mach1", dict(ARMS["p16sgf"][1]))
ARMS["p16sg1"] = ("mach1", dict(ARMS["p16sgf"][1]))
# the goal composition: sgf + task width 8 + the n512 spine split
ARMS["p16goal"] = (
    "mach1",
    {
        **ARMS["p16sgf"][1],
        "GGML_MACH1_MEGA_TW": "8",
        "GGML_MACH1_RT_IMMA8_SPLITK_N512": "1",
    },
)
ARMS["p16go0"] = ("mach1", dict(ARMS["p16goal"][1]))
ARMS["p16go1"] = ("mach1", dict(ARMS["p16goal"][1]))
# level-2 state fusion: the conv chain collapses into the indexed conv kernel
ARMS["p16goal2"] = ("mach1", {**ARMS["p16goal"][1], "GGML_MACH1_GDN_SGF": "2"})
ARMS["p16h0"] = ("mach1", dict(ARMS["p16goal2"][1]))
ARMS["p16h1"] = ("mach1", dict(ARMS["p16goal2"][1]))
# tw race on the goal2 base: gate/up saturates at tw=8 (32 chunks), the down
# phase (128 chunks) still has widening headroom
ARMS["p16g2t12"] = ("mach1", {**ARMS["p16goal2"][1], "GGML_MACH1_MEGA_TW": "12"})
ARMS["p16g2t16"] = ("mach1", {**ARMS["p16goal2"][1], "GGML_MACH1_MEGA_TW": "16"})
ARMS["p16g2t32"] = ("mach1", {**ARMS["p16goal2"][1], "GGML_MACH1_MEGA_TW": "32"})
# whole-graph per-OP census (TIME=2) on the goal2 stack: ranks what remains
# of the step after both GDN state fusions
ARMS["p16g2cen"] = ("mach1", {**ARMS["p16goal2"][1], "GGML_MACH1_TIME": "2"})
# typed B16 per-op census pair (TIME=3 event windows; megarace forces graphs off)
ARMS["q4cen"]    = ("q4km",  {"GGML_MACH1_TIME": "3"})
# goal3: the batched GDN prep-chain fuse on top of goal2 (six pre-chain
# launches per GDN layer collapse to one; bitwise stock math)
ARMS["p16goal3"] = ("mach1", {**ARMS["p16goal2"][1], "GGML_MACH1_GDN_PREP": "1"})
ARMS["p16j0"] = ("mach1", dict(ARMS["p16goal3"][1]))
ARMS["p16j1"] = ("mach1", dict(ARMS["p16goal3"][1]))
# prefill walk-share probe on the goal3 stack (output garbage, stopwatch
# only): sizes the trellis re-walk share of the prefill exp_mega
ARMS["p16g3mp"] = ("mach1", {**ARMS["p16goal3"][1], "GGML_MACH1_MEGA_PROBE": "1"})
ARMS["p16g3cen"] = ("mach1", {**ARMS["p16goal3"][1], "GGML_MACH1_TIME": "3"})
# topk-moe memrange fix (logits scratch copy -> stock fused kernel engages at
# B16 decode and at the rejected prefill layers). KLD-class: new engagement
# sites shift ulps, so goal4 is a NEW candidate lane, not an np16 sibling.
ARMS["p16goal4"] = ("mach1", {**ARMS["p16goal3"][1], "GGML_MACH1_TOPK_FIX": "1"})
ARMS["p16g4cen"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "3"})
# T-Q-T flanks for the goal4 certification wall (goal2-wall-s1 protocol)
ARMS["p16j4a"] = ("mach1", dict(ARMS["p16goal4"][1]))
ARMS["p16j4b"] = ("mach1", dict(ARMS["p16goal4"][1]))
# nt32 campaign: typed census pair on the goal4 stack, one clean batch width
# per process (BENCH_NPL pins the batched lane's npl per arm). B1 hostgate
# context arms ride the same container.
ARMS["p16g4c16n"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "3",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16g4c32n"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "3",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "32"})
ARMS["p16g4b1"]   = ("mach1", {**ARMS["p16goal4"][1], "BENCH_NPL": "1"})
ARMS["q4b1"]      = ("q4km",  {"BENCH_NPL": "1"})
# env-only nt32 flank: WALK_TT admits nt<=32 (runtime TT walk, no new code)
# and TC_NT=32 keeps the TC u/out transforms on at 32. The exact-nt16 MMA
# paths still disengage above 16 - this brackets what the generic TT walk
# alone recovers at B32.
ARMS["p16g4tt32"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_WALK_TT": "32",
                               "GGML_MACH1_TC_NT": "32"})
ARMS["p16g4t32c"] = ("mach1", {**ARMS["p16g4tt32"][1], "GGML_MACH1_TIME": "3",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "32"})
# nt32 candidate: master switch + the two nt-cap knobs it composes with.
# WALK_TT stays 16 - the census killed the blanket TT-at-32 route (nt32-cen-s1:
# m8192 610 -> 923 ms; the per-chunk re-decode loses to the dense round trip).
ARMS["p16nt32"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_NT32": "1",
                              "GGML_MACH1_MEGA_NT": "32", "GGML_MACH1_TC_NT": "32"})
ARMS["p16n32a"]  = ("mach1", dict(ARMS["p16nt32"][1]))
ARMS["p16n32b"]  = ("mach1", dict(ARMS["p16nt32"][1]))
ARMS["p16nt32c"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_TIME": "3",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "32"})
ARMS["p16nt24c"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_TIME": "3",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "24"})
# per-mach1-kernel stopwatch (TIME=1) at the three widths: times exp_mega and
# every extended stage directly (per-kernel sync - stage table only, not walls)
ARMS["p16nt16t"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_TIME": "1",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16nt24t"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_TIME": "1",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "24"})
ARMS["p16nt32t"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_TIME": "1",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "32"})
# expert-fallback bracket: group-dense ZDP apply admitted at P >= 256
ARMS["p16g4dm256"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_DENSE_MIN": "256"})
# B32-only bisect arms (candidate minus one family each; bisect lane only,
# their B16 rows are not production shapes)
ARMS["p16x32m"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_MEGA_NT": "16"})
ARMS["p16x32s"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_QKVZ_MMA16_BATCH": "0",
                             "GGML_MACH1_P16_SIBLING_BATCH": "0"})
ARMS["p16x32r"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_RT_MMA16": "0"})
ARMS["p16x32g"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_GDN_SGF": "0"})
ARMS["p16x32h"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_HEAD_MMA16": "0"})
# KV-pressure probe: npl=32 exactly fills c=8192; 16384 removes the pressure
ARMS["p16n32k16"] = ("mach1", {**ARMS["p16nt32"][1], "BENCH_C": "16384"})
ARMS["q4kmk16"]   = ("q4km",  {"BENCH_C": "16384"})
# ntlo campaign: master switch admits nt in [2, 16) into the fused stacks
# (QONCE mega + GDN_SGF at nt >= 2; 16-token MMA spines, sibling batching and
# head at nt > 4 via zero-padded scratch). goal4 envs otherwise unchanged;
# nt == 16 paths byte-identical.
ARMS["p16ntlo"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_NTLO": "1"})
ARMS["p16loa"]  = ("mach1", dict(ARMS["p16ntlo"][1]))
ARMS["p16lob"]  = ("mach1", dict(ARMS["p16ntlo"][1]))
# GDN_FULL region cap at its in-code max on top (A/B vs SGF alone at nt=8)
ARMS["p16log8"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_GDN_NT": "8"})
# B2-B8 bisect arms (candidate minus one family; bisect lane only)
ARMS["p16xloq"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_QONCE": "0"})
ARMS["p16xlor"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_RT_MMA16": "0"})
ARMS["p16xloh"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_HEAD_MMA16": "0"})
ARMS["p16xlos"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_QKVZ_MMA16_BATCH": "0",
                             "GGML_MACH1_P16_SIBLING_BATCH": "0"})
ARMS["p16xlog"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_GDN_SGF": "0"})
# typed census pair at the B8 attribution width (one width per process)
ARMS["p16loc8"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_TIME": "3",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "8"})
# graphs-on TIME=4 debug: graphstat + expffn:nofuse reject lines at sub-16 npl
ARMS["p16lodbg"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_TIME": "4"})
# rung 2a: padded spine/pair/trio admitted from nt=2 (head and shexp keep >4)
ARMS["p16lo2"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_NTLO_MIN": "2"})
# graphs-off profile arms: this container's nsys cannot trace graph-node
# kernels, so the B1 kernel/gap decomposition rides plain launches
ARMS["p16g4ng"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_CUDA_DISABLE_GRAPHS": "1"})
ARMS["q4kmng"]  = ("q4km",  {"GGML_CUDA_DISABLE_GRAPHS": "1"})
# rung 2b: expert-sorted pair visitation in the QONCE mega (values bitwise;
# iteration order is the only change - same-expert payload re-reads go cache-hot)
ARMS["p16mo"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_ORDER": "1"})
# the mo B16 stall instruments: graphs-on TIME=4 (recapture loop?) and
# graphs-off TIME=3 census at npl 16 (in-kernel spin shows in the mega row)
ARMS["p16modbg"] = ("mach1", {**ARMS["p16mo"][1], "GGML_MACH1_TIME": "4"})
ARMS["p16moc16"] = ("mach1", {**ARMS["p16mo"][1], "GGML_MACH1_TIME": "3",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
# interior-floor knob race on the ntlo base: built-but-never-raced-on-goal4
# machinery. g3f deepens the nt<=4 GDN region over the projections (ulp class -
# KLD gate before promotion); zf4 rides the qkvz z op on s2 at nt 2-4;
# go2 is llama.cpp's own graph-opt pass.
ARMS["p16g3f"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_GDN_FULL": "3"})
ARMS["p16zf4"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_Z_NT": "4"})
ARMS["p16go2"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_CUDA_GRAPH_OPT": "1"})
# mega two-launch split on the ntlo base (env-only; QONCE is exclusive with
# SPLIT so this races split-sans-QONCE vs the single QONCE mega - the ncu
# profile says the single mega idles at 37 percent on both axes)
ARMS["p16ms"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_SPLIT": "1"})
# rung 4: per-pair slot mega at nt in [2,8] - P independent barrier-free
# blocks replace the P-flat cooperative mega at the interior widths
ARMS["p16sl8"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_EXP_SLOT": "1",
                            "GGML_MACH1_EXP_SLOT_NT": "8"})
# rung 4b: producer barrier -> per-pair record flags in the QONCE mega
# (scheduling only; engages at every qonce width including 16)
ARMS["p16pf"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_QONCE_PFLAGS": "1"})
# rung 5 probe: drop the sgf extra-state maintenance pairs (steady-state only)
ARMS["p16sx"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_SKIP_EXTRA": "1"})
# rung 7: head mma16 admission floor - the warp head at nt 1-4 does a full
# 5-bit vocab decode per <=4-token chunk (506 us/call at nt=2, ~2.5x its
# bandwidth floor); the PARTIAL mma16 form decodes once regardless of nt, so
# padding waste does not price it (the rung 2a spine kill does not transfer)
ARMS["p16hd2"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_HEAD_MMA_NTMIN": "2"})
ARMS["p16hd1"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_HEAD_MMA_NTMIN": "1"})
# rung 8: the round-230 nostage table form rebuilt on the pflags split - the
# walk reads the 64 KB z-table from L2/L1 instead of smem (mio_throttle 4.50
# is the top mega stall; carveout prefers L1 so the table can live there).
# NOSTAGE=2 additionally caps regs via launch_bounds for 3 blocks/SM.
ARMS["p16msn"]  = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_SPLIT": "1",
                             "GGML_MACH1_MEGA_NOSTAGE": "1"})
ARMS["p16msn3"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_SPLIT": "1",
                             "GGML_MACH1_MEGA_NOSTAGE": "2"})
# rung A (1.10x campaign): fold the shexp out stage back into the down kernel
# (FORK_DN=0 -> defer=0) - one fewer launch + no s2 join per layer at nt<=4,
# vs the fork's stream overlap. Schedule-only; values identical.
ARMS["p16nfd"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_DN": "0"})
# rung C (1.10x): scoped fp16 head bank at nt==1 - 478us warp head vs ~250us
# bank read; ~1 GiB VRAM, tolerance-class values (AGREE gate, not sha)
ARMS["p16hb"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_HEAD_BANK": "1"})
# rung B-adjacent: walk_tt occupancy pin (2 blocks/SM on the wg512 forms) -
# unraced env on the family that is ~27 pct of nt=2 kernel time
ARMS["p16wm2"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_WALK_MINB": "2"})
# post-merge B2 A/B: NTLOW_MIN=8 turns the chunked forms off below nt=8 so
# the OR'd pre-merge ntlo paths serve nt 2-4 - isolates whether the chunked
# nt=2 form is the consistent post-merge B2 ~236 (vs pre-merge 278 draws)
ARMS["p16nlm8"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_NTLOW_MIN": "8"})
# main-line knobs never raced on the goal stack: the shexp gate/up fork
# (region on s2 under the expert mega - hides ~13us/layer of interior) and
# the 1024-thread TC transform widening
# rung G: trio fork widened to nt<=4 (FORK_NT default is 1 - gu+down walk
# ride s2 under the mega at the interior widths; schedule-only, values exact)
ARMS["p16fn4"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_NT": "4"})
# rung G+H composed: trio fork + head bank GEMM at nt 2-4
ARMS["p16all"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_NT": "4",
                            "GGML_MACH1_HEAD_BANK": "1"})
# gfork default-on discriminator: explicit off vs the default-on control
ARMS["p16nogf"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_P16": "0"})
ARMS["p16gf"]  = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_FORK_P16": "1"})
ARMS["p16tcw"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_TC_WG": "1024"})
# rung 9: the L1 table on the COOPERATIVE mega (no split) - qs2-wall measured
# the L2/L1-served table 12-16 percent faster than smem within the split
# family at equal occupancy; this rides the same trade on the certified
# coop launch (same barrier structure, word-identical values)
ARMS["p16qns"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_MEGA_NOSTAGE": "1"})
# typed census at the short interior widths (one width per process)
ARMS["p16loc2"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_TIME": "3",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "2"})
ARMS["p16loc4"] = ("mach1", {**ARMS["p16ntlo"][1], "GGML_MACH1_TIME": "3",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "4"})
ARMS["q4c2"]    = ("q4km",  {"GGML_MACH1_TIME": "3",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "2"})
ARMS["p16g4c8"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "3",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "8"})
# ntlow campaign: the exact-16 fused decode stacks generalized DOWNWARD to the
# even widths below 16 (token-tile template on the audited 16-token kernels,
# {16,8,4,2}-token chunking). MEGA_NT stays 16 - the mega's own admission is
# the NTLOW gate below it - and TC_NT 16 already covers nt < 16.
ARMS["p16ntlow"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_NTLOW": "1"})
ARMS["p16nla"]   = ("mach1", dict(ARMS["p16ntlow"][1]))
ARMS["p16nlb"]   = ("mach1", dict(ARMS["p16ntlow"][1]))
# composed with the nt32 rung: NTLOW + NT32 admit every even width in [2, 32],
# so B24 tiles as 16+8 and B32 keeps its two 16-chunks
ARMS["p16ntl32"] = ("mach1", {**ARMS["p16nt32"][1], "GGML_MACH1_NTLOW": "1"})
ARMS["p16nl32a"] = ("mach1", dict(ARMS["p16ntl32"][1]))
ARMS["p16nl32b"] = ("mach1", dict(ARMS["p16ntl32"][1]))
# width-boundary race: the fused stack's per-step cost is fixed in the batch,
# so NTLOW_MIN brackets where admitting it starts paying
ARMS["p16nl8"]  = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_NTLOW_MIN": "8"})
ARMS["p16nl4"]  = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_NTLOW_MIN": "4"})
ARMS["p16nl832"] = ("mach1", {**ARMS["p16ntl32"][1], "GGML_MACH1_NTLOW_MIN": "8"})
# C-S-Q-S-C flank clones for the ntlow certification wall
ARMS["p16nl8a"] = ("mach1", dict(ARMS["p16nl832"][1]))
ARMS["p16nl8b"] = ("mach1", dict(ARMS["p16nl832"][1]))
# per-family bisect on the ntlow candidate (bisect lane only)
ARMS["p16xlm"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_MEGA_QONCE": "0"})
ARMS["p16xls"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_QKVZ_MMA16_BATCH": "0",
                            "GGML_MACH1_P16_SIBLING_BATCH": "0"})
ARMS["p16xlr"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_RT_MMA16": "0"})
ARMS["p16xlg"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_GDN_SGF": "0"})
ARMS["p16xlh"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_HEAD_MMA16": "0"})
# per-mach1-kernel stopwatch (TIME=1) at the low widths, control and candidate:
# times exp_mega and every extended stage directly (per-kernel sync - stage
# table only, never a wall)
ARMS["p16g4t8"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "8"})
ARMS["p16g4t4"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "4"})
ARMS["p16g4t2"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "2"})
ARMS["p16nlt8"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "8"})
ARMS["p16nlt4"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "4"})
ARMS["p16nlt2"] = ("mach1", {**ARMS["p16ntlow"][1], "GGML_MACH1_TIME": "1",
                             "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "2"})
# PP lane sizing: spine bank apply as a tensor-core GEMM (fp16 u, 32F acc;
# activation-rounding quality class - sizing arm until the KLD gate runs)
ARMS["p16pp1"] = ("mach1", {**ARMS["p16goal3"][1], "GGML_MACH1_RT_APPLY_TC": "1"})
# + expert zdp apply as s8 MMA (true-int8 tiles; integer part exact vs dp4a)
ARMS["p16pp2"] = ("mach1", {**ARMS["p16pp1"][1], "GGML_MACH1_EXP_APPLY_MMA": "1"})
# decode-floor stopwatch: the zdp MMA kernel stops after the expand phase
ARMS["p16pp2d"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ZMMA_PROBE": "1"})
# probe=2: warp-skip disabled (A/B lane)
ARMS["p16pp2w"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ZMMA_PROBE": "2"})
# bank residency probe for the m=8192 TC apply anomaly
ARMS["p16pp2c"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_RT_TC_COPY": "1"})
# prefill stream-structure A/B: upstream concurrency gate / fully off
ARMS["p16pp3a"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_CUDA_CONCURRENT_ALL": "0"})
ARMS["p16pp3b"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_CUDA_CONCURRENT_MAX": "0"})
# graphstat probe: does the prefill ubatch ever reach graph replay?
ARMS["p16g4"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_TIME": "4"})
# non-mach1 prefill floor: all mach1 weight stages ablated (garbage output,
# stopwatch only) - what remains is stock ops + glue + copies + gaps
ARMS["p16ppab"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ABLATE": "255"})
# single-family ablations: each arm's PP delta vs p16pp2 = that family's
# wall share at prefill (garbage output, stopwatch only)
ARMS["p16ab2"]  = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ABLATE": "2"})    # rt walk/apply
ARMS["p16ab7"]  = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ABLATE": "7"})    # rt u+walk+out
ARMS["p16ab32"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ABLATE": "32"})   # exp walk/apply
ARMS["p16ab120"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_ABLATE": "120"}) # exp grp+u+walk+out
# B16 DECODE attribution on the goal4 base (the decode-goal wall). Same
# stopwatch-only contract as the prefill ablations: each arm's B16 TG delta vs
# the p16goal4 control is that family's decode-step share, and the 255 arm is
# the non-mach1 decode floor (stock ops + GDN delta rule + FA + glue + launch
# tail) that no amount of mach1 kernel work can go below.
ARMS["p16g4ab"]   = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_ABLATE": "255"})
ARMS["p16g4a7"]   = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_ABLATE": "7"})   # rt spine
ARMS["p16g4a120"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_ABLATE": "120"}) # expert mega
ARMS["p16g4a128"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_ABLATE": "128"}) # head
# TC-FWHT transform stages at prefill: lift the nt <= 16 gate to 512
ARMS["p16pp4"] = ("mach1", {**ARMS["p16pp2"][1], "GGML_MACH1_TC_NT": "512"})  # note: engages B1 nt=128 too; B1 PP floor noted in ledger
# fp16-A expert apply on top of pp4 (LUT expansion, no q8 records)
ARMS["p16pp5"] = ("mach1", {**ARMS["p16pp4"][1], "GGML_MACH1_EXP_APPLY_FP16": "1"})
# grouped-GEMM expert apply on top of pp5 (hot groups -> transient banks + cuBLAS)
ARMS["p16pp6"] = ("mach1", {**ARMS["p16pp5"][1], "GGML_MACH1_EXP_APPLY_GG": "1"})
# GG hot-threshold sweep + per-group GemmEx loop (no batched call)
# GG stage census (TIME=1, sync-inflated) and its cold decode-floor stopwatch
# (probe=1 stops the cold fused kernel after decode; garbage output)
ARMS["p16pp6t"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_TIME": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"})
ARMS["p16pp6dt"] = ("mach1", {**ARMS["p16pp6t"][1], "GGML_MACH1_ZMMA_PROBE": "1"})
# GG fork: the cold fused kernel on a side stream, overlapping the hot chain
ARMS["p16pp6f"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_FORK": "1"})
# with cold and chain concurrent, banking more of the middle rebalances the
# two streams - re-sweep the hot threshold under the fork
ARMS["p16pp6f32"] = ("mach1", {**ARMS["p16pp6f"][1], "GGML_MACH1_GG_MIN": "32"})
ARMS["p16pp6f96"] = ("mach1", {**ARMS["p16pp6f"][1], "GGML_MACH1_GG_MIN": "96"})
ARMS["p16pp6tf"] = ("mach1", {**ARMS["p16pp6t"][1], "GGML_MACH1_GG_FORK": "1"})
# distinct labels for an A/B/A/B drift-controlled wall
ARMS["p16pp6x"] = ("mach1", dict(ARMS["p16pp6"][1]))
ARMS["p16pp6fx"] = ("mach1", dict(ARMS["p16pp6f"][1]))
ARMS["p16pp6m32"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_MIN": "32"})
ARMS["p16pp6m64"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_MIN": "64"})
ARMS["p16pp6m96"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_MIN": "96"})
ARMS["p16pp6m128"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_MIN": "128"})
ARMS["p16pp6l"] = ("mach1", {**ARMS["p16pp6"][1], "GGML_MACH1_GG_BATCHED": "0"})
# n_ubatch 2048: one prefill ubatch instead of four - the per-ubatch weight
# decode tax amortizes 4x (q4 is already compute-bound and gains less)
ARMS["p16pp7"]  = ("mach1", {**ARMS["p16pp6"][1], "BENCH_UB": "2048"})
ARMS["p16pp7f"] = ("mach1", {**ARMS["p16pp6f"][1], "BENCH_UB": "2048"})
# embed-mirror control: EMBED_DEV defaults on, e0 restores host zero-copy reads
ARMS["p16pp7fe0"] = ("mach1", {**ARMS["p16pp7f"][1], "GGML_MACH1_EMBED_DEV": "0"})
# skinny f32 mm -> f16 GemmEx lane (GDN beta/alpha projections + router logits)
ARMS["p16pp7fs"] = ("mach1", {**ARMS["p16pp7f"][1], "GGML_MACH1_SKINNY_MM": "1"})
ARMS["q4km2k"]  = ("q4km",  {"BENCH_UB": "2048"})
# ub4096 rung: expert weight-decode is a fixed per-ubatch cost, so doubling
# the ubatch halves its per-token share. Prompt must be >= ub to exercise it;
# the p4 controls hold -p fixed at 4096 so ub is the only moving part.
ARMS["p16pp8f"]   = ("mach1", {**ARMS["p16pp6f"][1], "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp8ff"]  = ("mach1", {**ARMS["p16pp8f"][1], "GGML_MACH1_TOPK_FIX": "1"})
ARMS["p16pp7fp4"] = ("mach1", {**ARMS["p16pp6f"][1], "BENCH_UB": "2048", "BENCH_P": "4096"})
ARMS["q4km4k"]    = ("q4km",  {"BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["q4km2kp4"]  = ("q4km",  {"BENCH_UB": "2048", "BENCH_P": "4096"})
# Ada retune: the PP stack was tuned on H200 containers; race the cumulative
# knob chain at ub4096/p4096 on L40S to find Ada-negative knobs
ARMS["p16g3u4"]  = ("mach1", {**ARMS["p16goal3"][1], "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp1u4"] = ("mach1", {**ARMS["p16pp1"][1],  "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp2u4"] = ("mach1", {**ARMS["p16pp2"][1],  "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp4u4"] = ("mach1", {**ARMS["p16pp4"][1],  "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp5u4"] = ("mach1", {**ARMS["p16pp5"][1],  "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp8"]   = ("mach1", {**ARMS["p16pp6"][1],  "BENCH_UB": "4096", "BENCH_P": "4096"})
ARMS["p16pp6k"]  = ("mach1", {**ARMS["p16pp6"][1],  "GGML_MACH1_TOPK_FIX": "1"})
ARMS["p16pp9f"]   = ("mach1", {**ARMS["p16pp6f"][1], "BENCH_UB": "8192", "BENCH_P": "8192"})
ARMS["p16pp9fs"]  = ("mach1", {**ARMS["p16pp9f"][1], "GGML_MACH1_SKINNY_MM": "1"})
ARMS["p16pp9ff"]  = ("mach1", {**ARMS["p16pp9f"][1], "GGML_MACH1_TOPK_FIX": "1"})
# full composition: ub8192 + fork + topk fix + i8 hot lane + s8 cold slice
ARMS["p16pp10"]   = ("mach1", {**ARMS["p16pp9ff"][1], "GGML_MACH1_GG_I8": "1",
                               "GGML_MACH1_GG_COLD_S8": "1"})
# gates-shape twin of the full composition (no BENCH_* keys; ub set by the lane)
ARMS["p16pp6all"] = ("mach1", {**ARMS["p16pp6f"][1], "GGML_MACH1_TOPK_FIX": "1",
                               "GGML_MACH1_GG_I8": "1", "GGML_MACH1_GG_COLD_S8": "1"})
ARMS["p16pp8fp8"] = ("mach1", {**ARMS["p16pp6f"][1], "BENCH_UB": "4096", "BENCH_P": "8192"})
ARMS["q4km8k"]    = ("q4km",  {"BENCH_UB": "8192", "BENCH_P": "8192"})
ARMS["q4km4kp8"]  = ("q4km",  {"BENCH_UB": "4096", "BENCH_P": "8192"})
# GG Ada rungs: s8 cold slice (integer class, warm-up) and the int8 hot lane
# (i8 banks + per-row-quantized B -> 32I batched GEMMs at Ada's imma rate;
# gamma folds into the i8 lattice per row tile - KLD class, not bitwise)
ARMS["p16pp8fc8"] = ("mach1", {**ARMS["p16pp8f"][1], "GGML_MACH1_GG_COLD_S8": "1"})
ARMS["p16pp8fi8"] = ("mach1", {**ARMS["p16pp8f"][1], "GGML_MACH1_GG_I8": "1"})
ARMS["p16pp8fb8"] = ("mach1", {**ARMS["p16pp8fi8"][1], "GGML_MACH1_GG_COLD_S8": "1"})
# gate lanes on the pp6f stack (gates/ntcheck size their own batches)
ARMS["p16pp6i8"]  = ("mach1", {**ARMS["p16pp6f"][1], "GGML_MACH1_GG_I8": "1"})
ARMS["p16pp6c8"]  = ("mach1", {**ARMS["p16pp6f"][1], "GGML_MACH1_GG_COLD_S8": "1"})
# engagement/probe receipts (DEBUG=1 prints the ENGAGED + i8 probe lines)
ARMS["p16pp6i8d"] = ("mach1", {**ARMS["p16pp6i8"][1], "GGML_MACH1_GG_COLD_S8": "1",
                               "GGML_MACH1_DEBUG": "1"})
# bank-cap sweep under the i8 lane: at ub4096 the 200MB cap escalates the hot
# threshold; i8 banks are 1 B/weight so a bigger cap banks (nearly) all groups
ARMS["p16pp8fb8m5"] = ("mach1", {**ARMS["p16pp8fb8"][1], "GGML_MACH1_GG_MB": "512"})
ARMS["p16pp8fb8w"]  = ("mach1", {**ARMS["p16pp8fb8"][1], "GGML_MACH1_GG_MB": "512",
                                 "GGML_MACH1_GG_MIN": "32"})
# family ablations on the pp5 stack (garbage output, stopwatch only)
for _b in ("2", "7", "32", "120", "255"):
    ARMS["p16p5a" + _b] = ("mach1", {**ARMS["p16pp5"][1], "GGML_MACH1_ABLATE": _b})
ARMS["p16qace16"] = ("mach1", {**ARMS["p16qace"][1], "GGML_MACH1_MEGA_TW": "16"})
# distinct labels for the interleaved graphs-on wall
ARMS["p16qa0"] = ("mach1", dict(ARMS["p16qace"][1]))
ARMS["p16qa1"] = ("mach1", dict(ARMS["p16qace"][1]))

# Orthogonal two-factor attribution matrix: first bit = cp.async, second bit =
# all validated sibling batches; native IMMA8 is held on in every cell. Each
# cell gets its own subprocess so static env gates cannot leak across cells.
ARMS["p16f00"] = ("mach1", dict(ARMS["p16rti8"][1]))
ARMS["p16f10"] = ("mach1", dict(ARMS["p16rti8cp"][1]))
ARMS["p16f01"] = ("mach1", dict(ARMS["p16i8sib"][1]))
ARMS["p16f11"] = ("mach1", dict(ARMS["p16combo"][1]))
# Distinct labels preserve both candidate logs and receipts in a T-Q-T wall;
# repeating one arm name would overwrite its first result file.
ARMS["p16t0"] = ("mach1", dict(ARMS["p16combo"][1]))
ARMS["p16t1"] = ("mach1", dict(ARMS["p16combo"][1]))

# pplow campaign: the PP lanes at SHORT-prompt ubatches.
# p16goal4 sets NO PP env at all, so the sweep-s3 S_PP column measured the PP
# stack switched off, not gated out. p16pp6all is the same goal4 decode stack
# with the certified PP lanes on and no BENCH_* override, i.e. the lane's own
# ubatch - the pau/g4u pair below separates "lanes off" from "lanes gated out".
# ub ladder for the per-ubatch cost model (bench lane, -p 2048): the wall at
# each ub divides into T/ubatch, which fits fixed + per-token per stack.
for _ub in ("64", "128", "256", "512", "2048"):
    ARMS["q4u" + _ub]    = ("q4km",  {"BENCH_UB": _ub})
    ARMS["p16g4u" + _ub] = ("mach1", {**ARMS["p16goal4"][1],  "BENCH_UB": _ub})
    ARMS["p16pau" + _ub] = ("mach1", {**ARMS["p16pp6all"][1], "BENCH_UB": _ub})
# PPLOW: the same PP stack with the nt >= 256 admission floor lowered. GG keeps
# its own floor (it blocks the host on a D2H group-count fetch per call), so
# the plain arm leaves GG at 256 and the "g" arm drops it too.
ARMS["p16pl128"]  = ("mach1", {**ARMS["p16pp6all"][1], "GGML_MACH1_PPLOW": "1",
                               "GGML_MACH1_PPLOW_MIN": "128"})
ARMS["p16plg128"] = ("mach1", {**ARMS["p16pl128"][1], "GGML_MACH1_PPLOW_GG_MIN": "128"})
ARMS["p16pl64"]   = ("mach1", {**ARMS["p16pp6all"][1], "GGML_MACH1_PPLOW": "1",
                               "GGML_MACH1_PPLOW_MIN": "64"})
ARMS["p16plg64"]  = ("mach1", {**ARMS["p16pl64"][1], "GGML_MACH1_PPLOW_GG_MIN": "64"})
# expert-lane race below the GG floor: with GG gated out the apply lands on
# fp16 (pl) or, with FP16 off, on the s8 MMA form (pls). Identical to pl at
# nt >= 256, where the GG branch owns the op and zf16 is moot.
ARMS["p16pls128"] = ("mach1", {**ARMS["p16pl128"][1], "GGML_MACH1_EXP_APPLY_FP16": "0"})
ARMS["p16pls64"]  = ("mach1", {**ARMS["p16pl64"][1],  "GGML_MACH1_EXP_APPLY_FP16": "0"})
for _a, _ub in (("p16pls128", "128"), ("p16pls64", "64")):
    ARMS[_a + "u" + _ub] = ("mach1", {**ARMS[_a][1], "BENCH_UB": _ub})
for _a, _ub in (("p16pl128", "128"), ("p16plg128", "128"),
                ("p16pl64", "64"), ("p16plg64", "64")):
    ARMS[_a + "u" + _ub] = ("mach1", {**ARMS[_a][1], "BENCH_UB": _ub})
# composed short-prompt candidate for the batched lane: PP lanes on, admission
# floor at 128, and the ubatch raised to the lane's n_batch. npp=128 x npl
# gives min(128*npl, 2048) tokens per ubatch, so a SHORT prompt still amortizes
# mach1's fixed per-ubatch weight decode over a WIDE ubatch.
ARMS["p16pl2k"] = ("mach1", {**ARMS["p16pl128"][1], "BENCH_UB": "2048"})
ARMS["p16pa2k"] = ("mach1", {**ARMS["p16pp6all"][1], "BENCH_UB": "2048"})
# certification flank clones for the short-prompt wall
ARMS["p16paa"] = ("mach1", dict(ARMS["p16pp6all"][1]))
ARMS["p16pab"] = ("mach1", dict(ARMS["p16pp6all"][1]))
ARMS["p16pl2ka"] = ("mach1", dict(ARMS["p16pl2k"][1]))
ARMS["p16pl2kb"] = ("mach1", dict(ARMS["p16pl2k"][1]))
# the fixed per-ubatch cost the cost model isolates is the spine's trellis
# decode into a transient fp16 bank, paid AGAIN every ubatch because
# GGML_MACH1_BANK defaults off. Banking makes it a once-per-process cost, which
# is exactly the term a short ubatch cannot amortize. NOTE this also puts the
# nt == 1 rt op on the bank apply instead of the walk, so it is a decode-class
# change and cannot ride the PPLOW default-off argument.
ARMS["p16bk"]  = ("mach1", {**ARMS["p16pp6all"][1], "GGML_MACH1_BANK": "1"})
ARMS["p16bkl"] = ("mach1", {**ARMS["p16pl128"][1],  "GGML_MACH1_BANK": "1"})
for _ub in ("128", "512", "2048"):
    ARMS["p16bku" + _ub]  = ("mach1", {**ARMS["p16bk"][1],  "BENCH_UB": _ub})
    ARMS["p16bklu" + _ub] = ("mach1", {**ARMS["p16bkl"][1], "BENCH_UB": _ub})
# skinny stock-op GEMM lane at short ubatches (wall-neutral at long prompts)
ARMS["p16sku512"]  = ("mach1", {**ARMS["p16pp6all"][1], "GGML_MACH1_SKINNY_MM": "1",
                                "BENCH_UB": "512"})
ARMS["p16sklu128"] = ("mach1", {**ARMS["p16pl128"][1],  "GGML_MACH1_SKINNY_MM": "1",
                                "BENCH_UB": "128"})
# bank fault localization: -v so the swallowed CUDA_CHECK message prints, and
# a graphs-off clone (the memcheck lane is clean with graphs disabled)
ARMS["p16bkvu128"]  = ("mach1", {**ARMS["p16bk"][1], "BENCH_UB": "128", "BENCH_V": "1"})
ARMS["p16bkngu128"] = ("mach1", {**ARMS["p16bk"][1], "BENCH_UB": "128",
                                 "GGML_CUDA_DISABLE_GRAPHS": "1"})
# decode-class A/B for the bank: BANK_NT=1 drops the token floor so the nt == 1
# rt op reads the fp16 bank instead of walking the compressed stream. Expected
# to LOSE (a decoded spine is ~3 GB/step against ~638 MB compressed).
ARMS["p16bkd"] = ("mach1", {**ARMS["p16bk"][1], "GGML_MACH1_BANK_NT": "1"})
# prefill-shaped stage profile: which stage the bank actually removes, and what
# the apply costs when it reads a 2.62 GiB persistent image instead of the one
# pool block the transient decode keeps rewriting
for _a, _s in (("p16pau512", "p16past"), ("p16bku512", "p16bkst")):
    ARMS[_s] = ("mach1", {**ARMS[_a][1], "BENCH_P": "2048", "BENCH_N": "8"})
# bench-lane flanks for the BANK_NT decode reproduction (the first receipt is
# one same-container observation on the PP stack)
ARMS["p16bkda"] = ("mach1", dict(ARMS["p16bkd"][1]))
ARMS["p16bkdb"] = ("mach1", dict(ARMS["p16bkd"][1]))
# the same lever on the CERTIFIED DECODE stack (ntlow + NTLOW_MIN=8 + nt32),
# which is the arm the cross-batch position vs q4 is quoted from. p16bkd rides
# the PP stack, whose B1 decode is ~15 percent slower than this one, so a win
# there does not by itself move the position.
ARMS["p16nlbk0"] = ("mach1", {**ARMS["p16nl832"][1], "GGML_MACH1_BANK": "1"})
ARMS["p16nlbk"]  = ("mach1", {**ARMS["p16nlbk0"][1], "GGML_MACH1_BANK_NT": "1"})
ARMS["p16nlbka"] = ("mach1", dict(ARMS["p16nlbk"][1]))
ARMS["p16nlbkb"] = ("mach1", dict(ARMS["p16nlbk"][1]))
# bank-at-the-prefill-floor flanks: same 2.62 GiB resident image, decode
# textually unchanged, so the pair isolates the BANK_NT lever from the bank
ARMS["p16nlbk0a"] = ("mach1", dict(ARMS["p16nlbk0"][1]))
ARMS["p16nlbk0b"] = ("mach1", dict(ARMS["p16nlbk0"][1]))
# admission-floor sweep: BANK_NT is a MIN, so 2/4/8 keep nt below it on the
# walk. Bracketing where the bytes-vs-ALU trade inverts as the batch grows.
for _bn in ("2", "4", "8"):
    ARMS["p16nlbk" + _bn] = ("mach1", {**ARMS["p16nlbk0"][1], "GGML_MACH1_BANK_NT": _bn})
# BANK_NT_MAX caps the lowered floor at a WINDOW [1, cap] so the widths the
# curve says lose keep their certified path (cap 0 = uncapped, the old form)
ARMS["p16nlbkc1"] = ("mach1", {**ARMS["p16nlbk"][1], "GGML_MACH1_BANK_NT_MAX": "1"})
ARMS["p16nlbkc1a"] = ("mach1", dict(ARMS["p16nlbkc1"][1]))
ARMS["p16nlbkc1b"] = ("mach1", dict(ARMS["p16nlbkc1"][1]))
ARMS["p16nlbkc4"] = ("mach1", {**ARMS["p16nlbk"][1], "GGML_MACH1_BANK_NT_MAX": "4"})

# thread-count A/B inside ONE container (BENCH_T pins -t/-tb per arm). The two
# batched lanes differ in core count as well as CPU request - default 17,
# cpu=16 32 - so the thread count has to be priced on its own before either
# lane can be called authoritative.
ARMS["p16t17a"] = ("mach1", {**ARMS["p16goal4"][1], "BENCH_T": "17"})
ARMS["p16t17b"] = ("mach1", dict(ARMS["p16t17a"][1]))
ARMS["p16t32a"] = ("mach1", {**ARMS["p16goal4"][1], "BENCH_T": "32"})
ARMS["p16t32b"] = ("mach1", dict(ARMS["p16t32a"][1]))
ARMS["q4kmt17"] = ("q4km", {"BENCH_T": "17"})
ARMS["q4kmt32"] = ("q4km", {"BENCH_T": "32"})

# node-count census: graphstat counters (TIME>=4) key by cgraph->n_nodes, so a
# decode replay row names how many nodes each stack launches per step. The
# thread A/B showed mach1 loses 3.6-6.8 pct to extra host threads while q4
# moves <0.25 pct, which says host-side launch work is a real mach1-only
# component of the step - this counts it.
ARMS["p16nl8g4"] = ("mach1", {**ARMS["p16nl8a"][1], "GGML_MACH1_TIME": "4"})
ARMS["q4kmg4"]   = ("q4km",  {"GGML_MACH1_TIME": "4"})

# spine OPT rung: two exact restructurings of the native int8 spine, each its
# own bit of GGML_MACH1_RT_SPINE_OPT so the wall can price them separately.
#   1 WALK8 - the codec window is word-aligned, so the funnel shift and one of
#             the two shared code loads per state are dead work
#   2 QPAD  - the staged q8 rows are 128 B apart, which puts all eight
#             A-fragment rows of a warp in the same four shared banks
# Values are bit-identical at every setting; 0 is the shipped source.
ARMS["p16so1"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_RT_SPINE_OPT": "1"})
ARMS["p16so2"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_RT_SPINE_OPT": "2"})
ARMS["p16so3"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_RT_SPINE_OPT": "3"})
ARMS["p16so3a"] = ("mach1", dict(ARMS["p16so3"][1]))
ARMS["p16so3b"] = ("mach1", dict(ARMS["p16so3"][1]))
# chunked-width token gate: NTLOW+NT32 tile B24 as 16+8, so the second chunk
# issues at a non-zero token offset - the only path where the OPT dispatch's
# q8-slot pointer arithmetic differs from a pure pass-through
ARMS["p16nl8so3"] = ("mach1", {**ARMS["p16nl832"][1], "GGML_MACH1_RT_SPINE_OPT": "3"})
# parallelism negative control: split-K off drops the m2048 spine grids from
# 256 CTAs back to 64, at identical total work. If the walk were bound by its
# own arithmetic this would be free; the in-tree stream probe says 64-CTA grids
# cap at ~320 GB/s against ~978 at 256.
ARMS["p16sk0"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_RT_IMMA8_SPLITK": "0",
                            "GGML_MACH1_RT_IMMA8_SPLITK_N512": "0"})
# launch-cost probe: the rt u stage is re-issued once per spine op (idempotent,
# values unchanged), so the wall delta divided by the 80 decode u launches per
# step is the marginal cost of one small mach1 launch.
ARMS["p16lp1"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_LAUNCH_PROBE": "1"})
# TRANSFORM-LAUNCH TAIL rung. The spine OPT rung priced one small decode
# launch at 3.69 us and killed per-weight arithmetic as a mechanism, so the
# lever is launch COUNT. RT_TAIL folds the elementwise consumers of the two
# standalone rt projections per layer into their out transform's store: the
# attention/GDN residual ADD, and the shared-expert sigmoid-gate MUL plus the
# moe and ffn-residual ADDs. 120 launches per step (40 + 80) leave the chain.
# Same fp32 ops in the same order (__fmul_rn/__fadd_rn pin the roundings).
ARMS["p16rtt"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_RT_TAIL": "1"})
ARMS["p16rtta"] = ("mach1", dict(ARMS["p16rtt"][1]))
ARMS["p16rttb"] = ("mach1", dict(ARMS["p16rtt"][1]))
# four-flank forms: the rung's effect is under 1 pct and the c16 containers
# drift up to 2 pct across a five-arm pass, so the draw needs four of each
# interleaved rather than two
ARMS["p16rttc"] = ("mach1", dict(ARMS["p16rtt"][1]))
ARMS["p16rttd"] = ("mach1", dict(ARMS["p16rtt"][1]))
ARMS["p16j4c"]  = ("mach1", dict(ARMS["p16goal4"][1]))
ARMS["p16j4d"]  = ("mach1", dict(ARMS["p16goal4"][1]))
ARMS["p16rttt16"] = ("mach1", {**ARMS["p16rtt"][1], "GGML_MACH1_TIME": "1",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
# TRANSFORM WIDTH. If the 3.69 us of a re-issued u stage is that kernel's own
# execution rather than launch overhead, the lever on the transform tail is
# threads per transform, not launches. GGML_MACH1_TC_WG=1024 widens the two
# transform families whose factorization keeps every warp group on an mi row
# at 1024 threads (A = 64: the standalone n = 4096 u stage and the qkvz/qkv
# sibling out batch at m = 4096/8192). Bit-identical at either width.
ARMS["p16tcw"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TC_WG": "1024"})
ARMS["p16tcwa"] = ("mach1", dict(ARMS["p16tcw"][1]))
ARMS["p16tcwb"] = ("mach1", dict(ARMS["p16tcw"][1]))
ARMS["p16tcwc"] = ("mach1", dict(ARMS["p16tcw"][1]))
ARMS["p16tcwd"] = ("mach1", dict(ARMS["p16tcw"][1]))
# per-OP census pair (TIME=2, graphs off, one clean B16 width per process):
# counts the ADD/MUL node executions the fold removes and the per-step
# rttail:* fold counts, so "the marker fired once" cannot stand in for "every
# site engaged"
ARMS["p16g4c2b16"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "2",
                                 "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16rttc2b16"] = ("mach1", {**ARMS["p16rtt"][1], "GGML_MACH1_TIME": "2",
                                 "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
# per-kernel stopwatch (TIME=1, graphs off) at B16, control and candidate
ARMS["p16g4t16"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_TIME": "1",
                              "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16so3t16"] = ("mach1", {**ARMS["p16so3"][1], "GGML_MACH1_TIME": "1",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})

# CONCURRENCY rung. The B16 step is two different bottlenecks in series: the
# routed expert mega (bandwidth) and the shexp/spine chain (execution on small
# grids). They are independent until the ffn add, so the question is whether
# they can run at the same time. The mega is a COOPERATIVE launch sized to
# max occupancy - 1 block/SM at 1024 threads and 64 registers - so while it
# runs it owns every SM and nothing else can co-reside.
#   FORK_P16 puts the p16 shared-expert gate/up region (u_tcb, two TT walks,
#     out+glu) on the fork's second stream behind the event the expert matcher
#     records before the mega, joining before the down rt node.
#   MEGA_RSV holds blocks back from the mega grid at decode, which is the only
#     way a second stream gets an SM. Bitwise: every output row tile belongs to
#     exactly one task and the task loop strides by the grid.
ARMS["p16fk"]     = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_FORK_P16": "1"})
ARMS["p16fka"]    = ("mach1", dict(ARMS["p16fk"][1]))
ARMS["p16fkb"]    = ("mach1", dict(ARMS["p16fk"][1]))
ARMS["p16rsv8"]   = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_MEGA_RSV": "8"})
ARMS["p16rsv16"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_MEGA_RSV": "16"})
ARMS["p16rsv32"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_MEGA_RSV": "32"})
ARMS["p16rsv64"]  = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_MEGA_RSV": "64"})
ARMS["p16fkr8"]   = ("mach1", {**ARMS["p16fk"][1], "GGML_MACH1_MEGA_RSV": "8"})
ARMS["p16fkr16"]  = ("mach1", {**ARMS["p16fk"][1], "GGML_MACH1_MEGA_RSV": "16"})
ARMS["p16fkr16a"] = ("mach1", dict(ARMS["p16fkr16"][1]))
ARMS["p16fkr16b"] = ("mach1", dict(ARMS["p16fkr16"][1]))
ARMS["p16fkr16c"] = ("mach1", dict(ARMS["p16fkr16"][1]))
ARMS["p16fkr16d"] = ("mach1", dict(ARMS["p16fkr16"][1]))
ARMS["p16fkr32"]  = ("mach1", {**ARMS["p16fk"][1], "GGML_MACH1_MEGA_RSV": "32"})
ARMS["p16fkr32a"] = ("mach1", dict(ARMS["p16fkr32"][1]))
ARMS["p16fkr32b"] = ("mach1", dict(ARMS["p16fkr32"][1]))
# overlap receipt: %globaltimer stamps around the mega (main) and the forked
# region (s2), dumped from the next uncaptured matcher call. Graphs off so the
# host reaches that call every step; one batch width per process.
ARMS["p16fkst"]   = ("mach1", {**ARMS["p16fk"][1], "GGML_MACH1_FORK_STAMP": "1",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16fkrst"]  = ("mach1", {**ARMS["p16fkr16"][1], "GGML_MACH1_FORK_STAMP": "1",
                               "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16fkrst32"] = ("mach1", {**ARMS["p16fkr32"][1], "GGML_MACH1_FORK_STAMP": "1",
                                "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})
ARMS["p16fkrst64"] = ("mach1", {**ARMS["p16goal4"][1], "GGML_MACH1_FORK_P16": "1",
                                "GGML_MACH1_MEGA_RSV": "64", "GGML_MACH1_FORK_STAMP": "1",
                                "GGML_CUDA_DISABLE_GRAPHS": "1", "BENCH_NPL": "16"})

ARM_ENGAGEMENT_MARKERS = {
    "p16rti8cp": [
        "mach1: zero-VRAM native rt_imma8 cp.async double-buffer ENGAGED (s_red alias)",
    ],
    "p16gdnb": [
        "mach1: p16 qkvz sibling transforms + rt_mma16 ENGAGED",
    ],
    "p16sib": [
        "mach1: p16 qkvz sibling transforms + rt_mma16 ENGAGED",
        "mach1: p16 attention qkv sibling batch + rt_mma16 ENGAGED",
        "mach1: p16 shared gate/up sibling batch ENGAGED",
    ],
    "p16i8sib": [
        "mach1: p16 qkvz sibling transforms + rt_imma8 ENGAGED",
        "mach1: p16 attention qkv sibling batch + rt_imma8 ENGAGED",
        "mach1: p16 shared gate/up sibling batch ENGAGED",
    ],
    "p16combo": [
        "mach1: zero-VRAM native rt_imma8 cp.async double-buffer ENGAGED (s_red alias)",
        "mach1: p16 qkvz sibling transforms + rt_imma8 ENGAGED",
        "mach1: p16 attention qkv sibling batch + rt_imma8 ENGAGED",
        "mach1: p16 shared gate/up sibling batch ENGAGED",
    ],
    "p16f00": [
        "mach1: zero-VRAM compressed rt_imma8 ENGAGED",
    ],
    "p16f10": [
        "mach1: zero-VRAM native rt_imma8 cp.async double-buffer ENGAGED (s_red alias)",
    ],
    "p16f01": [
        "mach1: p16 qkvz sibling transforms + rt_imma8 ENGAGED",
        "mach1: p16 attention qkv sibling batch + rt_imma8 ENGAGED",
        "mach1: p16 shared gate/up sibling batch ENGAGED",
    ],
    "p16f11": [
        "mach1: zero-VRAM native rt_imma8 cp.async double-buffer ENGAGED (s_red alias)",
        "mach1: p16 qkvz sibling transforms + rt_imma8 ENGAGED",
        "mach1: p16 attention qkv sibling batch + rt_imma8 ENGAGED",
        "mach1: p16 shared gate/up sibling batch ENGAGED",
    ],
}
for _arm in ("p16fk", "p16fka", "p16fkb", "p16fkr8", "p16fkr16", "p16fkr16a",
             "p16fkr16b", "p16fkr16c", "p16fkr16d", "p16fkr32", "p16fkr32a", "p16fkr32b"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        "mach1: p16 shared gate/up FORK ENGAGED (s2 under the expert mega)",
    ]
for _arm in ("p16rsv8", "p16rsv16", "p16rsv32", "p16rsv64", "p16fkr8", "p16fkr16",
             "p16fkr16a", "p16fkr16b", "p16fkr16c", "p16fkr16d", "p16fkr32",
             "p16fkr32a", "p16fkr32b"):
    ARM_ENGAGEMENT_MARKERS.setdefault(_arm, []).append(
        "mach1: expert mega grid RESERVED")

ARM_ENGAGEMENT_MARKERS["p16t0"] = list(ARM_ENGAGEMENT_MARKERS["p16combo"])
ARM_ENGAGEMENT_MARKERS["p16t1"] = list(ARM_ENGAGEMENT_MARKERS["p16combo"])
for _arm in ("p16kc0", "p16kc1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = list(ARM_ENGAGEMENT_MARKERS["p16combo"])
for _arm in ("p16ks4", "p16ks0", "p16ks1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *ARM_ENGAGEMENT_MARKERS["p16combo"],
        "mach1: native rt_imma8 split-K4 m2048 n4096 nt16 ENGAGED "
        "(384 KiB transient, 0 persistent)",
    ]
for _arm in ("p16qfs", "p16qf0", "p16qf1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *ARM_ENGAGEMENT_MARKERS["p16combo"],
        "mach1: QONCE producer FWHT scale fold ENGAGED (bitwise records, zero scratch delta)",
    ]
ARM_ENGAGEMENT_MARKERS["p16std"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ks4"],
    "mach1: standard-output row-scale epilogues ENGAGED (TIMING_ONLY WRONG_BASIS NOT_QUALITY)",
    "mach1: expert mega standard-output ENGAGED (TIMING_ONLY WRONG_BASIS NOT_QUALITY)",
]
# the fused out+glu branch replaces the shexp sibling-batch launch, so its
# marker replaces that marker rather than joining it
for _arm in ("p16qace", "p16qa0", "p16qa1", "p16sgf", "p16sg0", "p16sg1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *[m for m in ARM_ENGAGEMENT_MARKERS["p16ks4"]
          if m != "mach1: p16 shared gate/up sibling batch ENGAGED"],
        "mach1: QONCE mega scheduling ENGAGED (tw=4 dnct=32 slotrel=1)",
        "mach1: p16 shared gate/up fused out+glu ENGAGED",
    ]
for _arm in ("p16goal", "p16go0", "p16go1", "p16goal2", "p16h0", "p16h1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *[m for m in ARM_ENGAGEMENT_MARKERS["p16ks4"]
          if m != "mach1: p16 shared gate/up sibling batch ENGAGED"],
        "mach1: QONCE mega scheduling ENGAGED (tw=8 dnct=32 slotrel=1)",
        "mach1: p16 shared gate/up fused out+glu ENGAGED",
    ]
for _arm in ("p16goal3", "p16j0", "p16j1"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *ARM_ENGAGEMENT_MARKERS["p16goal2"],
        "mach1: p16 GDN prep chain fused ENGAGED (11-node window, 6 launches -> 1)",
    ]
# ntlo sweep receipts assume the npl 1,2,4,8,16 lane: QONCE first engages
# sub-16 at npl=2, the padded 16-token forms at npl=8
# NTLOW_MIN default 8 (chunkab-wall-s4): the walk owns nt 2-4, so spine/
# sibling/head first engage at nt=8; the mega's ntlo clause still fires nt=2
for _arm in ("p16ntlo", "p16loa", "p16lob", "p16log8"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        "mach1: ntlo QONCE mega nt=2 P=16 ENGAGED",
        "mach1: ntlo rt spine nt=8 ENGAGED",
        "mach1: ntlo sibling pair nt=8 ENGAGED",
        "mach1: ntlo sibling trio nt=8 ENGAGED",
        "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
    ]
# NTLO_MIN=2: spine/pair/trio first engage at npl=2; head keeps its >4 bound
ARM_ENGAGEMENT_MARKERS["p16lo2"] = [
    "mach1: ntlo QONCE mega nt=2 P=16 ENGAGED",
    "mach1: ntlo rt spine nt=2 ENGAGED",
    "mach1: ntlo sibling pair nt=2 ENGAGED",
    "mach1: ntlo sibling trio nt=2 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
# pair-order first engages at the 16-token warmup decode (P=128)
ARM_ENGAGEMENT_MARKERS["p16mo"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ntlo"],
    "mach1: QONCE mega pair-order ENGAGED (P=128)",
]
for _arm in ("p16g3f", "p16zf4", "p16go2"):
    ARM_ENGAGEMENT_MARKERS[_arm] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
# slot takes the expert region below nt=9, so the QONCE marker is replaced
ARM_ENGAGEMENT_MARKERS["p16sl8"] = [
    "mach1: exp_slot nt=2 P=16 ENGAGED",
    "mach1: ntlo rt spine nt=8 ENGAGED",
    "mach1: ntlo sibling pair nt=8 ENGAGED",
    "mach1: ntlo sibling trio nt=8 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
# pflags first engages at the 16-token warmup decode
ARM_ENGAGEMENT_MARKERS["p16pf"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ntlo"],
    "mach1: QONCE mega pflags ENGAGED (P=128)",
]
ARM_ENGAGEMENT_MARKERS["p16sx"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ntlo"],
    "mach1: SKIP_EXTRA maintenance pairs DROPPED",
]
# head admission floor arms: the wall lane is ONE process over npl 1..16 and
# the head print is a once-flag, so the lowered floor MOVES the single print
# to the first admitted width (the warmup decode carries no logits, so the
# head first runs at the npl=1 leg). nt=8 never prints in these arms.
ARM_ENGAGEMENT_MARKERS["p16hd2"] = [
    *[m for m in ARM_ENGAGEMENT_MARKERS["p16ntlo"]
      if m != "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED"],
    "mach1: zero-VRAM compressed head_mma16 nt=2 ENGAGED",
]
ARM_ENGAGEMENT_MARKERS["p16hd1"] = [
    *[m for m in ARM_ENGAGEMENT_MARKERS["p16ntlo"]
      if m != "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED"],
    "mach1: zero-VRAM compressed head_mma16 nt=1 ENGAGED",
]
# qsplit owns the whole QONCE range under MEGA_SPLIT, so the coop-mega ntlo
# QONCE print never fires in this arm; the mode substring pins which table
# form actually ran (half smem vs nostage L2/L1) and the blocks/SM intent
ARM_ENGAGEMENT_MARKERS["p16ms"] = [
    "mach1: QONCE mega split ENGAGED",
    "wg512 half 2/SM pflags",
    "mach1: ntlo rt spine nt=8 ENGAGED",
    "mach1: ntlo sibling pair nt=8 ENGAGED",
    "mach1: ntlo sibling trio nt=8 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
ARM_ENGAGEMENT_MARKERS["p16msn"] = [
    "mach1: QONCE mega split ENGAGED",
    "wg512 nostage 2/SM pflags",
    "mach1: ntlo rt spine nt=8 ENGAGED",
    "mach1: ntlo sibling pair nt=8 ENGAGED",
    "mach1: ntlo sibling trio nt=8 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
ARM_ENGAGEMENT_MARKERS["p16msn3"] = [
    "mach1: QONCE mega split ENGAGED",
    "wg512 nostage 3/SM pflags",
    "mach1: ntlo rt spine nt=8 ENGAGED",
    "mach1: ntlo sibling pair nt=8 ENGAGED",
    "mach1: ntlo sibling trio nt=8 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
# the coop nostage print fires once at the warmup decode (P=128); the ntlo
# sub-16 QONCE print still fires at npl=2 because the form stays mega_qonce
ARM_ENGAGEMENT_MARKERS["p16nfd"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
ARM_ENGAGEMENT_MARKERS["p16hb"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
ARM_ENGAGEMENT_MARKERS["p16wm2"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
# nlm8: chunks off below nt=8 -> the merged head (nt_ok-gated) runs the warp
# form at nt 2-4, so its once-print first fires at nt=8; ntlo spine/sibling
# prints still fire at nt=2
# under min=8 the sibling/spine chunked forms sit out nt 2-4 too (both
# OR-clauses floor above them), so only the mega ntlo print fires at nt=2;
# spine/sibling/head first engage at nt=8
ARM_ENGAGEMENT_MARKERS["p16fn4"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
ARM_ENGAGEMENT_MARKERS["p16all"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
ARM_ENGAGEMENT_MARKERS["p16nogf"] = list(ARM_ENGAGEMENT_MARKERS["p16ntlo"])
ARM_ENGAGEMENT_MARKERS["p16gf"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ntlo"],
    "mach1: p16 shared gate/up FORK ENGAGED (s2 under the expert mega)",
]
ARM_ENGAGEMENT_MARKERS["p16nlm8"] = [
    "mach1: ntlo QONCE mega nt=2 P=16 ENGAGED",
    "mach1: ntlo rt spine nt=8 ENGAGED",
    "mach1: ntlo sibling pair nt=8 ENGAGED",
    "mach1: ntlo sibling trio nt=8 ENGAGED",
    "mach1: zero-VRAM compressed head_mma16 nt=8 ENGAGED",
]
ARM_ENGAGEMENT_MARKERS["p16qns"] = [
    *ARM_ENGAGEMENT_MARKERS["p16ntlo"],
    "mach1: QONCE mega nostage coop ENGAGED",
]
# both tail patterns must appear: a matcher that silently stops taking one of
# them is otherwise indistinguishable from a slower control
for _arm in ("p16rtt", "p16rtta", "p16rttb", "p16rttc", "p16rttd"):
    ARM_ENGAGEMENT_MARKERS[_arm] = [
        *ARM_ENGAGEMENT_MARKERS["p16goal3"],
        "mach1: rt tail fold ENGAGED (residual add)",
        "mach1: rt tail fold ENGAGED (shexp down gate+add)",
    ]

# Revision-suffixed source tarball: the volume served STALE bytes for the
# fixed /src/tree.tar.gz path twice (ntlo-sweep-s1, ms-sweep-s1 - engagement
# markers caught both), so uploads now go to a unique per-rev path via
# benches/modal/upload_tree.sh, which also rewrites SRC_REV here. A stale
# cache can only fail loudly (missing file), never silently build old code.
SRC_REV = "8d897c903"
SRC_TAR = f"/vol/src/tree-{SRC_REV}.tar.gz"
BUILD_TARGETS = ["llama-bench", "llama-cli"]


def run_vram(cmd: str, env: dict, timeout=None):
    """Run cmd while polling nvidia-smi; returns (CompletedProcess, peak MiB).

    The container owns the GPU, so device memory.used is this process plus a
    small fixed context. A persistent weight bank is a whole-process VRAM
    cost that no llama.cpp counter reports, so it has to be measured here.
    """
    peak = [0]
    stop = threading.Event()
    q = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"

    def sample():
        while not stop.is_set():
            r = subprocess.run(q, shell=True, capture_output=True, text=True)
            try:
                peak[0] = max(peak[0], int(r.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            stop.wait(0.25)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    finally:
        stop.set()
        th.join(timeout=10)
    return j, peak[0]


def sh(cmd: str, **kw) -> str:
    print(f"+ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise RuntimeError(f"command failed: {cmd}")
    return r.stdout


# cpu=16: the bench containers previously ran at Modal's default CPU request.
# The mach1 graph does far more host-side work per decode step than q4km's
# (more nodes per replay, cooperative launches), so CPU contention shows up
# as the "bimodal glue" - identical configs reading 0.705 vs 0.837 of the
# q4km bar across containers while q4km and prefill never move.
@APP.function(image=IMAGE, gpu="L40S", cpu=16, volumes={"/vol": VOL}, timeout=4*3600)
def bench_l40s(arms: str, rounds: int, tag: str, smoke: bool):
    """Ada (sm_89) lane of the optima campaign."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return bench.local(arms, rounds, tag, smoke)


@APP.function(image=IMAGE, gpu="H200", cpu=16, volumes={"/vol": VOL}, timeout=4*3600)
def bench(arms: str, rounds: int, tag: str, smoke: bool):
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    cfg = ("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF ")
    launchers = ("-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
                 "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    try:
        sh(cfg + launchers)
        sh(f"cmake --build /work/build -j --target {' '.join(BUILD_TARGETS)}")
    except RuntimeError:
        print("ccache build failed, retrying without launchers", flush=True)
        sh("rm -rf /work/build")
        sh(cfg)
        sh(f"cmake --build /work/build -j --target {' '.join(BUILD_TARGETS)}")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)

    arm_list = [a.strip() for a in arms.split(",")]
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    results: dict[str, list[float]] = {a: [] for a in arm_list}
    pp: dict[str, list[float]] = {a: [] for a in arm_list}
    vram: dict[str, list[int]] = {a: [] for a in arm_list}

    failed: set[str] = set()
    for r in range(rounds):
        for arm in arm_list:
            if arm in failed:
                continue
            model_key, extra = ARMS[arm]
            extra = dict(extra)
            ub = extra.pop("BENCH_UB", None)   # per-arm n_ubatch override (parity with the batched lane)
            plen = extra.pop("BENCH_P", None)  # per-arm prompt-length override (must be >= ub to exercise it)
            # llama-bench nulls the ggml log callback unless -v, which swallows
            # the CUDA_CHECK message an abort prints before its backtrace
            verbose = extra.pop("BENCH_V", None)
            env = dict(os.environ, **extra)
            out = f"{outdir}/r{r}_{arm}.json"
            cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
                   f"-fa 1 -p {plen or 2048} -n 128 -o json")
            if ub:
                cmd += f" -b {ub} -ub {ub}"
            if verbose:
                cmd += " -v"
            print(f"--- round {r} arm {arm} ---", flush=True)
            j, peak = run_vram(cmd, env)
            if j.returncode != 0:
                print(j.stderr[-4000:])
                print(f"ARM FAILED: {arm} (continuing without it)", flush=True)
                with open(f"{outdir}/fail_{arm}.txt", "w") as f:
                    f.write(j.stdout + j.stderr)
                VOL.commit()
                failed.add(arm)
                continue
            with open(out, "w") as f:
                f.write(j.stdout)
            VOL.commit()
            for rec in json.loads(j.stdout):
                tps = rec["avg_ts"]
                if rec["n_gen"] > 0:
                    results[arm].append(tps)
                else:
                    pp[arm].append(tps)
            vram[arm].append(peak)
            print(f"    tg {results[arm][-1]:.2f} pp {pp[arm][-1]:.1f} peakvram {peak} MiB", flush=True)

    if smoke:
        # 64-token greedy stream per mach1 arm, compared pairwise to m1/exact
        prompts = "The capital of France is"
        streams = {}
        for arm in arm_list:
            model_key, extra = ARMS[arm]
            if model_key != "mach1":
                continue
            env = dict(os.environ, **extra)
            cmd = (f"/work/build/bin/llama-cli -m {MODELS[model_key]} -no-cnv -st --simple-io "
                   f"-ngl 999 -fa 1 -c 1024 --no-warmup --temp 0 -n 64 -p '{prompts}' "
                   f"--no-display-prompt 2>/dev/null < /dev/null")
            try:
                j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=900)
            except subprocess.TimeoutExpired:
                print(f"smoke {arm}: TIMEOUT", flush=True)
                continue
            # console::log timings land on STDOUT - drop per-run noise, keep tokens
            toks = [l for l in j.stdout.splitlines()
                    if "| Generation:" not in l and "Exiting" not in l]
            streams[arm] = "\n".join(toks).strip()
        ref = streams.get("m1")
        refname = "m1"
        if ref is None and streams:
            # no m1 arm in the run: the FIRST mach1 arm is the reference, so
            # bit-identity claims read as pairwise verdicts against it
            refname = next(iter(streams))
            ref = streams[refname]
        for arm, s in streams.items():
            verdict = "MATCH" if ref is not None and s == ref else "DIFFERS"
            print(f"smoke {arm}: {verdict} (vs {refname})")
            with open(f"{outdir}/smoke_{arm}.txt", "w") as f:
                f.write(s)
        VOL.commit()

    print("\n=== paired summary (tg128 tok/s) ===")
    for arm in arm_list:
        rr = results[arm]
        mean = sum(rr)/len(rr) if rr else 0.0
        print(f"{arm:8s} {' '.join(f'{x:.2f}' for x in rr)}  mean {mean:.2f}")
    print("\n=== prefill + peak VRAM ===")
    for arm in arm_list:
        pr = pp[arm]
        vv = vram[arm]
        pmean = sum(pr)/len(pr) if pr else 0.0
        print(f"{arm:12s} pp {' '.join(f'{x:.1f}' for x in pr)}  mean {pmean:.1f}"
              f"   peakvram {max(vv) if vv else 0} MiB")
    return {a: results[a] for a in arm_list}


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=3600)
def bitdump(tag: str):
    """Bitwise A/B of the rt fold paths per decode shape (pocs/mach1-bitdump).

    Bit-exact arms are compared with filecmp (IDENTICAL required). The
    GGML_MACH1_TC_FWHT arms are EXPECTED to differ at the bit level (fp16
    two-dot); they get a tolerance check instead: finite everywhere and
    rel-RMS <= 1e-3 vs the exact path (probe: 2.8e-4 on uniform inputs).
    """
    import array
    import filecmp
    import math

    def relcmp(ref_path, cand_path, label):
        a = array.array("f")
        b = array.array("f")
        with open(ref_path, "rb") as fh:
            a.frombytes(fh.read())
        with open(cand_path, "rb") as fh:
            b.frombytes(fh.read())
        if len(a) != len(b):
            print(f"bitdump {label}: LENGTH MISMATCH -> FAIL", flush=True)
            return
        finite = all(math.isfinite(v) for v in b)
        num = sum((x - y)*(x - y) for x, y in zip(a, b))
        den = sum(x*x for x in a)
        rel_rms = math.sqrt(num/den) if den > 0 else math.sqrt(num)
        dmax = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        ok = finite and rel_rms <= 1e-3
        print(f"bitdump {label}: rel-RMS {rel_rms:.2e} max|d| {dmax:.2e} "
              f"{'finite' if finite else 'INF/NAN'} -> {'PASS' if ok else 'FAIL'}", flush=True)

    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-bitdump")
    arms = {
        "base":       {"GGML_MACH1_TC_FWHT": "0"},
        "ofuse":      {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_OFUSE": "1"},
        "base-nouf":  {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_UFUSE_MAXB": "0"},
        "ofuse-nouf": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_OFUSE": "1", "GGML_MACH1_UFUSE_MAXB": "0"},
        # TC-FWHT arms (tolerance-compared, see the docstring); TIME=1 proves
        # engagement via the rt_u_tc/rt_out_tc keys in the exit table
        "tcf":        {"GGML_MACH1_TC_FWHT": "1", "GGML_MACH1_TIME": "1"},
        "tcf-nouf":   {"GGML_MACH1_TC_FWHT": "1", "GGML_MACH1_UFUSE_MAXB": "0", "GGML_MACH1_TIME": "1"},
        "tcf-ofuse":  {"GGML_MACH1_TC_FWHT": "1", "GGML_MACH1_OFUSE": "1", "GGML_MACH1_TIME": "1"},
    }
    tc_err: dict[str, str] = {}
    for shp in ["8192,2048", "2048,4096", "4096,2048", "512,2048", "2048,512"]:
        outs = {}
        for name, extra in arms.items():
            env = dict(os.environ, **extra)
            path = f"/tmp/bd_{name}.bin"
            r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} {shp}",
                               shell=True, capture_output=True, text=True, env=env)
            outs[name] = path if r.returncode == 0 else None
            if r.returncode != 0:
                print(f"bitdump {shp} {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
            elif name.startswith("tcf"):
                tc_err[name] = tc_err.get(name, "") + r.stderr
        for name, ref in [("ofuse", "base"), ("base-nouf", "base"), ("ofuse-nouf", "base-nouf")]:
            if outs.get(name) and outs.get(ref):
                same = filecmp.cmp(outs[ref], outs[name], shallow=False)
                print(f"bitdump {shp} {name} vs {ref}: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
        for name, ref in [("tcf", "base"), ("tcf-nouf", "base-nouf"), ("tcf-ofuse", "base")]:
            if outs.get(name) and outs.get(ref):
                relcmp(outs[ref], outs[name], f"{shp} {name} vs {ref} (tolerance)")
    for name, key in [("tcf", "rt_out_tc"), ("tcf-nouf", "rt_u_tc"), ("tcf-nouf", "rt_out_tc")]:
        engaged = key in tc_err.get(name, "")
        print(f"bitdump tcf engagement ({name}): {key} {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)

    # shared-expert fused region (GGML_MACH1_SHEXP_FFN): fused vs node path.
    # GGML_MACH1_TIME=1 on the fused arm proves the matcher engaged (its
    # shexp_gu/shexp_down keys appear in the exit table); syncs do not change
    # bits.
    outs = {}
    for name, extra in {"shexp0": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_SHEXP_FFN": "0"},
                        "shexp1": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_SHEXP_FFN": "1", "GGML_MACH1_TIME": "1"}}.items():
        env = dict(os.environ, **extra)
        path = f"/tmp/bd_{name}.bin"
        r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} --shexp",
                           shell=True, capture_output=True, text=True, env=env)
        outs[name] = path if r.returncode == 0 else None
        if r.returncode != 0:
            print(f"bitdump shexp {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
        elif name == "shexp1":
            engaged = "shexp_gu" in r.stderr and "shexp_down" in r.stderr
            print(f"bitdump shexp matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
    if outs.get("shexp0") and outs.get("shexp1"):
        same = filecmp.cmp(outs["shexp0"], outs["shexp1"], shallow=False)
        print(f"bitdump shexp shexp1 vs shexp0: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)

    # GDN v_tiled fold (GGML_MACH1_VTILED): fused vs node path over both glue
    # patterns; the nouf arms force the walk's UFUSE=0 instantiations. TIME=1
    # on the fused arms proves engagement (vtiled_walk keys in the exit table).
    outs = {}
    for name, extra in {"vtiled0":      {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_VTILED": "0"},
                        "vtiled1":      {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_VTILED": "1", "GGML_MACH1_TIME": "1"},
                        "vtiled0-nouf": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_VTILED": "0", "GGML_MACH1_UFUSE_MAXB": "0"},
                        "vtiled1-nouf": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_VTILED": "1", "GGML_MACH1_UFUSE_MAXB": "0",
                                         "GGML_MACH1_TIME": "1"},
                        "vtiled1-tcf":  {"GGML_MACH1_VTILED": "1", "GGML_MACH1_TC_FWHT": "1",
                                         "GGML_MACH1_TIME": "1"}}.items():
        env = dict(os.environ, **extra)
        path = f"/tmp/bd_{name}.bin"
        r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} --vtiled",
                           shell=True, capture_output=True, text=True, env=env)
        outs[name] = path if r.returncode == 0 else None
        if r.returncode != 0:
            print(f"bitdump vtiled {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
        elif name == "vtiled1-tcf":
            engaged = "vtiled_walk_tc" in r.stderr
            print(f"bitdump vtiled tcf matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
        elif name.startswith("vtiled1"):
            engaged = "vtiled_walk" in r.stderr
            print(f"bitdump vtiled matcher ({name}): {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
    for name, ref in [("vtiled1", "vtiled0"), ("vtiled1-nouf", "vtiled0-nouf")]:
        if outs.get(name) and outs.get(ref):
            same = filecmp.cmp(outs[ref], outs[name], shallow=False)
            print(f"bitdump vtiled {name} vs {ref}: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
    if outs.get("vtiled1-tcf") and outs.get("vtiled0"):
        relcmp(outs["vtiled0"], outs["vtiled1-tcf"], "vtiled vtiled1-tcf vs vtiled0 (tolerance)")

    # same-input rt batching (GGML_MACH1_QKV_BATCH): batched vs per-op paths
    # over the attention trio and the GDN qkvz pair; the nouf arms force the
    # rt_u + UFUSE=0 instantiations. TIME=1 on the batched arms proves
    # engagement of BOTH patterns (qkvb_walk and qkvzb_walk keys).
    outs = {}
    for name, extra in {"qkvb0":      {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_QKV_BATCH": "0"},
                        "qkvb1":      {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_QKV_BATCH": "1", "GGML_MACH1_TIME": "1"},
                        "qkvb0-nouf": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_QKV_BATCH": "0", "GGML_MACH1_UFUSE_MAXB": "0"},
                        "qkvb1-nouf": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_QKV_BATCH": "1", "GGML_MACH1_UFUSE_MAXB": "0",
                                       "GGML_MACH1_TIME": "1"},
                        "qkvb1-tcf":  {"GGML_MACH1_QKV_BATCH": "1", "GGML_MACH1_TC_FWHT": "1",
                                       "GGML_MACH1_TIME": "1"}}.items():
        env = dict(os.environ, **extra)
        path = f"/tmp/bd_{name}.bin"
        r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} --qkv",
                           shell=True, capture_output=True, text=True, env=env)
        outs[name] = path if r.returncode == 0 else None
        if r.returncode != 0:
            print(f"bitdump qkv {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
        elif name == "qkvb1-tcf":
            engaged = "qkvb_walk_tc" in r.stderr and "qkvzb_walk_tc" in r.stderr
            print(f"bitdump qkv tcf matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
        elif name.startswith("qkvb1"):
            engaged = "qkvb_walk" in r.stderr and "qkvzb_walk" in r.stderr
            print(f"bitdump qkv matcher ({name}): {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
    for name, ref in [("qkvb1", "qkvb0"), ("qkvb1-nouf", "qkvb0-nouf")]:
        if outs.get(name) and outs.get(ref):
            same = filecmp.cmp(outs[ref], outs[name], shallow=False)
            print(f"bitdump qkv {name} vs {ref}: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
    if outs.get("qkvb1-tcf") and outs.get("qkvb0"):
        relcmp(outs["qkvb0"], outs["qkvb1-tcf"], "qkv qkvb1-tcf vs qkvb0 (tolerance)")

    # GDN-core fused region (GGML_MACH1_GDN_FUSE): fused vs node path over the
    # norm/gate/delta/gated-norm chain at the real decode dims. TIME=1 on the
    # fused arm proves engagement (gdn_core key in the exit table).
    outs = {}
    for name, extra in {"gdnf0": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_GDN_FUSE": "0"},
                        "gdnf1": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_GDN_FUSE": "1", "GGML_MACH1_TIME": "1"},
                        # full region: state gather folded (1), plus conv+silu (2)
                        "gdnx1": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_GDN_FULL": "1", "GGML_MACH1_TIME": "1"},
                        "gdnx2": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_GDN_FULL": "2", "GGML_MACH1_TIME": "1"}}.items():
        env = dict(os.environ, **extra)
        path = f"/tmp/bd_{name}.bin"
        r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} --gdn",
                           shell=True, capture_output=True, text=True, env=env)
        outs[name] = path if r.returncode == 0 else None
        if r.returncode != 0:
            print(f"bitdump gdn {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
        elif name == "gdnf1":
            engaged = "gdn_core" in r.stderr
            print(f"bitdump gdn matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
        elif name in ("gdnx1", "gdnx2"):
            key = "gdn_full_conv" if name == "gdnx2" else "gdn_full "
            engaged = key in r.stderr
            print(f"bitdump gdn {name} matcher ({key.strip()}): "
                  f"{'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
    for name in ["gdnf1", "gdnx1", "gdnx2"]:
        if outs.get("gdnf0") and outs.get(name):
            same = filecmp.cmp(outs["gdnf0"], outs[name], shallow=False)
            print(f"bitdump gdn {name} vs gdnf0: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)

    # routed-expert tier (GGML_MACH1_EXP_FFN / GGML_MACH1_EXP_MEGA): the two
    # fused forms vs the per-op node path over the decode expert chain. TIME=1
    # on the fused arms proves engagement (exp_ffn_gu / exp_mega keys).
    outs = {}
    for name, extra in {"exp0":    {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_FFN": "0", "GGML_MACH1_EXP_MEGA": "0"},
                        "expffn":  {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_FFN": "1", "GGML_MACH1_TIME": "1"},
                        "expmega": {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_MEGA": "1", "GGML_MACH1_TIME": "1"},
                        "expdn":   {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_MEGA": "1",
                                    "GGML_MACH1_EXP_DNROWS": "1", "GGML_MACH1_TIME": "1"},
                        # integer walks: tolerance-compared, not bit-exact
                        "zdp":     {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_ZDP": "1",
                                    "GGML_MACH1_TIME": "1"},
                        "dp4a":    {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_DP4A": "1",
                                    "GGML_MACH1_TIME": "1"},
                        # the same walk on the split path (mega off, table in global)
                        "spdp4a":  {"GGML_MACH1_TC_FWHT": "0", "GGML_MACH1_EXP_FFN": "0",
                                    "GGML_MACH1_EXP_MEGA": "0", "GGML_MACH1_EXP_DP4A": "1",
                                    "GGML_MACH1_TIME": "1"}}.items():
        env = dict(os.environ, **extra)
        path = f"/tmp/bd_{name}.bin"
        r = subprocess.run(f"/work/build/bin/llama-mach1-bitdump {path} --exp",
                           shell=True, capture_output=True, text=True, env=env)
        outs[name] = path if r.returncode == 0 else None
        if r.returncode != 0:
            print(f"bitdump exp {name}: RUN FAIL {r.stderr[-300:]}", flush=True)
        elif name == "expffn":
            engaged = "exp_ffn_gu" in r.stderr and "exp_ffn_down" in r.stderr
            print(f"bitdump exp expffn matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
        elif name in ("expmega", "expdn"):
            engaged = "exp_mega" in r.stderr
            print(f"bitdump exp {name} matcher: {'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
        elif name in ("zdp", "dp4a", "spdp4a"):
            # zdp=<mode> in the exp_mega / exp_fused stage key is what proves
            # the integer walk ran: 1 = q8 activations, 2 = the 15-bit planes
            stage = "exp_fused" if name == "spdp4a" else "exp_mega"
            key = "zdp=1" if name == "zdp" else "zdp=2"
            engaged = any(stage in l and key in l for l in r.stderr.splitlines())
            print(f"bitdump exp {name} walk ({stage} {key}): "
                  f"{'ENGAGED' if engaged else 'NOT ENGAGED'}", flush=True)
    for name in ["expffn", "expmega", "expdn"]:
        if outs.get(name) and outs.get("exp0"):
            same = filecmp.cmp(outs["exp0"], outs[name], shallow=False)
            print(f"bitdump exp {name} vs exp0: {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
    for name in ["zdp", "dp4a", "spdp4a"]:
        if outs.get(name) and outs.get("exp0"):
            relcmp(outs["exp0"], outs[name], f"exp {name} vs exp0 (tolerance)")


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=1800)
def splitk_bitdump_l40s(tag: str):
    """Fail-closed exact-shape numeric and determinism gate for KS4.

    This intentionally builds and runs only one m2048,n4096,nt16 RT op. It is
    the cheapest non-vacuous device gate: the control proves the existing KS1
    route, the candidate proves both KS4 entries, and a second candidate
    process proves deterministic fixed-order partial folding.
    """
    import array
    import filecmp
    import math

    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-bitdump")

    split_marker = (
        "mach1: native rt_imma8 split-K4 m2048 n4096 nt16 ENGAGED "
        "(384 KiB transient, 0 persistent)"
    )
    cp_marker = (
        "mach1: zero-VRAM native rt_imma8 cp.async double-buffer "
        "ENGAGED (s_red alias)"
    )
    control_key = "mach1-time: rt_imma8 m=2048 n=4096 nt=16"
    control_out_key = "mach1-time: rt_out_tc m=2048 n=4096 nt=16"
    candidate_key = "mach1-time: rt_imma8_split4 m=2048 n=4096 nt=16"
    candidate_out_key = "mach1-time: rt_out_tc_split4 m=2048 n=4096 nt=16"
    binary = "/work/build/bin/llama-mach1-bitdump"
    runs: dict[str, dict[str, object]] = {}
    for name, split_on in (("control", False), ("candidate0", True),
                           ("candidate1", True)):
        extra = dict(ARMS["p16combo"][1])
        extra.update({
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "GGML_MACH1_DEBUG": "1",
            "GGML_MACH1_TIME": "1",
            "GGML_MACH1_RT_IMMA8_SPLITK": "1" if split_on else "0",
        })
        path = f"/tmp/splitk_{name}.bin"
        try:
            run = subprocess.run(
                [binary, path, "--splitk"], capture_output=True, text=True,
                env=dict(os.environ, **extra), timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"split-K bitdump {name} timed out") from exc
        log = run.stdout + run.stderr
        if run.returncode != 0:
            raise RuntimeError(
                f"split-K bitdump {name} failed rc={run.returncode}:\n{log[-4000:]}"
            )
        marker_counts = {
            "cpasync": log.count(cp_marker),
            "split4": log.count(split_marker),
            "control_core": log.count(control_key),
            "control_out": log.count(control_out_key),
            "candidate_core": log.count(candidate_key),
            "candidate_out": log.count(candidate_out_key),
        }
        expected = ({"cpasync": 1, "split4": 1, "control_core": 0,
                     "control_out": 0,
                     "candidate_core": 1, "candidate_out": 1}
                    if split_on else
                    {"cpasync": 1, "split4": 0, "control_core": 1,
                     "control_out": 1,
                     "candidate_core": 0, "candidate_out": 0})
        if marker_counts != expected:
            raise RuntimeError(
                f"split-K bitdump {name} engagement mismatch: "
                f"expected={expected} actual={marker_counts}\n{log[-4000:]}"
            )
        runs[name] = {"path": path, "marker_counts": marker_counts}
        print(f"SPLITK_BITDUMP_RUN {name} "
              f"{json.dumps(marker_counts, sort_keys=True)}", flush=True)

    def load_f32(path: str) -> array.array:
        values = array.array("f")
        with open(path, "rb") as handle:
            values.frombytes(handle.read())
        return values

    control = load_f32(str(runs["control"]["path"]))
    candidate = load_f32(str(runs["candidate0"]["path"]))
    expected_values = 2048 * 16
    if len(control) != expected_values or len(candidate) != expected_values:
        raise RuntimeError(
            f"split-K bitdump length mismatch: control={len(control)} "
            f"candidate={len(candidate)} expected={expected_values}"
        )
    finite = all(math.isfinite(v) for v in control) and all(
        math.isfinite(v) for v in candidate
    )
    delta_l2 = sum((a - b) * (a - b) for a, b in zip(control, candidate))
    ref_l2 = sum(a * a for a in control)
    rel_l2 = math.sqrt(delta_l2 / ref_l2) if ref_l2 > 0.0 else math.sqrt(delta_l2)
    max_abs_delta = max(abs(a - b) for a, b in zip(control, candidate))
    max_abs_ref = max(abs(a) for a in control)
    max_scaled = max_abs_delta / max(max_abs_ref, 1.0e-12)
    repeat_bitwise = filecmp.cmp(
        str(runs["candidate0"]["path"]), str(runs["candidate1"]["path"]),
        shallow=False,
    )
    passed = finite and rel_l2 <= 2.0e-5 and max_scaled <= 1.0e-4 and repeat_bitwise
    receipt = {
        "shape": {"m": 2048, "n": 4096, "nt": 16},
        "source_format_bpw": 4.0,
        "transient_bytes_delta": 3 * 16 * 2048 * 4,
        "persistent_bytes_delta": 0,
        "finite": finite,
        "relative_l2": rel_l2,
        "relative_l2_limit": 2.0e-5,
        "max_scaled": max_scaled,
        "max_scaled_limit": 1.0e-4,
        "candidate_repeat_bitwise": repeat_bitwise,
        "runs": runs,
        "passed": passed,
    }
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/p16_splitk_bitdump.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOL.commit()
    print(f"P16 SPLITK BITDUMP GATE: {'PASS' if passed else 'FAIL'} "
          f"relL2={rel_l2:.3e} max_scaled={max_scaled:.3e} "
          f"repeat={'BITWISE' if repeat_bitwise else 'DIFFERS'}", flush=True)
    if not passed:
        raise RuntimeError("p16 split-K bitdump gate failed")
    return receipt


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=3600)
def chainbench_l40s(tag: str, args: str = ""):
    """Ada (sm_89) lane: persistent rt-chain prefetch prototype."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return chainbench.local(tag, args)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=3600)
def chainbench(tag: str, args: str = ""):
    """Persistent rt-chain prefetch prototype (pocs/mach1-chainbench): arm A
    split walk+out launches vs arm B persistent kernel with cp.async trellis
    prefetch across the serial op chain."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-chainbench")
    r = subprocess.run(f"/work/build/bin/llama-mach1-chainbench {args}",
                       shell=True, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise RuntimeError("chainbench failed")


def _standard_output_ptxas_records(build_log: str):
    """Extract the six exact timing-kernel resource records.

    Runtime localSizeBytes is checked again by the executable.  This compile
    gate is the authoritative stack/spill receipt that cudaFuncAttributes
    cannot expose.
    """
    specs = [
        {"name": "control_f32", "needles": ("stdob_control_kernel", "ILb0EE"), "max_regs": 128},
        {"name": "control_f16", "needles": ("stdob_control_kernel", "ILb1EE"), "max_regs": 128},
        {"name": "row_scale_f32", "needles": ("stdob_row_scale_kernel", "ILb0EE"), "max_regs": 64},
        {"name": "row_scale_f16", "needles": ("stdob_row_scale_kernel", "ILb1EE"), "max_regs": 64},
        {"name": "split_direct", "needles": ("stdob_split_sum_kernel", "ILi1EE"), "max_regs": 64},
        {"name": "split_sum4", "needles": ("stdob_split_sum_kernel", "ILi4EE"), "max_regs": 64},
    ]
    lines = build_log.splitlines()
    receipts = []
    for spec in specs:
        starts = [i for i, line in enumerate(lines)
                  if "Compiling entry function" in line and
                  all(needle in line for needle in spec["needles"])]
        if len(starts) != 1:
            matches = [line for line in lines
                       if all(needle in line for needle in spec["needles"])]
            raise RuntimeError(
                f"expected one ptxas entry for {spec['name']}, starts={len(starts)}, "
                f"matches={matches[-10:]}"
            )
        start = starts[0]
        end = next((i for i in range(start + 1, len(lines))
                    if "Compiling entry function" in lines[i]), len(lines))
        record = "\n".join(lines[start:end])
        frame = re.search(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
                          r"(\d+) bytes spill loads", record)
        regs = re.search(r"Used\s+(\d+) registers", record)
        if frame is None or regs is None:
            raise RuntimeError(f"incomplete ptxas record for {spec['name']}:\n{record}")
        stack_bytes, spill_stores, spill_loads = (int(v) for v in frame.groups())
        registers = int(regs.group(1))
        passed = (stack_bytes == 0 and spill_stores == 0 and spill_loads == 0 and
                  registers <= spec["max_regs"])
        receipts.append({
            "name": spec["name"],
            "needles": list(spec["needles"]),
            "registers": registers,
            "max_registers": spec["max_regs"],
            "stack_bytes": stack_bytes,
            "spill_store_bytes": spill_stores,
            "spill_load_bytes": spill_loads,
            "passed": passed,
            "ptxas_record": record,
        })
        print(record, flush=True)
        print(f"STANDARD OUTPUT {spec['name']} PTXAS: "
              f"{'PASS' if passed else 'FAIL'} regs={registers}/{spec['max_regs']} "
              f"stack={stack_bytes} spill_store={spill_stores} "
              f"spill_load={spill_loads}", flush=True)
    return receipts


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=3600, cpu=32)
def standard_output_ptxgate(tag: str, arch: str = "89"):
    """CPU-only compile gate for the standard-output timing falsifier."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DLLAMA_CURL=OFF")
    built = subprocess.run(
        ["cmake", "--build", "/work/build", "-j", "--target", "llama-mach1-chainbench"],
        capture_output=True, text=True)
    build_log = built.stdout + built.stderr
    if built.returncode != 0:
        print(build_log[-12000:], flush=True)
        raise RuntimeError("standard-output chainbench ptxas build failed")
    receipts = _standard_output_ptxas_records(build_log)
    result = {
        "arch": arch,
        "passed": all(r["passed"] for r in receipts),
        "kernels": receipts,
    }
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/standard_output_ptxgate.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    VOL.commit()
    print(f"STANDARD OUTPUT PTXAS GATE: {'PASS' if result['passed'] else 'FAIL'}",
          flush=True)
    if not result["passed"]:
        raise RuntimeError("standard-output ptxas resource gate failed; timing forbidden")
    return result


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=3600)
def standard_output_l40s(tag: str, args: str = "64 9"):
    """Exact 19,360-vector TC-out vs standard-basis timing falsifier."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-chainbench")
    cmd = f"/work/build/bin/llama-mach1-chainbench --standard-output-basis {args or '64 9'}"
    env = dict(os.environ, GGML_MACH1_TIME="1")
    print(f"+ GGML_MACH1_TIME=1 {cmd}", flush=True)
    run = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(run.stdout, flush=True)
    if run.returncode != 0:
        print(run.stderr[-8000:], flush=True)
        raise RuntimeError("standard-output timing falsifier failed")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/standard_output_l40s.txt", "w") as f:
        f.write(run.stdout + run.stderr)
    VOL.commit()
    return run.stdout


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=3600)
def layerbench_l40s(tag: str, args: str = ""):
    """Ada (sm_89) lane: layer-scale persistent megakernel probe.  Ada is
    bandwidth-scarce where Hopper is latency-bound, so the fusion verdict
    is not portable between them and has to be re-measured here."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return layerbench.local(tag, args)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=3600)
def layerbench(tag: str, args: str = ""):
    """Layer-scale persistent megakernel probe (pocs/mach1-chainbench --layer):
    one MoE layer's dependent weight chain as per-op launches (arm A) vs one
    persistent kernel with the p4 codebook in global (B), in shared (C) and
    with every independent op of a step batched (D)."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    blog = sh("cmake --build /work/build -j --target llama-mach1-chainbench 2>&1")
    # -Xptxas -v: registers/spill per kernel, straight from the compiler
    keep = [ln for ln in blog.splitlines()
            if "lay_persist_kernel" in ln or "Used" in ln or "spill" in ln]
    print("\n".join(keep[-400:]), flush=True)
    r = subprocess.run(f"/work/build/bin/llama-mach1-chainbench --layer {args}",
                       shell=True, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise RuntimeError("chainbench --layer failed")


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=5400)
def dp4abench(tag: str, args: str = ""):
    """Integer-lattice walk probe (pocs/mach1-chainbench --layer --dp4a): the
    expert walk's per-weight tail as two __dp4a instead of eight nibble
    extracts and sixteen fp32 ops, against the production fp32 body, plus the
    strip ladder that decomposes what the state loop actually spends."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    blog = sh("cmake --build /work/build -j --target llama-mach1-chainbench 2>&1")
    keep = [ln for ln in blog.splitlines()
            if "lay_persist_kernel" in ln or "Used" in ln or "spill" in ln]
    print("\n".join(keep[-400:]), flush=True)
    r = subprocess.run(f"/work/build/bin/llama-mach1-chainbench --layer --dp4a {args}",
                       shell=True, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise RuntimeError("chainbench --dp4a failed")


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=3600)
def getdraft(url: str = ""):
    """Fetch a small same-tokenizer draft gguf to /vol/models/draft.gguf."""
    import os as _os
    cands = [url] if url else [
        "https://huggingface.co/unsloth/Qwen3.6-1.7B-GGUF/resolve/main/Qwen3.6-1.7B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.6-1.7B-Instruct-GGUF/resolve/main/Qwen3.6-1.7B-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/Qwen/Qwen3.6-1.7B-GGUF/resolve/main/qwen3.6-1.7b-q4_k_m.gguf",
        "https://huggingface.co/unsloth/Qwen3.6-0.6B-GGUF/resolve/main/Qwen3.6-0.6B-Q4_K_M.gguf",
    ]
    for u in cands:
        if not u:
            continue
        print(f"trying {u}", flush=True)
        r = subprocess.run(f"curl -sSfL -o /vol/models/draft.gguf '{u}'",
                           shell=True, capture_output=True, text=True)
        if r.returncode == 0 and _os.path.getsize("/vol/models/draft.gguf") > 100_000_000:
            print(f"OK {u} -> {_os.path.getsize('/vol/models/draft.gguf')/1e9:.2f} GB", flush=True)
            VOL.commit()
            return
        print(f"failed: {r.stderr[-300:]}", flush=True)
    raise RuntimeError("no draft candidate worked")


@APP.function(image=IMAGE, gpu="L40S", cpu=16, volumes={"/vol": VOL}, timeout=2*3600)
def spec_l40s(arms: str, tag: str):
    """Ada lane: speculative decode with the small draft."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return spec.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", cpu=16, volumes={"/vol": VOL}, timeout=2*3600)
def spec(arms: str, tag: str):
    """Speculative decode: llama-speculative with /vol/models/draft.gguf as
    the drafter, K swept. The verify step is a small-batch forward - the
    codec's good regime - so this is where the m1/q4km gap should inVERT."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-speculative")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    prompt = "The history of computing begins with"
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        for K in (4, 8, 12):
            cmd = (f"/work/build/bin/llama-speculative -m {MODELS[model_key]} "
                   f"-md /vol/models/draft.gguf -ngl 999 -ngld 999 -fa 1 "
                   f"--draft-max {K} --draft-min 1 --temp 0 -c 2048 -n 256 "
                   f"-p '{prompt}' 2>&1")
            print(f"--- spec arm {arm} K={K} ---", flush=True)
            j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=1200)
            keep = [l for l in (j.stdout + j.stderr).splitlines()
                    if "accept" in l or "decoded" in l or "draft" in l.lower()[:10]
                    or "t/s" in l or "error" in l.lower()]
            print("\n".join(keep[-12:]), flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def graphstat_l40s(arms: str, tag: str):
    """Ada lane of the graph capture/replay census."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return graphstat.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def graphstat(arms: str, tag: str):
    """Does this MoE model actually REPLAY its cuda graph, or recapture?

    Round 121 put the codec at 78% of the step and 2.4x the dense path,
    and nothing queued is plausibly a 2.4x.  One candidate is that the
    graph is being rebuilt every token - expert routing changes each
    step - in which case the ~5-6 us per-op launch cost round 117
    measured is being paid in SERVING, not just in the graphs-disabled
    census, and collapsing it is worth far more than any single kernel.

    ggml logs capture-vs-replay counts under GGML_MACH1_TIME >= 4, and
    crucially WITHOUT disabling graphs - which is why the censuses
    (TIME=3 plus GGML_CUDA_DISABLE_GRAPHS=1) could never show it."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-bench")
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra, GGML_MACH1_TIME="4")
        env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
               f"-fa 1 -p 16 -n 120 -r 1 -o json")
        print(f"--- graphstat arm {arm} ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           env=env)
        # graphstat: direct/capture/replay counts and the nodes whose
        # properties flip.  expffn: whether the expert-FFN matcher engages,
        # and if not, the source line of the guard that rejected it - a
        # fused region is skipped before it is ever timed, so a matcher
        # that stopped matching is invisible to the per-op census (r133).
        keys = [l for l in j.stderr.splitlines()
                if "graphstat" in l or "graphflip" in l
                or "graph-prop-flip" in l or "expffn" in l]
        print("\n".join(keys[-40:]) or "(no graphstat lines)", flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def stagetime_l40s(arms: str, tag: str):
    """Ada (sm_89) lane: per-stage mach1 timers."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return stagetime.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def stagetime(arms: str, tag: str):
    """Internal mach1 stage keys: GGML_MACH1_TIME=1 + GGML_CUDA_DISABLE_GRAPHS=1
    (TIME=2/3 emit ggml op names and can never show a stage key). The key
    carries the shape string, which is how a shape-gated dispatch is checked."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-bench")
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        extra = dict(extra)
        # BENCH_UB/BENCH_P/BENCH_N give this lane a PREFILL-shaped profile; the
        # default -p 16 keeps every prefill lane and the bank below their token
        # floors, so a stage key that only exists past one never appears
        ub   = extra.pop("BENCH_UB", None)
        plen = extra.pop("BENCH_P", None)
        ngen = extra.pop("BENCH_N", "120")
        extra.pop("BENCH_V", None)
        for k in ("GGML_MACH1_TIME", "GGML_CUDA_DISABLE_GRAPHS", "GGML_MACH1_DEBUG"):
            extra.pop(k, None)
        env = dict(os.environ, **extra,
                   GGML_MACH1_TIME="1", GGML_CUDA_DISABLE_GRAPHS="1",
                   GGML_MACH1_DEBUG="2")
        cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
               f"-fa 1 -p {plen or 16} -n {ngen} -r 1 -o json")
        if ub:
            cmd += f" -b {ub} -ub {ub}"
        print(f"--- stagetime arm {arm} ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        if j.returncode != 0:
            print(f"ARM FAILED rc={j.returncode}", flush=True)
            print((j.stdout + j.stderr)[-2000:], flush=True)
            continue
        keys = [l for l in j.stderr.splitlines()
                if "mach1-time" in l or "exp_mega" in l or "mach1-gdnprev" in l
                or "mach1-ffnprev" in l or "fuse BAIL" in l or "fuse SEES" in l
                or ("mach1-lattice" in l and "wave_gamma" not in l) or "tlut repack" in l]
        print("\n".join(keys[-120:]), flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def census_l40s(arms: str, tag: str):
    """Ada (sm_89) lane: per-op decode/prefill census."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return census.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def census(arms: str, tag: str):
    """Per-op decode/prefill census: llama-bench under GGML_MACH1_TIME=3 +
    GGML_CUDA_DISABLE_GRAPHS=1, full cuda-op-time table printed per phase mix.
    A decode-heavy run (-p 16 -n 120) and a prefill-heavy run (-p 2048 -n 8)
    separate which phase an op count belongs to."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        # dict-literal merge: census arms may already pin TIME/GRAPHS in extra
        # (optq does), and dict(**extra, GGML_MACH1_TIME=...) would TypeError
        env = {**os.environ, **extra,
               "GGML_MACH1_TIME": "3", "GGML_CUDA_DISABLE_GRAPHS": "1"}
        for phase, pn in [("decode", "-p 16 -n 120"), ("prefill", "-p 2048 -n 8"),
                          ("prefill2k", "-p 2048 -n 8 -ub 2048")]:
            cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
                   f"-fa 1 {pn} -r 1 -o json")
            print(f"--- census arm {arm} phase {phase} ---", flush=True)
            j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            table = [l for l in j.stderr.splitlines() if l.startswith("cuda-op-time:")]
            print("\n".join(table), flush=True)
            with open(f"{outdir}/census_{arm}_{phase}.txt", "w") as f:
                f.write(j.stderr)
            VOL.commit()


@APP.function(image=NCU_IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def ncuprof_l40s(arms: str, tag: str, kregex: str = "mach1", npl: str = "2", count: int = 12,
                 skip: int = 0):
    """Per-kernel stall analysis via Nsight Compute on Ada (sm_89). Profiles
    the first `count` launches matching `kregex` in a short graphs-off decode
    run; SpeedOfLight + scheduler sections name the decode-ALU bound."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    import shutil as _shutil, glob as _glob
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    ncu = _shutil.which("ncu")
    if not ncu:
        cand = _glob.glob("/usr/local/cuda*/bin/ncu") + _glob.glob("/opt/nvidia/nsight-compute*/ncu")
        ncu = cand[0] if cand else None
    if not ncu:
        print("ncuprof: NO NCU BINARY IN IMAGE", flush=True)
        return
    print(f"ncuprof: using {ncu}", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra, GGML_CUDA_DISABLE_GRAPHS="1")
        print(f"ncuprof env check: NTLO={env.get('GGML_MACH1_NTLO')} "
              f"DEBUG={env.get('GGML_MACH1_DEBUG')} n_extra={len(extra)}", flush=True)
        stall = ("smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
                 .format("{}"))
        mets = ",".join(stall.format(s) for s in
                        ("barrier", "membar", "long_scoreboard", "short_scoreboard",
                         "wait", "not_selected", "lg_throttle", "mio_throttle",
                         "branch_resolving", "sleeping"))
        # duration + SoL throughputs so census captures can rank kernels by
        # time, not just stall shape
        mets += (",gpu__time_duration.sum"
                 ",sm__throughput.avg.pct_of_peak_sustained_elapsed"
                 ",gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        cmd = (f"{ncu} -k 'regex:{kregex}' -c {count} -s {skip} --set basic "
               f"--section SpeedOfLight --section SchedulerStats --section WarpStateStats "
               f"--metrics {mets} "
               f"--clock-control none --target-processes all --csv "
               f"/work/build/bin/llama-batched-bench -m {MODELS[model_key]} "
               f"-fa 1 -c 2048 -ub 512 -npp 32 -ntg 8 -npl {npl} -ngl 999 2>&1")
        print(f"--- ncuprof arm {arm} kregex={kregex} npl={npl} ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=3000)
        out = j.stdout + j.stderr
        with open(f"{outdir}/ncu_{arm}_pl{npl}.csv", "w") as f:
            f.write(out)
        VOL.commit()
        err = [l for l in out.splitlines() if "ERR" in l or "error" in l.lower()][:6]
        keep = [l for l in out.splitlines()
                if ("mach1" in l and "," in l) or "Kernel Name" in l][:8]
        print("\n".join(err + keep) or out[-1200:], flush=True)


@APP.function(image=NSYS_IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def nsysprof_l40s(arms: str, tag: str, npl: str = "2"):
    """Ada (sm_89) lane of the graphs-on kernel profile."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return nsysprof.local(arms, tag, npl)


@APP.function(image=NSYS_IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def nsysprof(arms: str, tag: str, npl: str = "2"):
    """Graphs-ON per-kernel decomposition via nsys.

    The event census (TIME=1 + graphs off) inflates every drained-stream
    launch by the host launch latency - the start event completes, the
    stream idles for the launch, and the gap lands inside the measured
    window. Summing those and subtracting from the graphs-on wall cannot
    size the stock-kernel share at small nt: r277's residual came out
    BELOW the plain launch-count floor of the stock GDN chain. nsys reads
    kernel begin/end from the driver for every kernel in the replayed
    graph - stock included - with nothing injected into the stream.

    One npl value per profile so decode kernels do not aggregate across
    nt under the same name; npp=32 keeps the prefill share of same-named
    kernels negligible against 128 decode steps."""
    import glob as _glob
    import shutil as _shutil
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    nsys = _shutil.which("nsys")
    if not nsys:
        cand = (_glob.glob("/opt/nvidia/nsight-systems*/bin/nsys") +
                _glob.glob("/opt/nvidia/nsight-systems*/target-linux-x64/nsys") +
                _glob.glob("/usr/local/cuda*/bin/nsys"))
        nsys = cand[0] if cand else None
    if not nsys:
        print("nsysprof: NO NSYS BINARY IN IMAGE", flush=True)
        return
    print(f"nsysprof: using {nsys}", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        for pl in [p.strip() for p in npl.split(",")]:
            # "pl@npp" runs a prefill-shaped profile (large npp, 1 decode step)
            npp, ntg = 32, 128
            if "@" in pl:
                pl, npp_s = pl.split("@", 1)
                npp, ntg = int(npp_s), 1
            rep = f"/tmp/prof_{arm}_pl{pl}"
            gflag = "--cuda-graph-trace=node " if ntg > 1 else ""
            cmd = (f"{nsys} profile -t cuda -s none --cpuctxsw=none "
                   f"{gflag}"
                   f"--force-overwrite=true -o {rep} "
                   f"/work/build/bin/llama-batched-bench -m {MODELS[model_key]} "
                   f"-fa 1 -c 8192 -npp {npp} -ntg {ntg} -npl {pl} -ngl 999 2>&1")
            print(f"--- nsysprof arm {arm} npl={pl} ---", flush=True)
            j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            rows = [l for l in (j.stdout + j.stderr).splitlines() if "|" in l and " 32 " in l]
            print("\n".join(rows[-2:]), flush=True)
            s = subprocess.run(f"{nsys} stats --force-export=true -r cuda_gpu_kern_sum --format csv {rep}.nsys-rep",
                               shell=True, capture_output=True, text=True)
            out = s.stdout + s.stderr
            if "does not contain CUDA kernel data" in out:
                s2 = subprocess.run(f"{nsys} stats -r cuda_api_sum --format csv {rep}.nsys-rep",
                                    shell=True, capture_output=True, text=True)
                out += "\n=== api_sum fallback ===\n" + (s2.stdout + s2.stderr)[:3000]
            with open(f"{outdir}/nsys_{arm}_pl{pl}.csv", "w") as f:
                f.write((j.stdout + j.stderr)[-1500:] + "\n=== kern_sum ===\n" + out)
            csv = [l for l in out.splitlines() if "," in l]
            print("\n".join(csv[:45]), flush=True)
            VOL.commit()


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def mpsserve_l40s(arms: str, tag: str, nproc: str = "2,4"):
    """Concurrent B=1 serving under CUDA MPS: N independent llama-batched-bench
    processes (npl=1) share the GPU. The m1 decode is latency-bound with idle
    SMs, which MPS converts to aggregate throughput; the aggregate tg vs the
    batched-bench npl=N rows is the serving comparison."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    subprocess.run("mkdir -p /tmp/mps /tmp/mpslog && "
                   "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mpslog "
                   "nvidia-cuda-mps-control -d", shell=True)
    print("mps daemon started", flush=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra, CUDA_MPS_PIPE_DIRECTORY="/tmp/mps")
        for np_ in [int(x) for x in nproc.split(",")]:
            cmd = (f"/work/build/bin/llama-batched-bench -m {MODELS[model_key]} "
                   f"-fa 1 -c 4096 -npp 128 -ntg 128 -npl 1 -ngl 999")
            t1 = time.time()
            procs = [subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, env=env)
                     for _ in range(np_)]
            outs = [p.communicate()[0] for p in procs]
            wall = time.time() - t1
            tgs = []
            for o in outs:
                rows = [l for l in o.splitlines() if l.startswith("|   128 |    128 |    1 |")]
                if rows:
                    tgs.append(float(rows[-1].split("|")[8]))
            agg = sum(tgs)
            print(f"mpsserve {arm} nproc={np_}: per-proc tg={['%.1f' % t for t in tgs]} "
                  f"aggregate={agg:.1f} t/s wall={wall:.0f}s", flush=True)
    subprocess.run("echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps nvidia-cuda-mps-control",
                   shell=True)


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=2*3600, cpu=16, memory=49152)
def cpucheck(arms: str, tag: str):
    """Release-configuration CPU build + CPU-only throughput.

    Two questions the release pipeline depends on and nothing else answers:
    (1) does the fork build under the flags the release workflow uses -
    GGML_BACKEND_DL=ON splits the backends into loadable modules and
    GGML_CPU_ALL_VARIANTS=ON compiles several ISA variants, and our seven
    custom ops live in the core op enum with reference implementations in
    ggml-cpu; (2) is the CPU path fast enough that shipping CPU archives
    (and pointing Apple Silicon at them) is honest."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    b = subprocess.run(
        "cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON "
        "-DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF "
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
        "&& cmake --build /work/build -j --target llama-bench",
        shell=True, capture_output=True, text=True, errors="replace")
    if b.returncode != 0:
        print("RELEASE-CONFIG CPU BUILD FAILED", flush=True)
        print((b.stdout + b.stderr)[-4000:], flush=True)
        return
    print(f"release-config CPU build OK in {time.time()-t0:.0f}s "
          f"(BACKEND_DL + CPU_ALL_VARIANTS)", flush=True)
    print(subprocess.run("ls /work/build/bin/ | head -20", shell=True,
                         capture_output=True, text=True, errors="replace").stdout, flush=True)
    # Container diagnostics. cpu=16 is a REQUEST, not an isolation guarantee -
    # the same build measured 3.71 and 0.07 tg32 on two containers - so an
    # absolute CPU t/s is only interpretable next to the machine it came from
    # AND next to a control arm measured in the SAME container. Always pass the
    # teacher (q4km) alongside m1 and read the RATIO; the absolute number is
    # informational only.
    print("--- container ---", flush=True)
    print(subprocess.run(
        "nproc; grep -m1 'model name' /proc/cpuinfo; "
        "grep -o 'avx512_vnni\\|avx512f\\|avx2\\|f16c\\|fma' /proc/cpuinfo | sort -u | tr '\\n' ' '; echo; "
        "free -g | head -2; cat /sys/fs/cgroup/cpu.max 2>/dev/null",
        shell=True, capture_output=True, text=True, errors="replace").stdout, flush=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        # Pull the weights into page cache first. /vol is a network volume and
        # llama.cpp mmaps the file: a cold cache makes CPU decode re-fault the
        # whole model every token, which is what turned tg32 3.71 into 0.07 on
        # an otherwise identical build. Warm it and report the rate, so a run
        # that still could not cache the model is identifiable rather than
        # silently reported as a slow kernel.
        w0 = time.time()
        wsz = subprocess.run(f"cat {MODELS[model_key]} > /dev/null; "
                             f"stat -c %s {MODELS[model_key]}",
                             shell=True, capture_output=True, text=True,
                             errors="replace").stdout.strip()
        try:
            gb = int(wsz)/2**30
            print(f"warm {arm}: {gb:.2f} GiB in {time.time()-w0:.0f}s "
                  f"({gb/max(time.time()-w0, 1e-3):.2f} GiB/s)", flush=True)
        except ValueError:
            print(f"warm {arm}: could not stat model ({wsz!r})", flush=True)
        print(subprocess.run("free -g | head -2", shell=True, capture_output=True,
                             text=True, errors="replace").stdout, flush=True)
        # -r 3: in-container repeats, so llama-bench's own stddev separates
        # run-to-run noise from the container-to-container kind
        cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
               f"-ngl 0 -p 64 -n 32 -r 3 -t 16")
        print(f"--- cpucheck arm {arm} (CPU only) ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace", env=env,
                           timeout=3600)
        print((j.stdout + j.stderr)[-1800:], flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def ops_l40s(arms: str, tag: str):
    """Ada (sm_89) lane: op-level gates."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return ops.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def ops(arms: str, tag: str):
    """test-backend-ops for mach1 ops under each arm's env - localizes numeric
    breaks to the op level, no model needed."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target test-backend-ops")
    for arm in [a.strip() for a in arms.split(",")]:
        _, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        for op in ["MACH1_RT_MM", "MACH1_EXP_MM"]:
            j = subprocess.run(f"/work/build/bin/test-backend-ops test -b CUDA0 -o {op}",
                               shell=True, capture_output=True, text=True, env=env, timeout=1800)
            tail = (j.stdout + j.stderr)[-600:]
            ok = "OK" if ("FAIL" not in tail and j.returncode == 0) else "FAIL"
            print(f"ops {arm} {op}: {ok}", flush=True)
            if ok == "FAIL":
                print(tail, flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600)
def batched_l40s(arms: str, tag: str, npl: str = "1,2,4,8,16"):
    """Batch-size axis on Ada (sm_89)."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return batched.local(arms, tag, npl)


@APP.function(image=IMAGE, gpu="L40S", cpu=16, volumes={"/vol": VOL}, timeout=2*3600)
def batched_l40s_c16(arms: str, tag: str, npl: str = "1,2,4,8,16"):
    """Batch axis at the BENCH lane's CPU request. Same code as `batched`,
    different CPU allocation only. Note the allocation also moves the core
    count llama.cpp sees (32 here, 17 on the default request), which is its
    own lane variable - pin it with the npl "@<threads>" suffix."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return batched.local(arms, tag, npl)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def batched(arms: str, tag: str, npl: str = "1,2,4,8,16"):
    """Step-cost scaling vs tokens per step: llama-batched-bench at n_parallel
    1..8 approximates how a speculative verify batch amortizes the codec."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # optional "<npl>@<threads>": pin -t/-tb instead of taking llama.cpp's
    # default. The two batched lanes see different core counts (default 17,
    # cpu=16 32), and at B1/B2 the mach1 graph is launch-bound, so the thread
    # count is a lane variable that has to be controlled before the CPU
    # request can be read as the cause of anything.
    npl, _, n_thr = npl.partition("@")
    thr_flag = f" -t {int(n_thr)} -tb {int(n_thr)}" if n_thr else ""
    tg_b1 = {}
    tg_b16 = {}
    arm_list = [a.strip() for a in arms.split(",")]
    splitk_order = ["p16kc0", "p16ks0", "q4km", "p16ks1", "p16kc1"]
    splitk_factorial = set(splitk_order).issubset(arm_list)
    if splitk_factorial and (arm_list != splitk_order or npl != "1,16"):
        raise RuntimeError(
            "split-K factorial must use exact order p16kc0,p16ks0,q4km,p16ks1,p16kc1 "
            "and npl=1,16"
        )
    for arm in arm_list:
        model_key, extra = ARMS[arm]
        extra = dict(extra)
        ub = extra.pop("BENCH_UB", "512")   # per-arm n_ubatch override
        # per-arm npl override: census arms need one clean batch width per
        # process (the TIME=3 table accumulates for the process lifetime)
        arm_npl = extra.pop("BENCH_NPL", npl)
        # per-arm context override (KV-pressure probes: npl=32 fills c=8192)
        arm_c = extra.pop("BENCH_C", "8192")
        # per-arm thread override, so a thread A/B rides ONE container
        arm_t = extra.pop("BENCH_T", None)
        arm_thr = f" -t {int(arm_t)} -tb {int(arm_t)}" if arm_t else thr_flag
        env = dict(os.environ, **extra)
        cmd = (f"/work/build/bin/llama-batched-bench -m {MODELS[model_key]} "
               f"-fa 1 -c {arm_c} -ub {ub} -npp 128 -ntg 128 -npl {arm_npl} -ngl 999{arm_thr} 2>&1")
        print(f"--- batched arm {arm} ---", flush=True)
        try:
            # a persistent weight bank is a whole-process VRAM cost no
            # llama.cpp counter reports, so sample the device through the run
            j, peak = run_vram(cmd, env, timeout=300 if splitk_factorial else None)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"split-K factorial arm {arm} timed out") from exc
        print(f"    peakvram {peak} MiB", flush=True)
        arm_output = j.stdout + j.stderr
        print(arm_output[-2500:], flush=True)
        if splitk_factorial and j.returncode != 0:
            raise RuntimeError(
                f"split-K factorial arm {arm} failed rc={j.returncode}:\n{arm_output[-4000:]}"
            )
        markers = ARM_ENGAGEMENT_MARKERS.get(arm)
        if markers is not None:
            marker_counts = {marker: arm_output.count(marker) for marker in markers}
            engaged_once = all(count == 1 for count in marker_counts.values())
            receipt = {
                "arm": arm,
                "marker_counts": marker_counts,
                "engaged_once": engaged_once,
            }
            print(f"ENGAGEMENT_RECEIPT {json.dumps(receipt, sort_keys=True)}", flush=True)
            with open(f"{outdir}/engagement_{arm}.json", "w") as f:
                json.dump(receipt, f, indent=2, sort_keys=True)
                f.write("\n")
            if not engaged_once:
                raise RuntimeError(
                    f"{arm} engagement marker counts must all be exactly one; got {marker_counts}"
                )
        eng = [l for l in arm_output.splitlines() if "ENGAGED" in l]
        if eng:
            print("\n".join(eng[:40]), flush=True)
        mt = [l for l in arm_output.splitlines() if "mach1-time" in l]
        if mt:
            print("\n".join(mt[-70:]), flush=True)
        stamps = [l for l in arm_output.splitlines() if "mach1-stamp" in l]
        if stamps:
            print("\n".join(stamps[-90:]), flush=True)
        with open(f"{outdir}/batched_{arm}.txt", "w") as f:
            f.write(j.stdout + j.stderr)
        tg1 = _tg_at_b1(j.stdout + j.stderr)
        if tg1 is not None:
            tg_b1[arm] = tg1
        tg16 = _tg_at_npl(arm_output, 16)
        if tg16 is not None:
            tg_b16[arm] = tg16
        VOL.commit()
        if splitk_factorial and arm == "q4km":
            first_control = tg_b1.get("p16kc0")
            q4_b1 = tg_b1.get("q4km")
            if first_control is None or q4_b1 is None:
                raise RuntimeError(
                    f"split-K factorial early host gate missing B1 rows: {tg_b1}"
                )
            if first_control < 150.0 or q4_b1 > 190.0:
                early = {
                    "stage": "early_host_gate",
                    "control_b1_tokens_per_second": first_control,
                    "control_min": 150.0,
                    "q4km_b1_tokens_per_second": q4_b1,
                    "q4km_max": 190.0,
                    "passed": False,
                    "decision": "discard_and_redraw",
                }
                with open(f"{outdir}/p16_splitk_factorial_b16.json", "w") as handle:
                    json.dump(early, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                VOL.commit()
                raise RuntimeError(
                    "split-K factorial host degraded after first bracket; discard and redraw"
                )

    # HOST GATE. L40S containers degrade, and they degrade m1 FAR harder than
    # q4km - round 316 saw m1 fall 162.8 -> 62.5 (2.6x) on a host where q4km
    # only lost 8%. An A/B taken there is worthless, and three of roughly
    # eight runs in one session landed on such a host. The gate is two-sided
    # because either arm alone is ambiguous: a slow q4km could be a slow host
    # OR a fast m1, and vice versa.
    _host_gate(tg_b1, tg_b16=tg_b16)
    _factorial_receipt(tg_b16, outdir)
    _splitk_factorial_receipt(tg_b1, tg_b16, outdir, required=splitk_factorial)


def _tg_at_b1(out: str):
    """S_TG for the B=1 row of a llama-batched-bench table, or None."""
    return _tg_at_npl(out, 1)


def _tg_at_npl(out: str, npl: int):
    """Aggregate S_TG for one n_parallel row, or None."""
    for line in out.splitlines():
        c = [x.strip() for x in line.split("|")]
        if len(c) >= 11 and c[3] == str(npl):
            try:
                return float(c[8])
            except ValueError:
                continue
    return None


def _factorial_receipt(tg_b16: dict, outdir: str):
    """Report cp/sibling interaction in B16 milliseconds per decode step."""
    names = ("p16f00", "p16f10", "p16f01", "p16f11")
    if not all(name in tg_b16 for name in names):
        return
    # At n_parallel=16, one decode step emits 16 generated tokens.
    step_ms = {name: 16_000.0/tg_b16[name] for name in names}
    cp_offsib = step_ms["p16f00"] - step_ms["p16f10"]
    cp_onsib = step_ms["p16f01"] - step_ms["p16f11"]
    sib_offcp = step_ms["p16f00"] - step_ms["p16f01"]
    sib_oncp = step_ms["p16f10"] - step_ms["p16f11"]
    receipt = {
        "n_parallel": 16,
        "tg_tokens_per_second": {name: tg_b16[name] for name in names},
        "step_ms": step_ms,
        "delta_cp_off_sibling_ms": cp_offsib,
        "delta_cp_on_sibling_ms": cp_onsib,
        "cp_retention": cp_onsib/cp_offsib if cp_offsib != 0.0 else None,
        "delta_sibling_off_cp_ms": sib_offcp,
        "delta_sibling_on_cp_ms": sib_oncp,
        "interaction_ms": cp_onsib - cp_offsib,
    }
    print(f"P16_FACTORIAL_RECEIPT {json.dumps(receipt, sort_keys=True)}", flush=True)
    with open(f"{outdir}/p16_factorial_b16.json", "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    VOL.commit()


def _splitk_factorial_receipt(tg_b1: dict, tg_b16: dict, outdir: str,
                               required: bool = False):
    """Fail-closed C-S-Q-S-C wall receipt for the exact KS4 candidate."""
    import math

    names = ("p16kc0", "p16ks0", "q4km", "p16ks1", "p16kc1")
    missing_b1 = [name for name in names if name not in tg_b1]
    missing_b16 = [name for name in names if name not in tg_b16]
    if missing_b1 or missing_b16:
        if required:
            raise RuntimeError(
                f"split-K factorial missing rows: B1={missing_b1} B16={missing_b16}"
            )
        return

    control_b1 = math.sqrt(tg_b1["p16kc0"] * tg_b1["p16kc1"])
    candidate_b1 = math.sqrt(tg_b1["p16ks0"] * tg_b1["p16ks1"])
    control_b16 = math.sqrt(tg_b16["p16kc0"] * tg_b16["p16kc1"])
    candidate_b16 = math.sqrt(tg_b16["p16ks0"] * tg_b16["p16ks1"])
    q4_b16 = tg_b16["q4km"]
    control_step_ms = 16_000.0 / control_b16
    candidate_step_ms = 16_000.0 / candidate_b16
    net_delta_ms = control_step_ms - candidate_step_ms
    b1_relative_delta = abs(candidate_b1 - control_b1) / control_b1
    host_pass = (tg_b1["q4km"] <= 190.0 and
                 tg_b1["p16kc0"] >= 150.0 and tg_b1["p16kc1"] >= 150.0)
    b1_neutral = b1_relative_delta <= 0.01
    material_win = net_delta_ms >= 0.20
    beats_q4 = candidate_b16 > q4_b16
    passed = host_pass and b1_neutral and material_win and beats_q4
    receipt = {
        "order": list(names),
        "n_parallel": [1, 16],
        "tg_b1_tokens_per_second": {name: tg_b1[name] for name in names},
        "tg_b16_tokens_per_second": {name: tg_b16[name] for name in names},
        "control_b1_geomean": control_b1,
        "candidate_b1_geomean": candidate_b1,
        "b1_relative_delta": b1_relative_delta,
        "b1_relative_delta_limit": 0.01,
        "control_b16_geomean": control_b16,
        "candidate_b16_geomean": candidate_b16,
        "q4km_b16": q4_b16,
        "control_step_ms": control_step_ms,
        "candidate_step_ms": candidate_step_ms,
        "net_delta_ms_per_step": net_delta_ms,
        "net_delta_ms_per_step_min": 0.20,
        "candidate_minus_q4_tokens_per_second": candidate_b16 - q4_b16,
        "host_pass": host_pass,
        "b1_neutral": b1_neutral,
        "material_win": material_win,
        "beats_q4": beats_q4,
        "transient_bytes_delta": 3 * 16 * 2048 * 4,
        "persistent_bytes_delta": 0,
        "passed": passed,
        "decision": "promote" if passed else (
            "discard_and_redraw" if not host_pass else "near_miss_or_retune"
        ),
    }
    print(f"P16_SPLITK_FACTORIAL_RECEIPT {json.dumps(receipt, sort_keys=True)}",
          flush=True)
    with open(f"{outdir}/p16_splitk_factorial_b16.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOL.commit()
    if required and not passed:
        raise RuntimeError(
            "p16 split-K factorial gate failed: "
            f"host={host_pass} b1_neutral={b1_neutral} "
            f"net_delta={net_delta_ms:.4f}ms beats_q4={beats_q4}"
        )


def _host_gate(tg_b1: dict, q_max: float = 190.0, m_min: float = 150.0,
               tg_b16: dict | None = None, m16_min: float = 850.0):
    q = tg_b1.get("q4km")
    ctrl = next((v for k, v in tg_b1.items() if k != "q4km"), None)
    if q is None or ctrl is None:
        print(f"HOSTGATE unknown (need q4km and one mach1 arm at B=1; got {tg_b1})", flush=True)
        return
    # the pool degrades the mach1 arms' INTERIOR while B1 stays healthy
    # (mbase-wall-s1: control B1 163.8 passed, B16 785 vs the 900-983 band).
    # A control B16 floor catches that pathology when the sweep includes B16.
    c16 = None
    if tg_b16 is not None:
        c16 = next((v for k, v in tg_b16.items() if k != "q4km"), None)
    ok = q <= q_max and ctrl >= m_min and (c16 is None or c16 >= m16_min)
    tail = f" control_b16={c16:.1f} (>= {m16_min})" if c16 is not None else ""
    print(f"HOSTGATE {'ok' if ok else 'DEGRADED'} q4km={q:.1f} (<= {q_max}) "
          f"control={ctrl:.1f} (>= {m_min}){tail} -- {'results usable' if ok else 'DISCARD AND REDRAW'}",
          flush=True)


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=1800)
def splitk_time_l40s(tag: str):
    """Fail-closed one-row stage-time gate for the exact KS4 production path."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    command = (
        "/work/build/bin/llama-batched-bench -m {model} -fa 1 -c 8192 "
        "-npp 128 -ntg 128 -npl 16 -ngl 999"
    )
    timed: dict[str, dict[str, dict[str, float | int]]] = {}
    throughput: dict[str, float] = {}
    line_re = re.compile(
        r"^mach1-time:\s+(.*?)\s+calls=\s*(\d+)\s+"
        r"total=\s*([0-9.]+)\s+us\s+avg=\s*([0-9.]+)\s+us\s*$"
    )
    for arm in ("p16combo", "p16ks4"):
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        env.update(GGML_MACH1_TIME="1", GGML_MACH1_DEBUG="1",
                   GGML_CUDA_DISABLE_GRAPHS="1")
        try:
            run = subprocess.run(
                command.format(model=MODELS[model_key]), shell=True,
                capture_output=True, text=True, env=env, timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"split-K TIME arm {arm} timed out") from exc
        log = run.stdout + run.stderr
        with open(f"{outdir}/p16_splitk_time_{arm}.txt", "w") as handle:
            handle.write(log)
        if run.returncode != 0:
            VOL.commit()
            raise RuntimeError(
                f"split-K TIME arm {arm} failed rc={run.returncode}:\n{log[-4000:]}"
            )
        markers = ARM_ENGAGEMENT_MARKERS[arm]
        marker_counts = {marker: log.count(marker) for marker in markers}
        if not all(count == 1 for count in marker_counts.values()):
            raise RuntimeError(
                f"split-K TIME arm {arm} engagement mismatch: {marker_counts}"
            )
        records: dict[str, dict[str, float | int]] = {}
        for line in log.splitlines():
            match = line_re.match(line)
            if match is None:
                continue
            label, calls, total_us, avg_us = match.groups()
            if label in records:
                raise RuntimeError(f"duplicate TIME record for {arm}: {label}")
            records[label] = {
                "calls": int(calls),
                "total_us": float(total_us),
                "avg_us": float(avg_us),
            }
        if not records:
            raise RuntimeError(f"split-K TIME arm {arm} emitted no timed records")
        tg = _tg_at_npl(log, 16)
        if tg is None:
            raise RuntimeError(f"split-K TIME arm {arm} missing B16 throughput row")
        timed[arm] = records
        throughput[arm] = tg
        print(f"SPLITK_TIME_ARM {arm} records={len(records)} B16={tg:.2f}", flush=True)

    control_core = "rt_imma8 m=2048 n=4096 nt=16"
    candidate_core = "rt_imma8_split4 m=2048 n=4096 nt=16"
    control_out = "rt_out_tc m=2048 n=4096 nt=16"
    candidate_out = "rt_out_tc_split4 m=2048 n=4096 nt=16"
    required = {
        ("p16combo", control_core), ("p16combo", control_out),
        ("p16ks4", candidate_core), ("p16ks4", candidate_out),
    }
    missing = [(arm, label) for arm, label in required if label not in timed[arm]]
    forbidden = [
        ("p16combo", candidate_core), ("p16combo", candidate_out),
        ("p16ks4", control_core),
    ]
    present_forbidden = [
        (arm, label) for arm, label in forbidden if label in timed[arm]
    ]
    if missing or present_forbidden:
        raise RuntimeError(
            f"split-K TIME path mismatch: missing={missing} forbidden={present_forbidden}"
        )

    def canonical_counts(arm: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for label, record in timed[arm].items():
            canonical = label
            if canonical.startswith("rt_imma8_split4 "):
                canonical = "rt_imma8 " + canonical.removeprefix("rt_imma8_split4 ")
            if canonical.startswith("rt_out_tc_split4 "):
                canonical = "rt_out_tc " + canonical.removeprefix("rt_out_tc_split4 ")
            if canonical in result:
                raise RuntimeError(f"TIME canonical-label collision for {arm}: {canonical}")
            result[canonical] = int(record["calls"])
        return result

    control_counts = canonical_counts("p16combo")
    candidate_counts = canonical_counts("p16ks4")
    if control_counts != candidate_counts:
        only_control = sorted(set(control_counts.items()) - set(candidate_counts.items()))
        only_candidate = sorted(set(candidate_counts.items()) - set(control_counts.items()))
        raise RuntimeError(
            f"split-K TIME call census changed: control_only={only_control} "
            f"candidate_only={only_candidate}"
        )
    core_calls = int(timed["p16combo"][control_core]["calls"])
    out_calls = int(timed["p16combo"][control_out]["calls"])
    candidate_core_calls = int(timed["p16ks4"][candidate_core]["calls"])
    candidate_out_calls = int(timed["p16ks4"][candidate_out]["calls"])
    if not (core_calls == out_calls == candidate_core_calls == candidate_out_calls and
            core_calls > 0 and core_calls % 40 == 0):
        raise RuntimeError(
            "split-K TIME exact-shape call counts must match and be divisible by 40: "
            f"{core_calls}, {out_calls}, {candidate_core_calls}, {candidate_out_calls}"
        )
    control_core_us = float(timed["p16combo"][control_core]["avg_us"])
    candidate_core_us = float(timed["p16ks4"][candidate_core]["avg_us"])
    control_out_us = float(timed["p16combo"][control_out]["avg_us"])
    candidate_out_us = float(timed["p16ks4"][candidate_out]["avg_us"])
    core_ratio = candidate_core_us / control_core_us
    net_delta_ms_per_step = 40.0 * (
        control_core_us + control_out_us - candidate_core_us - candidate_out_us
    ) / 1000.0
    passed = core_ratio <= 0.84 and net_delta_ms_per_step >= 0.20
    receipt = {
        "shape": {"m": 2048, "n": 4096, "nt": 16},
        "calls_per_model_step": 40,
        "timed_calls": core_calls,
        "measured_steps": core_calls // 40,
        "control_core_avg_us": control_core_us,
        "candidate_core_avg_us": candidate_core_us,
        "control_out_avg_us": control_out_us,
        "candidate_out_avg_us": candidate_out_us,
        "candidate_core_ratio": core_ratio,
        "candidate_core_ratio_limit": 0.84,
        "net_delta_ms_per_step": net_delta_ms_per_step,
        "net_delta_ms_per_step_min": 0.20,
        "b16_tokens_per_second": throughput,
        "transient_bytes_delta": 3 * 16 * 2048 * 4,
        "persistent_bytes_delta": 0,
        "call_census_equal": True,
        "passed": passed,
    }
    with open(f"{outdir}/p16_splitk_time_gate.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOL.commit()
    print(f"P16 SPLITK TIME GATE: {'PASS' if passed else 'FAIL'} "
          f"core_ratio={core_ratio:.4f} net_delta={net_delta_ms_per_step:.4f} ms/step",
          flush=True)
    if not passed:
        raise RuntimeError("p16 split-K TIME gate failed")
    return receipt


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=3600, cpu=16)
def qonce_ipack_probe_l40s(tag: str, args: str = "9 48"):
    """Byte oracle, resource gate, and isolated pricing replica for the inline
    scaled QONCE q8 pack (pocs/mach1-chainbench --qonce-ipack)."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-chainbench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, GGML_MACH1_TIME="1")
    run = subprocess.run(
        f"/work/build/bin/llama-mach1-chainbench --qonce-ipack {args}",
        shell=True, capture_output=True, text=True, env=env, timeout=1800,
    )
    log = run.stdout + run.stderr
    with open(f"{outdir}/qonce_ipack_probe.txt", "w") as handle:
        handle.write(log)
    VOL.commit()
    print(log, flush=True)
    if run.returncode != 0:
        raise RuntimeError(f"qonce-ipack probe failed rc={run.returncode}")
    required = [
        "QONCE_IPACK ENGAGED",
        "QONCE_IPACK_ORACLE",
        "QONCE_IPACK_CPU_ORACLE d=2048",
        "QONCE_IPACK_CPU_ORACLE d=512",
        "QONCE_IPACK_REPEAT",
        "QONCE_IPACK_TIMING",
        "QONCE_IPACK_DECISION",
    ]
    missing = [marker for marker in required if marker not in log]
    fails = [line for line in log.splitlines() if "status=FAIL" in line]
    if missing or fails:
        raise RuntimeError(f"qonce-ipack probe gate: missing={missing} fails={fails}")


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def qonce_ipack_time_l40s(tag: str):
    """Fail-closed exp_mega stage-time pair for the inline scaled QONCE pack
    on the promoted composition parent."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    command = (
        "/work/build/bin/llama-batched-bench -m {model} -fa 1 -c 8192 "
        "-npp 128 -ntg 128 -npl 16 -ngl 999"
    )
    line_re = re.compile(
        r"^mach1-time:\s+(.*?)\s+calls=\s*(\d+)\s+"
        r"total=\s*([0-9.]+)\s+us\s+avg=\s*([0-9.]+)\s+us\s*$"
    )
    res_re = re.compile(
        r"^mach1-time: exp_mega slot=(\d+) wg=(\d+) regs=(\d+) spill=(\d+) B "
        r"smem=(\d+) B (\d+) blk/SM$"
    )
    mega_label = "exp_mega m=512 n=2048 md=2048 n_used=8 nt=16 dnct=128 zdp=1 slot=0"
    timed: dict[str, dict[str, dict[str, float | int]]] = {}
    throughput: dict[str, float] = {}
    resources: dict[str, dict[str, int]] = {}
    for arm in ("p16combo", "p16qfs"):
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        env.update(GGML_MACH1_TIME="1", GGML_MACH1_DEBUG="1",
                   GGML_CUDA_DISABLE_GRAPHS="1")
        try:
            run = subprocess.run(
                command.format(model=MODELS[model_key]), shell=True,
                capture_output=True, text=True, env=env, timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"qonce-ipack TIME arm {arm} timed out") from exc
        log = run.stdout + run.stderr
        with open(f"{outdir}/qonce_ipack_time_{arm}.txt", "w") as handle:
            handle.write(log)
        if run.returncode != 0:
            VOL.commit()
            raise RuntimeError(
                f"qonce-ipack TIME arm {arm} failed rc={run.returncode}:\n{log[-4000:]}"
            )
        markers = ARM_ENGAGEMENT_MARKERS[arm]
        marker_counts = {marker: log.count(marker) for marker in markers}
        if not all(count == 1 for count in marker_counts.values()):
            raise RuntimeError(
                f"qonce-ipack TIME arm {arm} engagement mismatch: {marker_counts}"
            )
        ipack_marker = ARM_ENGAGEMENT_MARKERS["p16qfs"][-1]
        if arm == "p16combo" and ipack_marker in log:
            raise RuntimeError("control arm engaged the candidate pack")
        slot_want = 13 if arm == "p16qfs" else 12
        arm_res = None
        for line in log.splitlines():
            match = res_re.match(line.strip())
            if match is None:
                continue
            slot, wg, regs, spill, smem, bps = (int(g) for g in match.groups())
            if slot == slot_want:
                arm_res = {"slot": slot, "wg": wg, "regs": regs,
                           "spill": spill, "smem": smem, "blocks_per_sm": bps}
        if arm_res is None:
            raise RuntimeError(f"qonce-ipack TIME arm {arm} missing slot={slot_want} resource line")
        if arm_res["regs"] > 64 or arm_res["spill"] != 0:
            raise RuntimeError(f"qonce-ipack TIME arm {arm} resource gate: {arm_res}")
        records: dict[str, dict[str, float | int]] = {}
        for line in log.splitlines():
            match = line_re.match(line)
            if match is None:
                continue
            label, calls, total_us, avg_us = match.groups()
            if label in records:
                raise RuntimeError(f"duplicate TIME record for {arm}: {label}")
            records[label] = {
                "calls": int(calls),
                "total_us": float(total_us),
                "avg_us": float(avg_us),
            }
        if not records:
            raise RuntimeError(f"qonce-ipack TIME arm {arm} emitted no timed records")
        tg = _tg_at_npl(log, 16)
        if tg is None:
            raise RuntimeError(f"qonce-ipack TIME arm {arm} missing B16 throughput row")
        timed[arm] = records
        throughput[arm] = tg
        resources[arm] = arm_res
        print(f"QONCE_IPACK_TIME_ARM {arm} records={len(records)} B16={tg:.2f} "
              f"regs={arm_res['regs']} spill={arm_res['spill']}", flush=True)

    control_label = mega_label + " q1=1"
    candidate_label = mega_label + " q1=2"
    if control_label not in timed["p16combo"]:
        raise RuntimeError("control arm missing the QONCE exp_mega record")
    if candidate_label not in timed["p16qfs"]:
        raise RuntimeError("candidate arm missing the IPACK exp_mega record")
    if candidate_label in timed["p16combo"] or control_label in timed["p16qfs"]:
        raise RuntimeError("qonce-ipack TIME path mismatch: wrong producer engaged")

    def canonical_counts(arm: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for label, record in timed[arm].items():
            canonical = label.replace(" q1=2", " q1=1")
            if canonical in result:
                raise RuntimeError(f"TIME canonical-label collision for {arm}: {canonical}")
            result[canonical] = int(record["calls"])
        return result

    control_counts = canonical_counts("p16combo")
    candidate_counts = canonical_counts("p16qfs")
    if control_counts != candidate_counts:
        only_control = sorted(set(control_counts.items()) - set(candidate_counts.items()))
        only_candidate = sorted(set(candidate_counts.items()) - set(control_counts.items()))
        raise RuntimeError(
            f"qonce-ipack TIME call census changed: control_only={only_control} "
            f"candidate_only={only_candidate}"
        )

    calls = int(timed["p16combo"][control_label]["calls"])
    step_options = [steps for steps in (129, 128, 127) if calls % steps == 0]
    if len(step_options) != 1:
        raise RuntimeError(
            f"qonce-ipack TIME cannot infer decode steps: calls={calls} "
            f"divisible by {step_options}"
        )
    steps = step_options[0]
    layers = calls // steps
    control_avg_us = float(timed["p16combo"][control_label]["avg_us"])
    candidate_avg_us = float(timed["p16qfs"][candidate_label]["avg_us"])
    delta_ms_per_step = layers * (control_avg_us - candidate_avg_us) / 1000.0
    passed = delta_ms_per_step >= 0.150
    receipt = {
        "shape": {"m": 512, "n": 2048, "md": 2048, "n_used": 8, "nt": 16},
        "timed_calls": calls,
        "measured_steps": steps,
        "moe_layers": layers,
        "control_avg_us": control_avg_us,
        "candidate_avg_us": candidate_avg_us,
        "delta_ms_per_step": delta_ms_per_step,
        "delta_ms_per_step_min": 0.150,
        "b16_tokens_per_second": throughput,
        "resources": resources,
        "transient_bytes_delta": 0,
        "persistent_bytes_delta": 0,
        "call_census_equal": True,
        "passed": passed,
    }
    with open(f"{outdir}/qonce_ipack_time_gate.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOL.commit()
    print(f"QONCE IPACK TIME GATE: {'PASS' if passed else 'FAIL'} "
          f"layers={layers} control={control_avg_us:.2f}us candidate={candidate_avg_us:.2f}us "
          f"delta={delta_ms_per_step:.4f} ms/step", flush=True)
    if not passed:
        raise RuntimeError("qonce-ipack TIME gate failed")
    return receipt


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=3600, cpu=16)
def imma8_pipe_l40s(tag: str, args: str = "9"):
    """Pipeline-depth pricing for the imma8-family bandwidth ceiling
    (pocs/mach1-chainbench --imma8-pipe)."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-mach1-chainbench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, GGML_MACH1_TIME="1")
    run = subprocess.run(
        f"/work/build/bin/llama-mach1-chainbench --imma8-pipe {args}",
        shell=True, capture_output=True, text=True, env=env, timeout=1800,
    )
    log = run.stdout + run.stderr
    with open(f"{outdir}/imma8_pipe.txt", "w") as handle:
        handle.write(log)
    VOL.commit()
    print(log, flush=True)
    if run.returncode != 0:
        raise RuntimeError(f"imma8-pipe probe failed rc={run.returncode}")
    required = ["IMMA8_PIPE ENGAGED", "IMMA8_PIPE_BITWISE", "IMMA8_PIPE_CPU_ORACLE",
                "IMMA8_PIPE_TIMING", "IMMA8_PIPE_DECISION"]
    missing = [marker for marker in required if marker not in log]
    fails = [line for line in log.splitlines() if "status=FAIL" in line]
    if missing or fails:
        raise RuntimeError(f"imma8-pipe gate: missing={missing} fails={fails}")


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def megarace_l40s(tag: str, arms: str = "p16ks4,p16mp1"):
    """Exploration stage-time race at B16: per-arm exp_mega record, ranked
    nt=16 stage keys, and the instrumented throughput row. No promotion
    gates - receipts only. Probe arms produce garbage output by design."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    command = (
        "/work/build/bin/llama-batched-bench -m {model} -fa 1 -c 8192 "
        "-npp 128 -ntg 128 -npl 16 -ngl 999"
    )
    line_re = re.compile(
        r"^mach1-time:\s+(.*?)\s+calls=\s*(\d+)\s+"
        r"total=\s*([0-9.]+)\s+us\s+avg=\s*([0-9.]+)\s+us\s*$"
    )
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        env.setdefault("GGML_MACH1_TIME", "1")
        env.update(GGML_MACH1_DEBUG="1", GGML_CUDA_DISABLE_GRAPHS="1")
        try:
            run = subprocess.run(
                command.format(model=MODELS[model_key]), shell=True,
                capture_output=True, text=True, env=env, timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"MEGARACE_ARM {arm} TIMEOUT", flush=True)
            continue
        log = run.stdout + run.stderr
        with open(f"{outdir}/megarace_{arm}.txt", "w") as handle:
            handle.write(log)
        if run.returncode != 0:
            print(f"MEGARACE_ARM {arm} rc={run.returncode}\n{log[-2000:]}", flush=True)
            continue
        rows = []
        for line in log.splitlines():
            match = line_re.match(line)
            if match is None:
                continue
            label, calls, total_us, _ = match.groups()
            if " nt=16" in label:
                rows.append((float(total_us)/129.0, int(calls)//129, label))
        rows.sort(reverse=True)
        tg = _tg_at_npl(log, 16)
        print(f"MEGARACE_ARM {arm} B16={tg} nt16_total_ms_per_step="
              f"{sum(r[0] for r in rows)/1000.0:.4f}", flush=True)
        for us, calls, label in rows[:12]:
            print(f"MEGARACE_KEY arm={arm} us_per_step={us:.2f} calls={calls} {label!r}",
                  flush=True)
    VOL.commit()


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def stdout_time_l40s(tag: str):
    """Fail-closed stage-time pair for the standard-output row-scale epilogues
    (spine + expert mega) on the split-K composition parent. TIMING_ONLY:
    candidate outputs are wrong-basis garbage by design."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    command = (
        "/work/build/bin/llama-batched-bench -m {model} -fa 1 -c 8192 "
        "-npp 128 -ntg 128 -npl 16 -ngl 999"
    )
    line_re = re.compile(
        r"^mach1-time:\s+(.*?)\s+calls=\s*(\d+)\s+"
        r"total=\s*([0-9.]+)\s+us\s+avg=\s*([0-9.]+)\s+us\s*$"
    )
    res_re = re.compile(
        r"^mach1-time: exp_mega slot=(\d+) wg=(\d+) regs=(\d+) spill=(\d+) B "
        r"smem=(\d+) B (\d+) blk/SM$"
    )
    mega_label = "exp_mega m=512 n=2048 md=2048 n_used=8 nt=16 dnct=128 zdp=1 slot=0 q1=1"
    timed: dict[str, dict[str, dict[str, float | int]]] = {}
    throughput: dict[str, float] = {}
    resources: dict[str, dict[str, int]] = {}
    for arm in ("p16ks4", "p16std"):
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        env.update(GGML_MACH1_TIME="1", GGML_MACH1_DEBUG="1",
                   GGML_CUDA_DISABLE_GRAPHS="1")
        try:
            run = subprocess.run(
                command.format(model=MODELS[model_key]), shell=True,
                capture_output=True, text=True, env=env, timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"stdout TIME arm {arm} timed out") from exc
        log = run.stdout + run.stderr
        with open(f"{outdir}/stdout_time_{arm}.txt", "w") as handle:
            handle.write(log)
        if run.returncode != 0:
            VOL.commit()
            raise RuntimeError(
                f"stdout TIME arm {arm} failed rc={run.returncode}:\n{log[-4000:]}"
            )
        markers = ARM_ENGAGEMENT_MARKERS[arm]
        marker_counts = {marker: log.count(marker) for marker in markers}
        if not all(count == 1 for count in marker_counts.values()):
            raise RuntimeError(
                f"stdout TIME arm {arm} engagement mismatch: {marker_counts}"
            )
        stdout_markers = ARM_ENGAGEMENT_MARKERS["p16std"][-2:]
        if arm == "p16ks4" and any(marker in log for marker in stdout_markers):
            raise RuntimeError("control arm engaged the standard-output path")
        slot_want = 14 if arm == "p16std" else 12
        arm_res = None
        for line in log.splitlines():
            match = res_re.match(line.strip())
            if match is None:
                continue
            slot, wg, regs, spill, smem, bps = (int(g) for g in match.groups())
            if slot == slot_want:
                arm_res = {"slot": slot, "wg": wg, "regs": regs,
                           "spill": spill, "smem": smem, "blocks_per_sm": bps}
        if arm_res is None:
            raise RuntimeError(f"stdout TIME arm {arm} missing slot={slot_want} resource line")
        if arm_res["regs"] > 64 or arm_res["spill"] != 0:
            raise RuntimeError(f"stdout TIME arm {arm} resource gate: {arm_res}")
        records: dict[str, dict[str, float | int]] = {}
        for line in log.splitlines():
            match = line_re.match(line)
            if match is None:
                continue
            label, calls, total_us, avg_us = match.groups()
            if label in records:
                raise RuntimeError(f"duplicate TIME record for {arm}: {label}")
            records[label] = {
                "calls": int(calls),
                "total_us": float(total_us),
                "avg_us": float(avg_us),
            }
        if not records:
            raise RuntimeError(f"stdout TIME arm {arm} emitted no timed records")
        tg = _tg_at_npl(log, 16)
        if tg is None:
            raise RuntimeError(f"stdout TIME arm {arm} missing B16 throughput row")
        timed[arm] = records
        throughput[arm] = tg
        resources[arm] = arm_res
        print(f"STDOUT_TIME_ARM {arm} records={len(records)} B16={tg:.2f} "
              f"mega_regs={arm_res['regs']} mega_spill={arm_res['spill']}", flush=True)

    def canonical(label: str) -> str:
        out = label
        if out.startswith("rt_out_rs_sum4 "):
            out = "rt_out_tc_split4 " + out.removeprefix("rt_out_rs_sum4 ")
        elif out.startswith("rt_out_rs_plain "):
            out = "rt_out " + out.removeprefix("rt_out_rs_plain ")
        elif out.startswith("rt_out_rs "):
            out = "rt_out_tc " + out.removeprefix("rt_out_rs ")
        # sibling-batched epilogues, including the zout form
        out = out.replace("out_rsb", "out_tcb")
        return out

    def canonical_counts(arm: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for label, record in timed[arm].items():
            key = canonical(label)
            if key in result:
                raise RuntimeError(f"TIME canonical-label collision for {arm}: {key}")
            result[key] = int(record["calls"])
        return result

    control_counts = canonical_counts("p16ks4")
    candidate_counts = canonical_counts("p16std")
    if control_counts != candidate_counts:
        only_control = sorted(set(control_counts.items()) - set(candidate_counts.items()))
        only_candidate = sorted(set(candidate_counts.items()) - set(control_counts.items()))
        raise RuntimeError(
            f"stdout TIME call census changed: control_only={only_control} "
            f"candidate_only={only_candidate}"
        )
    if mega_label not in timed["p16ks4"] or mega_label not in timed["p16std"]:
        raise RuntimeError("stdout TIME missing the QONCE exp_mega record")

    calls = int(timed["p16ks4"][mega_label]["calls"])
    step_options = [steps for steps in (129, 128, 127) if calls % steps == 0]
    if len(step_options) != 1:
        raise RuntimeError(
            f"stdout TIME cannot infer decode steps: calls={calls} "
            f"divisible by {step_options}"
        )
    steps = step_options[0]
    changed: dict[str, dict[str, float]] = {}
    cand_by_canonical = {canonical(label): record for label, record in timed["p16std"].items()}
    delta_ms_per_step = 0.0
    for label, record in timed["p16ks4"].items():
        cand = cand_by_canonical[canonical(label)]
        moved = (label.startswith("rt_out") or "out_tcb" in label or
                 label.startswith("exp_mega ")) and " nt=16" in label
        if not moved:
            continue
        delta_us = float(record["total_us"]) - float(cand["total_us"])
        changed[label] = {
            "control_total_us": float(record["total_us"]),
            "candidate_total_us": float(cand["total_us"]),
            "delta_us_per_step": delta_us/steps,
        }
        delta_ms_per_step += delta_us/steps/1000.0
    passed = delta_ms_per_step >= 0.400
    receipt = {
        "timing_only": True,
        "wrong_basis_candidate": True,
        "measured_steps": steps,
        "b16_tokens_per_second": throughput,
        "mega_resources": resources,
        "changed_keys": changed,
        "composed_delta_ms_per_step": delta_ms_per_step,
        "projection_ms_per_step": 0.782927,
        "materiality_ms_per_step_min": 0.400,
        "persistent_bytes_delta": 0,
        "passed": passed,
    }
    with open(f"{outdir}/stdout_time_gate.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    VOL.commit()
    for label, row in sorted(changed.items()):
        print(f"STDOUT_TIME_KEY {label!r} ctl={row['control_total_us']:.0f}us "
              f"cand={row['candidate_total_us']:.0f}us "
              f"delta={row['delta_us_per_step']:.2f}us/step", flush=True)
    print(f"STDOUT TIME GATE: {'PASS' if passed else 'FAIL'} "
          f"composed_delta={delta_ms_per_step:.4f} ms/step "
          f"projection=0.7829 min=0.400", flush=True)
    if not passed:
        raise RuntimeError("stdout TIME gate failed")
    return receipt


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def san_l40s(arms: str, tag: str, npl: str = "2"):
    """memcheck lane on Ada (sm_89)."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return san.local(arms, tag, npl)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def san(arms: str, tag: str, npl: str = "2"):
    """Fault localization for nt>1: a short batched-bench run per arm, first
    plain with CUDA graphs off (splits kernel-bug from graph interaction),
    then under compute-sanitizer memcheck to name the faulting kernel and
    address class. Short shapes - the sanitizer serializes everything."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_CUDA_FLAGS=-lineinfo "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched-bench")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    sanbin = subprocess.run("which compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer 2>/dev/null | head -1",
                            shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    sanbin = sanbin[0] if sanbin else ""
    print(f"sanitizer: {sanbin or 'NOT FOUND'}", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # npp 128 so the dense prefill path (P >= 1024) is inside the memcheck net
    base = (f"/work/build/bin/llama-batched-bench -m {{model}} "
            f"-fa 1 -c 4096 -npp 128 -ntg 8 -npl {npl} -ngl 999")
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra, GGML_CUDA_DISABLE_GRAPHS="1")
        cmd = base.format(model=MODELS[model_key])
        j = subprocess.run(cmd + " 2>&1", shell=True, capture_output=True, text=True, env=env)
        plain = j.stdout + j.stderr
        crashed = "CUDA error" in plain or j.returncode != 0
        print(f"--- san {arm} plain graphs-off: {'CRASH' if crashed else 'CLEAN'} (rc={j.returncode}) ---", flush=True)
        print(plain[-700:], flush=True)
        dbg = [l for l in plain.splitlines() if "mach1-nt" in l]
        if dbg:
            print("\n".join(dbg[:44]), flush=True)
        with open(f"{outdir}/san_{arm}_plain.txt", "w") as f:
            f.write(plain)
        if sanbin:
            # timeout guard: a coop launch under instrumentation can lose its
            # residency promise and spin (the r241 hang class)
            j = subprocess.run(f"timeout -k 30 1500 {sanbin} --tool memcheck --error-exitcode 9 {cmd} 2>&1",
                               shell=True, capture_output=True, text=True, env=env)
            outp = j.stdout + j.stderr
            with open(f"{outdir}/san_{arm}_memcheck.txt", "w") as f:
                f.write(outp)
            recs = [l for l in outp.splitlines()
                    if re.search(r"={5,}|Invalid|out of bounds|Address .* is|mach1|ERROR SUMMARY", l)]
            print(f"--- san {arm} memcheck (rc={j.returncode}) ---", flush=True)
            print("\n".join(recs[:60]), flush=True)
        VOL.commit()


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def optime_l40s(arms: str, tag: str):
    """Ada (sm_89) lane: per-OP stream occupancy."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return optime.local(arms, tag)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def optime(arms: str, tag: str):
    """GGML_MACH1_TIME=3: CUDA event pairs recorded in stream order and drained
    once per graph, so each number is true GPU occupancy for that op - not
    wall clock around a sync. Keyed by ggml op NAME, which is what makes it the
    one decomposition directly comparable between the mach1 model and a stock
    quant (ours are MACH1_*, theirs MUL_MAT/MUL_MAT_ID/FLASH_ATTN_EXT).
    Needs graphs off, so the TOTAL is not shipped throughput - the SHARES are
    the deliverable."""
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-bench")
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        cmd = (f"/work/build/bin/llama-bench -m {MODELS[model_key]} "
               f"-fa 1 -p 16 -n 120 -r 1")
        print(f"--- optime arm {arm} ({model_key}) ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           errors="replace", env=env, timeout=3600)
        rows = [l for l in (j.stdout + j.stderr).splitlines() if "cuda-op-time" in l]
        print("\n".join(rows), flush=True)
        tps = [l for l in (j.stdout + j.stderr).splitlines() if "tg128" in l or "tg" in l and "|" in l]
        print("\n".join(tps[-3:]), flush=True)
        with open(f"{outdir}/optime_{arm}.txt", "w") as f:
            f.write(j.stdout + j.stderr)
        VOL.commit()


@APP.function(image=IMAGE, gpu="L40S", volumes={"/vol": VOL}, timeout=2*3600, cpu=16)
def ntcheck_l40s(arms: str, tag: str, nps: str = "1,2,4"):
    """nt correctness probe on Ada (sm_89)."""
    os.environ["MACH1_CUDA_ARCH"] = "89"
    return ntcheck.local(arms, tag, nps)


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=2*3600)
def ntcheck(arms: str, tag: str, nps: str = "1,2,4"):
    """nt>1 correctness probe: llama-batched, one shared prompt, temp 0. Every
    parallel continuation is the same token problem, so within a batch the
    streams must agree exactly - cross-token contamination in an nt>1 kernel
    shows up as divergent or garbage streams. Across arms the text is a
    tolerance-class comparison (reduction orders differ), not a gate."""
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-batched")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # long enough that prefill runs the dense expert path (P >= 1024 needs a
    # 128-token ubatch); the continuation still starts from the same clause
    prompt = ("In the framework of general relativity, the presence of mass and energy "
              "curves spacetime, and freely falling objects move along the geodesics of "
              "that curved geometry. ")*7 + "The theory of general relativity says that"
    arm_list = [a.strip() for a in arms.split(",")]
    np_list = [int(v) for v in nps.split(",")]
    generated_hashes: dict[tuple[str, int], str] = {}
    generated_counts: dict[tuple[str, int], int] = {}
    continuation_bytes: dict[tuple[str, int], int] = {}
    engagement_receipts: dict[str, dict[str, object]] = {}
    exact_pairs = []
    if {"p16f00", "p16f11"}.issubset(arm_list) and 16 in np_list:
        exact_pairs.append(("compose", "p16f00", "p16f11",
                            "p16_compose_full_continuation_np16.json"))
    if {"p16combo", "p16ks4"}.issubset(arm_list) and 16 in np_list:
        exact_pairs.append(("splitk", "p16combo", "p16ks4",
                            "p16_splitk_full_continuation_np16.json"))
    if {"p16ks4", "p16qace"}.issubset(arm_list) and "p16combo" not in arm_list and 16 in np_list:
        exact_pairs.append(("qace", "p16ks4", "p16qace",
                            "p16_qace_full_continuation_np16.json"))
    if {"p16qace", "p16sgf"}.issubset(arm_list) and "p16ks4" not in arm_list and 16 in np_list:
        exact_pairs.append(("sgf", "p16qace", "p16sgf",
                            "p16_sgf_full_continuation_np16.json"))
    if {"p16qace", "p16goal"}.issubset(arm_list) and 16 in np_list:
        exact_pairs.append(("goal", "p16qace", "p16goal",
                            "p16_goal_full_continuation_np16.json"))
    if {"p16qace", "p16goal3"}.issubset(arm_list) and 16 in np_list:
        exact_pairs.append(("goal3", "p16qace", "p16goal3",
                            "p16_goal3_full_continuation_np16.json"))
    if {"p16qace", "p16goal2"}.issubset(arm_list) and 16 in np_list:
        exact_pairs.append(("goal2", "p16qace", "p16goal2",
                            "p16_goal2_full_continuation_np16.json"))
    if len(exact_pairs) > 1:
        raise RuntimeError(f"ntcheck accepts one exact np16 pair at a time: {exact_pairs}")
    exact_pair = exact_pairs[0] if exact_pairs else None
    exact_np16_arms = set(exact_pair[1:3]) if exact_pair is not None else set()
    exact_n_predict = 272
    exact_min_generated_per_stream = 32
    for arm in arm_list:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        for np_ in np_list:
            # np1 is the DECODE shape (nt == 1) and the prompt alone is ~239
            # tokens, so the shared 240 budget left it generating ONE token
            n_predict = (exact_n_predict
                         if arm in exact_np16_arms and np_ == 16
                         else (480 if np_ == 1 else 240))
            cmd = (f"/work/build/bin/llama-batched -m {MODELS[model_key]} "
                   f"-p '{prompt}' -np {np_} -n {n_predict} --temp 0 "
                   f"-fa 1 -c 8192 -ngl 999 -kvu 2>&1")
            try:
                j = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   env=env, timeout=180)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{arm} np{np_} correctness run timed out") from exc
            if j.returncode != 0:
                raise RuntimeError(
                    f"{arm} np{np_} correctness run failed rc={j.returncode}:\n"
                    f"{(j.stdout + j.stderr)[-4000:]}"
                )
            outp = j.stdout + j.stderr
            with open(f"{outdir}/ntcheck_{arm}_np{np_}.txt", "w") as f:
                f.write(outp)
            if np_ == 1:
                print(f"--- ntcheck {arm} np=1 tail ---", flush=True)
                print(outp[-500:], flush=True)
                # np1 IS the decode shape (nt == 1). llama-batched prints the
                # single stream inline instead of the "sequence i:" block, so
                # hash what follows the echoed prompt up to the timing line.
                pos = outp.rfind(prompt)
                end = outp.find("decoded ", pos if pos >= 0 else 0)
                if pos >= 0 and end > pos:
                    # the generated tokens are printed with no newline between
                    # them, so the continuation is the first line left after
                    # the interleaved mach1 debug lines are dropped
                    seg = re.sub(r"mach1: [^\n]*\n?", "",
                                 outp[pos + len(prompt):end])
                    cont = seg.split("\n")[0].strip().encode()
                    generated_hashes[(arm, 1)] = hashlib.sha256(cont).hexdigest()
                    continuation_bytes[(arm, 1)] = len(cont)
                    print(f"np1 continuation sha256: {generated_hashes[(arm, 1)][:12]} "
                          f"({len(cont)} bytes)", flush=True)
                else:
                    print("np1 continuation: NOT FOUND (no hash)", flush=True)
            else:
                seqs = re.findall(r"sequence \d+:\n\n(.*?)\n\n", outp, re.S)
                agree = len(seqs) == np_ and len(set(seqs)) == 1
                print(f"--- ntcheck {arm} np={np_}: {len(seqs)} streams, "
                      f"intra-batch {'AGREE' if agree else 'DIVERGE'} ---", flush=True)
                if arm in exact_np16_arms and np_ == 16 and not agree:
                    raise RuntimeError(f"{arm} np16 continuations diverged")
                # the CONTINUATION is the evidence - the stream opens with the
                # prompt, and a head-print of it certifies nothing (the r267
                # lesson: three rounds of "identical text" were prompt prefix)
                for k, s in enumerate(seqs[:2]):
                    print(f"[{k}] tail: ...{s[-200:]}", flush=True)
                if seqs:
                    stream_hash = hashlib.sha1(seqs[0].encode()).hexdigest()
                    print(f"stream sha: {stream_hash[:12]}", flush=True)
                    if arm in exact_np16_arms and np_ == 16:
                        decoded = re.findall(r"decoded (\d+) tokens", outp)
                        if not decoded:
                            raise RuntimeError(f"{arm} np{np_} missing decoded-token receipt")
                        total_generated = int(decoded[-1])
                        if total_generated % np_ != 0:
                            raise RuntimeError(
                                f"{arm} np{np_} decoded count {total_generated} is not divisible")
                        generated_per_stream = total_generated // np_
                        if generated_per_stream < exact_min_generated_per_stream:
                            raise RuntimeError(
                                f"{arm} np{np_} generated only {generated_per_stream} tokens/stream; "
                                f"need {exact_min_generated_per_stream}")
                        prompt_pos = seqs[0].find(prompt)
                        if prompt_pos < 0:
                            raise RuntimeError(
                                f"{arm} np{np_} output did not contain the exact prompt")
                        continuation = seqs[0][prompt_pos + len(prompt):].encode()
                        generated_hash = hashlib.sha256(continuation).hexdigest()
                        generated_hashes[(arm, np_)] = generated_hash
                        generated_counts[(arm, np_)] = generated_per_stream
                        continuation_bytes[(arm, np_)] = len(continuation)
                        print(f"full-continuation sha256: {generated_hash[:12]} "
                              f"({generated_per_stream} tokens/stream)", flush=True)
                if arm in exact_np16_arms and np_ == 16:
                    markers = ARM_ENGAGEMENT_MARKERS[arm]
                    marker_counts = {marker: outp.count(marker) for marker in markers}
                    engaged_once = all(count == 1 for count in marker_counts.values())
                    engagement_receipt = {
                        "arm": arm,
                        "marker_counts": marker_counts,
                        "engaged_once": engaged_once,
                    }
                    engagement_receipts[arm] = engagement_receipt
                    print(f"NP16_ENGAGEMENT_RECEIPT "
                          f"{json.dumps(engagement_receipt, sort_keys=True)}",
                          flush=True)
                    if not engaged_once:
                        raise RuntimeError(
                            f"{arm} np16 engagement marker counts must all be exactly one; "
                            f"got {marker_counts}")
        VOL.commit()

    np1 = {a: h for (a, n), h in generated_hashes.items() if n == 1}
    if np1:
        ref_arm = arm_list[0]
        ref = np1.get(ref_arm)
        print("=== np1 (nt == 1 decode) continuation hashes ===", flush=True)
        for arm in arm_list:
            h = np1.get(arm)
            if h is None:
                continue
            verdict = "MATCH" if ref is not None and h == ref else "DIFFERS"
            print(f"np1 {arm:12s} {h[:12]} {verdict} (vs {ref_arm}), "
                  f"{continuation_bytes[(arm, 1)]} bytes", flush=True)

    if exact_pair is not None:
        pair_label, control_arm, candidate_arm, receipt_name = exact_pair
        ref_hash = generated_hashes.get((control_arm, 16))
        candidate_hash = generated_hashes.get((candidate_arm, 16))
        passed = (ref_hash is not None and candidate_hash is not None and
                  ref_hash == candidate_hash)
        receipt = {
            "pair": pair_label,
            "control_arm": control_arm,
            "candidate_arm": candidate_arm,
            "n_parallel": 16,
            "min_generated_tokens_per_stream": exact_min_generated_per_stream,
            "control_generated_tokens_per_stream": generated_counts.get((control_arm, 16)),
            "candidate_generated_tokens_per_stream": generated_counts.get((candidate_arm, 16)),
            "control_continuation_bytes": continuation_bytes.get((control_arm, 16)),
            "candidate_continuation_bytes": continuation_bytes.get((candidate_arm, 16)),
            "control_full_continuation_sha256": ref_hash,
            "candidate_full_continuation_sha256": candidate_hash,
            "engagement_receipts": engagement_receipts,
            "passed": passed,
        }
        print(f"P16 {pair_label.upper()} NP16 BITWISE FULL-CONTINUATION HASH GATE: "
              f"{'PASS' if passed else 'FAIL'} "
              f"control={ref_hash[:12] if ref_hash else 'missing'} "
              f"candidate={candidate_hash[:12] if candidate_hash else 'missing'}", flush=True)
        with open(f"{outdir}/{receipt_name}", "w") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write("\n")
        VOL.commit()
        if not passed:
            raise RuntimeError(
                f"p16 {pair_label} np16 bitwise full-continuation hash gate failed"
            )


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=3600, cpu=32)
def ptxv(arch: str = "89"):
    """CPU-only: compile mach1.cu with -Xptxas -v and report per-kernel
    registers/spill/smem - the occupancy inputs the fleet's dead profilers
    would normally provide."""
    os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs"
    sh("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1")
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DLLAMA_CURL=OFF "
       "-DCMAKE_CUDA_FLAGS='-Xptxas -v'")
    b = subprocess.run("cmake --build /work/build -j --target ggml-cuda",
                       shell=True, capture_output=True, text=True)
    if b.returncode != 0:
        print("PTXV BUILD FAILED", flush=True)
        print((b.stdout + b.stderr)[-6000:], flush=True)
        raise RuntimeError("ptxv build failed")
    keep = [l for l in (b.stdout + b.stderr).splitlines()
            if "Function properties" in l or "registers" in l or "spill" in l or "mach1" in l]
    print("\n".join(keep[-400:]), flush=True)


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=3600, cpu=32)
def p16_compose_ptxgate(arch: str = "89", tag: str = "p16-compose-ptxgate"):
    """Compile mach1.cu once and hard-gate every new composition kernel.

    The two mangled-template needles distinguish the q8 batch instantiations
    from the already validated fp16/default siblings in the same translation
    unit. The split-K candidate is a separate exact-shape entry, so this gate
    also proves the original KS1 kernel remains present and independently
    launchable. The wall is allowed only when every entry has no stack or
    spills and remains under its launchable register ceiling.
    """
    specs = [
        {
            "name": "imma8_cpasync",
            "needles": ("mach1_rt_spine_imma8_cpasync_p16_kernel",),
            "max_registers": 84,
            "dynamic_shared_bytes": 29 * 512,
        },
        {
            "name": "batched_u_q8",
            "needles": ("mach1_rt_u_tc_batch_kernel", "ILi512ELb0ELb1EE"),
            "max_registers": 128,
            "dynamic_shared_bytes": 20_992,
        },
        {
            "name": "mixed_u_q8",
            "needles": ("mach1_rt_u_tc_mixed_p16_kernel", "ILi512ELb1EE"),
            "max_registers": 128,
            "dynamic_shared_bytes": 20_992,
        },
        {
            "name": "out_tc_ks1_m2048",
            "needles": ("mach1_rt_out_tc_kernel", "ILi32ELi64EE"),
            # Frozen sm_89/CUDA-12.8 receipt from the unchanged SUMS=1 path.
            "expected_registers": 40,
            "max_registers": 40,
            "dynamic_shared_bytes": 20_992,
        },
        {
            "name": "imma8_split4",
            "needles": ("mach1_rt_spine_imma8_cpasync_split4_p16_kernel",),
            "max_registers": 85,
            "dynamic_shared_bytes": 29 * 512,
        },
        {
            "name": "out_tc_sum4",
            "needles": ("mach1_rt_out_tc_sum4_kernel",),
            "max_registers": 64,
            "dynamic_shared_bytes": 20_992,
        },
    ]
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    mapped = subprocess.run(
        ["python3", "/work/src/benches/splitk_imma_reference.py",
         "--source", "/work/src/ggml/src/ggml-cuda/mach1.cu"],
        cwd="/work/src/benches", capture_output=True, text=True, timeout=30,
    )
    if mapped.returncode != 0:
        raise RuntimeError(
            "split-K mapping/source gate failed:\n" + mapped.stdout + mapped.stderr
        )
    try:
        mapping_receipt = json.loads(mapped.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"split-K mapping gate emitted invalid JSON: {mapped.stdout}") from exc
    if mapping_receipt.get("status") != "PASS":
        raise RuntimeError(f"split-K mapping gate did not pass: {mapping_receipt}")
    print(f"P16 SPLITK MAPPING GATE: PASS "
          f"{json.dumps(mapping_receipt, sort_keys=True)}", flush=True)
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DLLAMA_CURL=OFF "
       "-DCMAKE_CUDA_FLAGS=-Xptxas=-v")

    listed = subprocess.run(["ninja", "-C", "/work/build", "-t", "targets", "all"],
                            capture_output=True, text=True)
    if listed.returncode != 0:
        raise RuntimeError(f"ninja target listing failed: {listed.stderr[-2000:]}")
    object_targets = [line.split(": ", 1)[0] for line in listed.stdout.splitlines()
                      if line.split(": ", 1)[0].endswith("mach1.cu.o")]
    if len(object_targets) != 1:
        raise RuntimeError(f"expected one mach1.cu object target, got {object_targets}")

    target = object_targets[0]
    print(f"+ ninja -C /work/build -j 1 {target}", flush=True)
    built = subprocess.run(["ninja", "-C", "/work/build", "-j", "1", target],
                           capture_output=True, text=True)
    build_log = built.stdout + built.stderr
    if built.returncode != 0:
        print(build_log[-8000:], flush=True)
        raise RuntimeError("p16 composition ptxas build failed")

    lines = build_log.splitlines()
    receipts = []
    for spec in specs:
        starts = [i for i, line in enumerate(lines)
                  if "Compiling entry function" in line and
                  all(needle in line for needle in spec["needles"])]
        if len(starts) != 1:
            matches = [line for line in lines
                       if all(needle in line for needle in spec["needles"])]
            raise RuntimeError(
                f"expected one ptxas entry for {spec['name']}, starts={len(starts)}, "
                f"matches={matches[-10:]}"
            )
        start = starts[0]
        end = next((i for i in range(start + 1, len(lines))
                    if "Compiling entry function" in lines[i]), len(lines))
        record = "\n".join(lines[start:end])
        frame = re.search(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
                          r"(\d+) bytes spill loads", record)
        regs = re.search(r"Used\s+(\d+) registers", record)
        if frame is None or regs is None:
            raise RuntimeError(f"incomplete ptxas record for {spec['name']}:\n{record}")
        stack_bytes, spill_stores, spill_loads = (int(v) for v in frame.groups())
        registers = int(regs.group(1))
        register_exact = (spec.get("expected_registers") is None or
                          registers == spec["expected_registers"])
        passed = (stack_bytes == 0 and spill_stores == 0 and spill_loads == 0 and
                  registers <= spec["max_registers"] and register_exact)
        receipt = {
            "name": spec["name"],
            "needles": list(spec["needles"]),
            "registers": registers,
            "expected_registers": spec.get("expected_registers"),
            "max_registers": spec["max_registers"],
            "dynamic_shared_bytes": spec["dynamic_shared_bytes"],
            "stack_bytes": stack_bytes,
            "spill_store_bytes": spill_stores,
            "spill_load_bytes": spill_loads,
            "passed": passed,
            "ptxas_record": record,
        }
        receipts.append(receipt)
        print(record, flush=True)
        print(f"{spec['name']} PTXAS: {'PASS' if passed else 'FAIL'} "
              f"regs={registers}/{spec['max_registers']} stack={stack_bytes} "
              f"spill_store={spill_stores} spill_load={spill_loads}", flush=True)

    result = {
        "arch": arch,
        "object_target": target,
        "mapping": mapping_receipt,
        "passed": all(receipt["passed"] for receipt in receipts),
        "kernels": receipts,
    }
    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/p16_compose_ptxgate.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    VOL.commit()
    print(f"P16 COMPOSE PTXAS GATE: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    if not result["passed"]:
        raise RuntimeError("p16 composition ptxas resource gate failed; do not run wall")
    return result


@APP.function(image=IMAGE, volumes={"/vol": VOL}, timeout=3600, cpu=32)
def build_only(targets: str = "llama-bench llama-cli llama-perplexity llama-quantize"):
    """Compile check without a GPU allocation (nvcc works fine CPU-only)."""
    t0 = time.time()
    # no driver on a CPU container: link against the toolkit's libcuda stub.
    # LIBRARY_PATH resolves -lcuda; the exe links also need the stub's SONAME
    # (libcuda.so.1) findable by ld for DT_NEEDED resolution, which the stubs
    # dir does not provide - alias it into the default search path.
    os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs"
    sh("ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1")
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    sh("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
       f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF "
       "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
       "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh(f"cmake --build /work/build -j --target {targets}")
    print(f"build ok in {time.time()-t0:.0f}s", flush=True)


WIKI_URL = ("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip")
CORPUS = "/vol/bench/wiki.test.trunc.raw"
BASE_LOGITS = "/vol/bench/teacher-wiki32k.kld"


@APP.function(image=IMAGE, gpu="H200", volumes={"/vol": VOL}, timeout=4*3600)
def gates(arms: str, tag: str, n_chunks: int = 64, ub: int = 1):
    """KL-divergence + top-1 agreement vs the bf16 teacher on a wikitext slice.

    The teacher's logits are dumped once to the volume (~10 GB for 64 512-token
    chunks); every candidate arm then reports mean KLD and same-top-1 %.
    """
    t0 = time.time()
    sh(f"mkdir -p /work/src /tmp/ccache-tmp && tar xf {SRC_TAR} -C /work/src")
    cfg = ("cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={os.environ.get('MACH1_CUDA_ARCH','90')} -DLLAMA_CURL=OFF ")
    sh(cfg + "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
             "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    sh("cmake --build /work/build -j --target llama-perplexity")
    print(f"build done in {time.time()-t0:.0f}s", flush=True)

    if not os.path.exists(CORPUS):
        sh(f"curl -L -o /tmp/wiki.zip {WIKI_URL} && python3 -c \"import zipfile; zipfile.ZipFile('/tmp/wiki.zip').extractall('/tmp')\"")
        sh(f"cp /tmp/wikitext-2-raw/wiki.test.raw {CORPUS}")
        VOL.commit()

    # 512-token chunks; n_chunks=64 keeps the uint16 logits file ~10 GB
    # -b 512: one sequence per batch. The default -b 2048 packs n_seq=4, and
    # the fused decode regions (mega/ZDP) gate on nt == 1 - r215 caught m1 and
    # zdp producing bit-identical logits because the fused path never ran.
    ppl_args = f"-f {CORPUS} -c 512 -b 512 --chunks {n_chunks} -fa 1"
    if not os.path.exists(BASE_LOGITS):
        sh(f"/work/build/bin/llama-perplexity -m {MODELS['bf16']} {ppl_args} "
           f"--save-all-logits {BASE_LOGITS} 2>&1 | tail -5")
        VOL.commit()

    outdir = f"/vol/bench/results/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # ub=1 (default): evaluate the corpus as single-token ubatches so the
    # DECODE kernel paths are what gets measured - the mach1 decode-only flags
    # (TC_FWHT, the fused regions, UFUSE, ...) all gate on nt == 1 and never
    # engage in a prefill-shaped eval, which would make this gate vacuous
    # for them. ub=512: prefill-shaped eval so the PP-stack lanes (nt >= 256
    # gates: RT_APPLY_TC, EXP_APPLY_MMA/FP16, GG) are what gets gated.
    # The teacher logits stay as dumped (the reference distribution);
    # candidate arms are compared to each other same-run.
    for arm in [a.strip() for a in arms.split(",")]:
        model_key, extra = ARMS[arm]
        env = dict(os.environ, **extra)
        cmd = (f"/work/build/bin/llama-perplexity -m {MODELS[model_key]} {ppl_args} -ub {ub} "
               f"--kl-divergence --kl-divergence-base {BASE_LOGITS}")
        print(f"--- gates arm {arm} ---", flush=True)
        j = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        tail = (j.stdout + j.stderr)[-3000:]
        print(tail, flush=True)
        with open(f"{outdir}/kld_{arm}.txt", "w") as f:
            f.write(j.stdout + j.stderr)
        VOL.commit()


@APP.local_entrypoint()
def main(arms: str = "m1,foldq,bf16", rounds: int = 3, tag: str = "", smoke: bool = True,
         mode: str = "bench", args: str = "", gpu: str = "H200"):
    tag = tag or time.strftime("s%m%d-%H%M")
    if mode == "build":
        build_only.remote()
        return
    if mode == "ptxv":
        ptxv.remote(args or "89")
        return
    if mode == "p16composegate":
        arch = args or ("89" if gpu.upper() == "L40S" else "90")
        p16_compose_ptxgate.remote(arch, tag)
        return
    if mode == "standardoutput":
        if gpu.upper() != "L40S":
            raise ValueError("standardoutput is frozen to the same-GPU L40S lane")
        standard_output_ptxgate.remote(tag, "89")
        standard_output_l40s.remote(tag, args or "64 9")
        return
    if mode == "splitkbitdump":
        if gpu.upper() != "L40S":
            raise ValueError("splitkbitdump is an L40S exact-shape gate")
        splitk_bitdump_l40s.remote(tag)
        return
    if mode == "splitktime":
        if gpu.upper() != "L40S":
            raise ValueError("splitktime is an L40S exact-shape gate")
        splitk_time_l40s.remote(tag)
        return
    if mode == "qonceipack":
        if gpu.upper() != "L40S":
            raise ValueError("qonceipack is an L40S exact-shape gate")
        qonce_ipack_probe_l40s.remote(tag, args or "9 48")
        return
    if mode == "qonceipacktime":
        if gpu.upper() != "L40S":
            raise ValueError("qonceipacktime is an L40S exact-shape gate")
        qonce_ipack_time_l40s.remote(tag)
        return
    if mode == "stdouttime":
        if gpu.upper() != "L40S":
            raise ValueError("stdouttime is an L40S exact-shape gate")
        stdout_time_l40s.remote(tag)
        return
    if mode == "megarace":
        if gpu.upper() != "L40S":
            raise ValueError("megarace is an L40S lane")
        megarace_l40s.remote(tag, args or "p16ks4,p16mp1")
        return
    if mode == "imma8pipe":
        if gpu.upper() != "L40S":
            raise ValueError("imma8pipe is an L40S lane")
        imma8_pipe_l40s.remote(tag, args or "9")
        return
    if mode == "gates":
        # args: "<n_chunks>" or "<n_chunks>@<ub>" (ub=512 engages the PP lanes)
        gc, _, gub = (args or "64").partition("@")
        gates.remote(arms, tag, int(gc or 64), int(gub or 1))
        return
    if mode == "batched16":
        batched_l40s_c16.remote(arms, tag, args if args else "1,2,4,8,16")
        return
    if mode == "batched":
        npl = args if args else "1,2,4,8,16"
        if gpu.upper() == "L40S":
            batched_l40s.remote(arms, tag, npl)
        else:
            batched.remote(arms, tag, npl)
        return
    if mode == "ntcheck":
        nps = args if args else "1,2,4"
        if gpu.upper() == "L40S":
            ntcheck_l40s.remote(arms, tag, nps)
        else:
            ntcheck.remote(arms, tag, nps)
        return
    if mode == "san":
        npl = args if args else "2"
        if gpu.upper() == "L40S":
            san_l40s.remote(arms, tag, npl)
        else:
            san.remote(arms, tag, npl)
        return
    ada = gpu.upper() == "L40S"
    if mode == "ops":
        (ops_l40s if ada else ops).remote(arms, tag)
        return
    if mode == "census":
        (census_l40s if ada else census).remote(arms, tag)
        return
    if mode == "optime":
        (optime_l40s if gpu.upper() == "L40S" else optime).remote(arms, tag)
        return
    if mode == "cpucheck":
        cpucheck.remote(arms, tag)
        return
    if mode == "mpsserve":
        mpsserve_l40s.remote(arms, tag, args or "2,4")
        return
    if mode == "nsysprof":
        (nsysprof_l40s if ada else nsysprof).remote(arms, tag, args or "2")
        return
    if mode == "stagetime":
        (stagetime_l40s if ada else stagetime).remote(arms, tag)
        return
    if mode == "graphstat":
        (graphstat_l40s if ada else graphstat).remote(arms, tag)
        return
    if mode == "bitdump":
        bitdump.remote(tag)
        return
    if mode == "getdraft":
        getdraft.remote(args)
        return
    if mode == "spec":
        (spec_l40s if ada else spec).remote(arms, tag)
        return
    if mode == "chainbench":
        (chainbench_l40s if ada else chainbench).remote(tag, args)
        return
    if mode == "layer":
        (layerbench_l40s if ada else layerbench).remote(tag, args)
        return
    if mode == "dp4a":
        dp4abench.remote(tag, args)
        return
    if gpu.upper() == "L40S":
        out = bench_l40s.remote(arms, rounds, tag, smoke)
    else:
        out = bench.remote(arms, rounds, tag, smoke)
    print(json.dumps(out))
