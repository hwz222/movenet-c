#!/bin/bash
# ================================================================
#  Master test runner
#
#  To add a new TB:
#    1. Add a compile step in the "COMPILE TBs" section
#    2. Add a run + compare block in the "RUN TEST CASES" section
#
#  Run:  bash tests/run_all_tb.sh
# ================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
TESTS="$ROOT/tests"
BUILD="$ROOT/build"
INC="$ROOT/include"

mkdir -p "$BUILD"

PASS=0
FAIL=0

# ── helper: run hex_compare and track result ──────────────────
compare() {   # compare <label> <out_file> <ref_file>
    local label="$1" out="$2" ref="$3"
    printf "  compare %-40s " "$label"
    if "$BUILD/hex_compare" "$out" "$ref"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

# ── helper: compile one TB, abort on error ────────────────────
compile_tb() {  # compile_tb <tb_src.c> [extra_src.c ...] -o <bin>
    gcc -I"$INC" -Wall -g "$@" || { echo "COMPILE FAILED: $*"; exit 1; }
}

# ================================================================
#  COMPILE TOOLS
# ================================================================
compile_tb "$TESTS/hex_compare.c" -o "$BUILD/hex_compare"

# ================================================================
#  COMPILE TBs
#  -- add one compile_tb line per new TB --
# ================================================================
compile_tb "$TESTS/tb_tensor3d_tensor_quantize_u8.c" "$SRC/tensor3d.c" -lm \
           -o "$BUILD/tb_tensor3d_tensor_quantize_u8"

compile_tb "$TESTS/tb_tensor3d_tensor_add_q.c" "$SRC/tensor3d.c" -lm \
           -o "$BUILD/tb_tensor3d_tensor_add_q"

# ================================================================
#  RUN TEST CASES
# ================================================================

# ── tensor_quantize_u8 ────────────────────────────────────────
echo "── tensor_quantize_u8 ──────────────────────────────────────"
TB_H=192
TB_W=192
TB_C=3
TB_IN="/home/jack/my_project/movenet-py/tensor_outputs/tensor_0"
TB_OUT="$BUILD/tb_quantize_u8_output.hex"
TB_REF="/home/jack/my_project/movenet-py/tensor_outputs/tensor_178"

"$BUILD/tb_tensor3d_tensor_quantize_u8" "$TB_H" "$TB_W" "$TB_C" "$TB_IN" "$TB_OUT"
compare "tensor_quantize_u8" "$TB_OUT" "$TB_REF"

# ── tensor_add_q ─────────────────────────────────────────────
echo "── tensor_add_q ────────────────────────────────────────────"
TB_AQ_H=24
TB_AQ_W=24
TB_AQ_C=32
TB_AQ_IN_A="/home/jack/my_project/movenet-py/tensor_outputs/tensor_192"   # edit
TB_AQ_IN_B="/home/jack/my_project/movenet-py/tensor_outputs/tensor_195"   # edit
TB_AQ_OUT="$BUILD/tb_add_q_output.hex"
TB_AQ_REF="/home/jack/my_project/movenet-py/tensor_outputs/tensor_196"    # edit
TB_AQ_SCALE_OUT=0.045551516115665436
TB_AQ_ZP_OUT=-14
TB_AQ_SCALE_A=0.0335574671626091
TB_AQ_ZP_A=-7
TB_AQ_SCALE_B=0.03476142883300781
TB_AQ_ZP_B=-5

"$BUILD/tb_tensor3d_tensor_add_q" \
    "$TB_AQ_H" "$TB_AQ_W" "$TB_AQ_C" \
    "$TB_AQ_IN_A" "$TB_AQ_IN_B" "$TB_AQ_OUT" \
    "$TB_AQ_SCALE_OUT" "$TB_AQ_ZP_OUT" \
    "$TB_AQ_SCALE_A"   "$TB_AQ_ZP_A" \
    "$TB_AQ_SCALE_B"   "$TB_AQ_ZP_B"
compare "tensor_add_q" "$TB_AQ_OUT" "$TB_AQ_REF"

# ── ADD NEW TB BLOCK BELOW ────────────────────────────────────
# echo "── <function_name> ─────────────────────────────────────────"
# TB_H=...  TB_W=...  TB_C=...
# TB_IN="..."  TB_OUT="$BUILD/<output>.hex"  TB_REF="..."
# "$BUILD/<tb_binary>" "$TB_H" "$TB_W" "$TB_C" "$TB_IN" "$TB_OUT"
# compare "<function_name>" "$TB_OUT" "$TB_REF"

# ================================================================
echo ""
echo "Results: $PASS PASS  /  $FAIL FAIL"
[ "$FAIL" -eq 0 ]
