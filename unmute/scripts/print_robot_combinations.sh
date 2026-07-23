#!/usr/bin/env bash
#
# Dump the domestic-robot system prompt and GBNF grammar for every meaningful
# combination, so you don't have to run the per-combo Python scripts by hand.
#
# Combinations covered:
#   1. real  | static places   -- physical robot vocabulary (robot_world tuples)
#   2. sim   | static places    -- AI2-THOR object set, static places
#   3. sim   | house places     -- AI2-THOR object set, places scoped to a house
#                                  (the SAMPLE_* vocab below; the bridge forwards
#                                   the real one from the backend's world_vocab())
#
# Usage (from anywhere):
#   unmute/scripts/print_robot_combinations.sh            # prompt + grammar
#   unmute/scripts/print_robot_combinations.sh prompt     # prompts only
#   unmute/scripts/print_robot_combinations.sh grammar    # grammars only
#
# Override the sample house vocab via env:
#   SAMPLE_ROOMS="bedroom,kitchen" SAMPLE_SURFACES="bed,sofa" \
#       unmute/scripts/print_robot_combinations.sh
set -euo pipefail

# Run from the unmute-ros/ project root (two levels up from this script) so that
# `uv run` resolves the right environment.
cd "$(dirname "$0")/../.."

WHAT="${1:-both}"   # both | prompt | grammar

# Sample house vocab -- matches the ProcTHOR house we've been testing. Override
# with SAMPLE_ROOMS / SAMPLE_SURFACES to inspect a different world.
SAMPLE_ROOMS="${SAMPLE_ROOMS:-bathroom,bedroom,kitchen,living room}"
SAMPLE_SURFACES="${SAMPLE_SURFACES:-arm chair,bed,counter top,dining table,dresser,shelf,sofa}"

banner() {
    echo
    echo "================================================================================"
    echo "== $1"
    echo "================================================================================"
}

# emit <label> <subcommand: prompt|grammar> [extra args...]
emit() {
    local label="$1" sub="$2"
    shift 2
    banner "$label  [$sub]"
    uv run python -m "unmute.scripts.print_robot_${sub}" "$@"
}

# emit both (or the selected one) for a combination
combo() {
    local label="$1"
    shift
    if [ "$WHAT" = "both" ] || [ "$WHAT" = "prompt" ]; then
        emit "$label" prompt "$@"
    fi
    if [ "$WHAT" = "both" ] || [ "$WHAT" = "grammar" ]; then
        emit "$label" grammar "$@"
    fi
}

combo "real  | static places"  --object-set real
combo "sim   | static places"  --object-set sim
combo "sim   | house places"   --object-set sim \
    --rooms "$SAMPLE_ROOMS" --surfaces "$SAMPLE_SURFACES"
