#!/usr/bin/env bash
# lora_strength_demo.sh
#
# Runs inference across multiple prompts and LoRA strength values using modl,
# so you can visually compare how the style bleeds in at 0 → 1.
#
# Usage:
#   ./scripts/lora_strength_demo.sh [LORA_NAME] [BASE_MODEL]
#
# Defaults:
#   LORA_NAME  = kids-art-sdxl-v2
#   BASE_MODEL = sdxl-base-1.0

set -euo pipefail

LORA="${1:-kids-art-sdxl-v2}"
BASE="${2:-sdxl-base-1.0}"
SEED=42

# ── Strength sweep ────────────────────────────────────────────────────────────
STRENGTHS=(0.0 0.4 0.7 1.0)

# ── Prompts ───────────────────────────────────────────────────────────────────
# Mix of trigger-word prompts and neutral ones to see the full spectrum
declare -a PROMPTS=(
  "KIDSART a dog playing in the park"
  "KIDSART a rocket ship flying through colorful space"
  "KIDSART a princess in a magical forest"
  "a dog playing in the park"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
BOLD=$(tput bold 2>/dev/null || echo "")
RESET=$(tput sgr0 2>/dev/null || echo "")
CYAN=$(tput setaf 6 2>/dev/null || echo "")

total=$(( ${#PROMPTS[@]} * ${#STRENGTHS[@]} ))
current=0

echo ""
echo "${BOLD}${CYAN}LoRA Strength Demo${RESET}"
echo "  LoRA:   ${LORA}"
echo "  Base:   ${BASE}"
echo "  Seed:   ${SEED}"
echo "  Combos: ${total}"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for prompt in "${PROMPTS[@]}"; do
  for strength in "${STRENGTHS[@]}"; do
    (( current++ )) || true
    echo "${BOLD}[${current}/${total}]${RESET} strength=${strength} | \"${prompt}\""

    modl generate "${prompt}" \
      --base "${BASE}" \
      --lora "${LORA}" \
      --lora-strength "${strength}" \
      --seed "${SEED}" \
      --size "1:1"

    echo ""
  done
done

echo "${BOLD}Done!${RESET} Images saved to ~/.modl/outputs/$(date +%Y-%m-%d)/"
