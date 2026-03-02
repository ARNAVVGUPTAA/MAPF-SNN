#!/bin/bash
# ============================================================================
# BDSM: BRAINS DON'T SIMPLY MULTIPLY
# Neuromorphic LSM Training Pipeline for MAPF
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
CONFIG="configs/config_lsm.yaml"
NUM_EPISODES=100
EVAL_EPISODES=50
MODE="reservoir_computing"

# Parse args
while getopts "c:n:e:m:h" opt; do
    case $opt in
        c) CONFIG="$OPTARG";;
        n) NUM_EPISODES="$OPTARG";;
        e) EVAL_EPISODES="$OPTARG";;
        m) MODE="$OPTARG";;
        h) 
            echo "Usage: $0 [-c config] [-n episodes] [-e eval_episodes] [-m mode]"
            echo "  -c: Config (default: configs/config_lsm.yaml)"
            echo "  -n: Train episodes (default: 100)"
            echo "  -e: Eval episodes (default: 50)"
            echo "  -m: Mode: reservoir_computing or end_to_end (default: reservoir_computing)"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}========================================================================"
echo "🧠 NEUROMORPHIC LSM FOR MAPF"
echo "========================================================================"
echo -e "   ${YELLOW}BDSM: BRAINS DON'T SIMPLY MULTIPLY!${NC}"
echo "========================================================================"
echo -e "${NC}"
echo "Config: $CONFIG | Mode: $MODE | Train: $NUM_EPISODES eps | Eval: $EVAL_EPISODES eps"
echo ""

# Update config
python3 << EOF
import yaml
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['readout']['collect_episodes'] = $NUM_EPISODES
config['training']['mode'] = '$MODE'
with open('$CONFIG', 'w') as f:
    yaml.dump(config, f)
print("✅ Config updated")
EOF

# Train
echo -e "${BLUE}🎓 TRAINING...${NC}"
python train_lsm.py --config "$CONFIG" || { echo -e "${RED}❌ Training failed${NC}"; exit 1; }
echo -e "${GREEN}✅ Training complete${NC}"

# Find checkpoint
CHECKPOINT=$(ls -t checkpoints/lsm/lsm_network_*.pt 2>/dev/null | head -1)
[ -z "$CHECKPOINT" ] && { echo -e "${RED}❌ No checkpoint found${NC}"; exit 1; }
echo -e "${GREEN}📦 Checkpoint: $CHECKPOINT${NC}"

# Evaluate
echo -e "${BLUE}📊 EVALUATING...${NC}"
python evaluate_lsm.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --num_episodes "$EVAL_EPISODES"
echo -e "${GREEN}✅ Complete! Check visualizations/lsm/ and logs/lsm/${NC}"
