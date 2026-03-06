# Hardware-Native Swarm MAPF Architecture

## Core Philosophy

**No matrix multiplication. No coordinate math. Pure spike routing.**

This is how insects navigate. This is what maps to FPGAs. This is **BDSM**  (Biologically-inspired Decentralized Spike-based Multi-agent pathfinding).

---

## The Three Modules

### 1. **Topographic Intent Map**
*Spatial broadcasts instead of abstract vectors*

#### The Problem
Standard ML: Agent broadcasts `action = 1` (UP). But neighbor doesn't know *where* you are. Requires coordinate math: `if abs(x_A - x_B) < 1 and dy_A == -dy_B: collision`.

#### The Biology
Your visual cortex physically maps space. Neurons that process "left side of visual field" are physically on the left side of V1. **Topology = anatomy.**

#### The Implementation
- Agent has 3×3 grid of neurons centered on self
- Moving EAST → fires neuron at position `(1, 0)` relative to center
- Neighbor receives broadcast → spike lands directly on *their* spatial grid
- If both agents fire same cell → massive lateral inhibition
- **Result**: Collision avoidance via physical wiring, not math

#### Hardware
- 9 neurons per agent
- Sparse spike routing between neighbors
- AND gates for overlap detection

---

### 2. **Central Pattern Generator (CPG)**
*Turn-taking without a coordinator*

#### The Problem
Lateral inhibition perfectly blocks head-on collisions. But both agents stay blocked forever. Classical solution: central coordinator flips coin. BDSM forbids this.

#### The Biology
Your spinal cord has rhythmic circuits that generate walking patterns without brain input. Cockroach legs coordinate via weak coupling between local oscillators.

#### The Implementation
- Two neurons per agent: `N_peak` (move) and `N_trough` (stay)
- `N_peak` fires → suppresses `N_trough` → fatigues after 8 ticks → stops
- `N_trough` wakes up → suppresses `N_peak` → fatigues → stops
- Agents weakly couple: if neighbor in `peak`, push self to `trough`
- **Result**: Deadlocked agents push out of phase. One moves, one waits. Emergent turn-taking.

#### Hardware
- 2 neurons per agent
- 2 counters (fatigue tracking)
- Bit-shift for decay (not floating point)

---

### 3. **Shadow Caster**
*Predictive collision via delay lines*

#### The Problem
Lateral inhibition stops face-to-face crashes. CPG resolves deadlocks. But neither prevents two agents from entering opposite ends of a long narrow corridor and trapping each other.

#### The Biology
Your cerebellum predicts collisions *before* they happen using forward models. Baseball outfielders don't compute parabolic trajectories — they track the ball's "optical acceleration" and predict the landing spot via interneuron delay lines.

#### The Implementation
1. **Velocity detection**: Agent B watches Agent A's intent over 2 ticks
   - Tick 1: Spike at grid position `x`
   - Tick 2: Spike at position `x+1`
   - Coincidence detector fires → "neighbor moving EAST"

2. **Cast the shadow**: Trigger "ghost" spikes that propagate EAST along B's spatial map, each delayed by 1 tick

3. **Wall collision**: Ghost spike overlaps with wall neuron → VETO spike

4. **VETO effect**: Routes to CPG, forcefully resets it to `trough` phase → agent STAYS

**Result**: Agent sees neighbor heading toward narrow corridor, predicts topological trap, yields *before* entering.

#### Hardware
- Shift registers (delay lines for ghost spikes)
- AND gates (coincidence detection)
- XOR with wall map (collision detection)

---

## Why This Works

### Emergent Behaviors
1. **Immediate collision avoidance**: Topographic overlap → lateral inhibition
2. **Deadlock resolution**: CPG phase coupling → turn-taking
3. **Corridor negotiation**: Shadow VETO → predictive yielding

### Hardware Benefits
- **No floating point**: Only spikes, counters, bit-shifts
- **No global state**: Each agent runs independently
- **No synchronization**: Asynchronous spike routing
- **Low latency**: O(1) operations, no matrix multiplies
- **Scalable**: Adding agents = adding parallel circuits

### Biological Plausibility
- **V1**: Topographic maps
- **Spinal CPGs**: Rhythmic coordination
- **Cerebellum**: Delay lines for prediction

---

## Training

**Reservoir computing**: Only train the observation → action readout. Everything else is fixed.

1. **Collect states**: Run expert trajectories, record reservoir activations
2. **Ridge regression**: Solve `W = (X^T X + αI)^-1 X^T Y` for each agent
3. **Done**: No backprop, no gradients, 10 seconds to train

---

## Code Structure

```
models/swarm_lsm.py
├── TopographicIntentMap (3×3 spike grid)
├── CPG (2-neuron oscillator)
├── ShadowCaster (delay lines + veto)
├── AgentLSM (combines all three)
└── SwarmLSM (multi-agent network)

train_swarm_lsm.py
├── Load dataset
├── Collect reservoir states
├── Train ridge regression
└── Evaluate
```

---

## The Wiring

```
Observation → Reservoir → Readout → Tentative Action
                                    ↓
                            Intent Map (3×3 spikes)
                                    ↓
                            ┌───────┴────────┐
                            ↓                ↓
                    Lateral Inhibition   Shadow Caster
                            ↓                ↓
                            └────→ CPG ←─────┘
                                    ↓
                            STAY or MOVE?
                                    ↓
                            Final Action
```

**Key insight**: CPG receives inputs from BOTH lateral inhibition (immediate deadlock) AND shadow caster (predictive VETO). This is the secret sauce.

---

##Comparison to Standard Approaches

| Feature | Standard MAPF | This Architecture |
|---------|---------------|-------------------|
| Collision check | `dist(A, B) < threshold` | Spike overlap on spatial grid |
| Deadlock | Central coordinator | CPG phase coupling |
| Planning | A* / CBS | Shadow casting delay lines |
| Hardware | GPU/CPU with FP32 | FPGA with binary spikes |
| Scalability | O(N²) communication | O(neighbors) spike routing |
| Latency | ~ms (matrix ops) | ~µs (spike propagation) |

---

## Next Steps

1. **Test minimal reservoir size**: Current=128N. Try 64N, 32N, even 16N.
2. **Visualize CPG phases**: Are agents actually going out of phase during deadlocks?
3. **Track shadow VETOs**: How often does predictive yielding prevent traps?
4. **FPGA prototype**: Map to actual spiking hardware

This is the path to **real-time swarm robotics at scale**.
