import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import multiprocessing as mp
import time
import torch
from collections import deque

def visualizer_process_worker(queue, config):
    """
    Main visualization process with 4 graphs:
    1. Training Loss
    2. Communication/Coordination/Oscillation Weights
    3. Agents Reached in Validation
    4. SNN Action Animation (60 steps)
    """
    print("🎬 Initializing visualization process...")
    
    grid_size = config.get('board_size', [9, 9])[0]
    
    # Create figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Enhanced SNN Training Visualization", fontsize=16, fontweight='bold')
    
    # Initialize empty plots for all 4 graphs
    
    # Graph 1: Training Loss (empty)
    axes[0, 0].set_title("Training Loss", fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].plot([], [], 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(0, 1)
    
    # Graph 2: Communication/Coordination/Oscillation Weights (empty)
    axes[0, 1].set_title("Weight Metrics", fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Weight Value")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].plot([], [], 'g-', linewidth=2, marker='s', markersize=4, label='Communication')
    axes[0, 1].plot([], [], 'r-', linewidth=2, marker='^', markersize=4, label='Coordination')
    axes[0, 1].plot([], [], 'm--', linewidth=2, marker='o', markersize=3, label='Oscillation')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 10)
    axes[0, 1].set_ylim(0, 10)
    
    # Graph 3: Agents Reached in Validation (empty)
    axes[1, 0].set_title("Validation: Agents Reached", fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Agents Reached (%)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].plot([], [], 'b-', linewidth=2, marker='s', markersize=4, label='Agents Reached')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_ylim(0, 100)
    
    # Graph 4: SNN Action Animation (empty grid)
    axes[1, 1].set_title("SNN Action Animation", fontweight='bold')
    axes[1, 1].set_xlim(-0.5, grid_size - 0.5)
    axes[1, 1].set_ylim(-0.5, grid_size - 0.5)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(grid_size/2, grid_size/2, 'Waiting for SNN episodes...', 
                    ha='center', va='center', fontsize=12, alpha=0.7)
    
    # Draw grid lines for animation plot
    for i in range(grid_size + 1):
        axes[1, 1].axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        axes[1, 1].axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Force the window to appear and stay on top
    fig.canvas.manager.window.wm_attributes('-topmost', True)
    fig.canvas.manager.window.wm_attributes('-topmost', False)  # Remove topmost after showing
    plt.show(block=False)
    plt.draw()
    plt.pause(0.5)  # Give it time to render
    
    print("✅ Visualization window created with empty plots")
    
    # Data storage for plots
    training_losses = []
    communication_weights = []
    coordination_weights = []
    oscillation_weights = []
    agents_reached_rates = []
    epochs = []
    
    # Animation state
    current_animation = None
    animation_step = 0
    last_animation_time = time.time()
    animation_speed = 0.15  # seconds between animation steps
    
    print("🔄 Starting main visualization loop...")
    
    while True:
        try:
            # Check for new data from training process (non-blocking)
            try:
                data = queue.get_nowait()  # Non-blocking get
                if data == "STOP":
                    break
                
                data_type = data.get('type', 'unknown')
                
                if data_type == 'training_metrics':
                    # Update training metrics plots
                    epoch = data['epoch']
                    loss = data['loss']
                    comm_weight = data.get('communication_weight', 0)
                    coord_weight = data.get('coordination_weight', 0)
                    osc_weight = data.get('oscillation_weight', 0)
                    agents_reached = data.get('agents_reached_rate', 0) * 100  # Convert to percentage
                    
                    # Store data
                    epochs.append(epoch)
                    training_losses.append(loss)
                    communication_weights.append(comm_weight)
                    coordination_weights.append(coord_weight)
                    oscillation_weights.append(osc_weight)
                    agents_reached_rates.append(agents_reached)
                    
                    # Update Graph 1: Training Loss
                    axes[0, 0].clear()
                    axes[0, 0].set_title("Training Loss", fontweight='bold')
                    axes[0, 0].set_xlabel("Epoch")
                    axes[0, 0].set_ylabel("Loss")
                    axes[0, 0].grid(True, alpha=0.3)
                    if epochs:
                        axes[0, 0].plot(epochs, training_losses, 'b-', linewidth=2, marker='o', markersize=4)
                    
                    # Update Graph 2: Weight Metrics
                    axes[0, 1].clear()
                    axes[0, 1].set_title("Weight Metrics", fontweight='bold')
                    axes[0, 1].set_xlabel("Epoch")
                    axes[0, 1].set_ylabel("Weight Value")
                    axes[0, 1].grid(True, alpha=0.3)
                    if epochs:
                        axes[0, 1].plot(epochs, communication_weights, 'g-', linewidth=2, marker='s', markersize=4, label='Communication')
                        axes[0, 1].plot(epochs, coordination_weights, 'r-', linewidth=2, marker='^', markersize=4, label='Coordination')
                        axes[0, 1].plot(epochs, oscillation_weights, 'm--', linewidth=2, marker='o', markersize=3, label='Oscillation')
                        axes[0, 1].legend()
                    
                    # Update Graph 3: Agents Reached
                    axes[1, 0].clear()
                    axes[1, 0].set_title("Validation: Agents Reached", fontweight='bold')
                    axes[1, 0].set_xlabel("Epoch")
                    axes[1, 0].set_ylabel("Agents Reached (%)")
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].set_ylim(0, 100)
                    if epochs:
                        axes[1, 0].plot(epochs, agents_reached_rates, 'b-', linewidth=2, marker='s', markersize=4)
                    
                    # Refresh display
                    plt.draw()
                    plt.pause(0.01)
                    print(f"📊 Updated training metrics - Epoch: {epoch}, Loss: {loss:.4f}")
                
                elif data_type == 'snn_episode':
                    # Start new SNN action animation
                    current_animation = {
                        'positions': data['initial_positions'],
                        'goals': data['goals'],
                        'obstacles': data['obstacles'],
                        'snn_actions': data['snn_actions'],  # 60-step SNN action sequence
                        'case_info': data['case_info'],
                        'total_steps': len(data['snn_actions'])
                    }
                    animation_step = 0
                    last_animation_time = time.time()
                    
                    print(f"🎬 Starting SNN animation for {current_animation['case_info']} - {current_animation['total_steps']} steps")
            
            except:
                # No data in queue, continue with animation
                pass
            
            # Handle SNN animation independently of queue data
            if current_animation is not None:
                current_time = time.time()
                if current_time - last_animation_time >= animation_speed:
                    #print(f"🎮 Animating step {animation_step + 1}/{current_animation['total_steps']}")
                    animate_snn_step(axes[1, 1], current_animation, animation_step, grid_size)
                    animation_step += 1
                    last_animation_time = current_time
                    
                    # Check if animation is complete
                    if animation_step >= current_animation['total_steps']:
                        current_animation = None
                        animation_step = 0
                        print("✅ SNN animation complete")
                    
                    plt.draw()
                    plt.pause(0.01)
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.01)
            
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"[Visualizer Process Error]: {e}")
            import traceback
            traceback.print_exc()
    
    plt.close(fig)
    print("[Visualizer Process Stopped]")

def animate_snn_step(ax, animation_data, step, grid_size):
    """
    Animate a single step of SNN actions
    """
    ax.clear()
    ax.set_title(f"SNN Actions: {animation_data['case_info']} - Step {step+1}/{animation_data['total_steps']}", fontweight='bold')
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Draw obstacles
    obstacles = animation_data['obstacles']
    if obstacles is not None and len(obstacles) > 0:
        for obs in obstacles:
            if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 2:
                obs_x, obs_y = int(obs[0]), int(obs[1])
                if 0 <= obs_x < grid_size and 0 <= obs_y < grid_size:
                    rect = patches.Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1, 
                                           facecolor='darkred', alpha=0.7, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
    
    # Colors for agents
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Draw goals
    goals = animation_data['goals']
    for i, goal in enumerate(goals):
        color = colors[i % len(colors)]
        goal_x, goal_y = int(goal[0]), int(goal[1])
        ax.plot(goal_x, goal_y, '*', markersize=20, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.text(goal_x, goal_y - 0.4, f'G{i}', ha='center', va='center', fontweight='bold', color=color, fontsize=10)
    
    # Calculate current positions based on SNN actions up to this step
    current_positions = np.array(animation_data['positions']).copy()
    snn_actions = animation_data['snn_actions']
    
    # Apply SNN actions sequentially up to current step
    action_deltas = {
        0: (0, 0),   # Stay
        1: (1, 0),   # Right
        2: (0, 1),   # Up
        3: (-1, 0),  # Left
        4: (0, -1)   # Down
    }
    
    for t in range(min(step + 1, len(snn_actions))):
        if t < len(snn_actions):
            step_actions = snn_actions[t]  # Actions for all agents at timestep t
            for agent_i in range(len(current_positions)):
                if agent_i < len(step_actions):
                    action = int(step_actions[agent_i])
                    if action in action_deltas:
                        dx, dy = action_deltas[action]
                        new_x = max(0, min(grid_size - 1, int(current_positions[agent_i][0] + dx)))
                        new_y = max(0, min(grid_size - 1, int(current_positions[agent_i][1] + dy)))
                        current_positions[agent_i] = [new_x, new_y]
    
    # Draw current agent positions
    for i, pos in enumerate(current_positions):
        color = colors[i % len(colors)]
        agent_x, agent_y = int(pos[0]), int(pos[1])
        
        # Agent as filled circle
        circle = patches.Circle((agent_x, agent_y), 0.35, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Agent number
        ax.text(agent_x, agent_y, str(i), color='white', ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Show current action as arrow if not stay
        if step < len(snn_actions) and i < len(snn_actions[step]):
            current_action = int(snn_actions[step][i])
            if current_action in action_deltas and current_action != 0:  # Not stay action
                dx, dy = action_deltas[current_action]
                ax.arrow(agent_x, agent_y, dx * 0.3, dy * 0.3, 
                        head_width=0.1, head_length=0.1, fc='yellow', ec='black', linewidth=2)

class MAPFVisualizer:
    """
    Enhanced visualizer with 4 graphs and SNN action animation
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.visualize_training = config.get('visualize_training', False)
        self.training_history = {}  # Will be set by trainer
        
        if self.visualize_training:
            # Use multiprocessing context for robustness
            ctx = mp.get_context('spawn')
            self.queue = ctx.Queue(maxsize=100)
            self.process = ctx.Process(target=visualizer_process_worker, args=(self.queue, self.config))
    
    def start(self):
        """Starts the separate visualization process."""
        if self.visualize_training:
            print("🎬 Starting enhanced SNN visualization process with 4 graphs...")
            self.process.start()
    
    def update_training_metrics(self, epoch, loss, communication_weight=0, coordination_weight=0, 
                              oscillation_weight=0, agents_reached_rate=0):
        """Send training metrics to visualization process"""
        if self.visualize_training and self.process.is_alive():
            try:
                data = {
                    'type': 'training_metrics',
                    'epoch': epoch,
                    'loss': loss,
                    'communication_weight': communication_weight,
                    'coordination_weight': coordination_weight,
                    'oscillation_weight': oscillation_weight,
                    'agents_reached_rate': agents_reached_rate
                }
                self.queue.put_nowait(data)
            except:
                pass  # Skip if queue is full
    
    def show_episode(self, initial_positions, goals, obstacles, snn_actions, case_info):
        """Send SNN episode data for animation - MUST be SNN actions for all 60 steps"""
        if self.visualize_training and self.process.is_alive():
            try:
                # Ensure snn_actions is properly formatted [timesteps, num_agents]
                if isinstance(snn_actions, torch.Tensor):
                    snn_actions = snn_actions.detach().cpu().numpy()
                
                data = {
                    'type': 'snn_episode',
                    'initial_positions': initial_positions,
                    'goals': goals,
                    'obstacles': obstacles,
                    'snn_actions': snn_actions,  # This MUST be SNN predicted actions
                    'case_info': case_info
                }
                self.queue.put_nowait(data)
                print(f"🎬 Queued SNN episode animation: {case_info} - {len(snn_actions)} steps")
            except Exception as e:
                print(f"⚠️ Failed to queue SNN episode: {e}")
    
    def stop(self):
        """Stops the visualization process gracefully."""
        if self.visualize_training and hasattr(self, 'process') and self.process.is_alive():
            print("🛑 Stopping enhanced visualization process...")
            try:
                self.queue.put("STOP")
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
            except Exception as e:
                print(f"⚠️ Error stopping visualization: {e}")
                if hasattr(self, 'process'):
                    self.process.terminate()