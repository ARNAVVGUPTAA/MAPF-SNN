import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, base
from spikingjelly.activation_based.learning import STDPLearner
from config import config
from debug_utils import debug_print


class AttentionFeatureExtractor(nn.Module):
    """
    Brain-like attention feature extractor that learns to focus on important spatial regions.
    
    This module solves the spatial blindness issue by:
    1. Creating a shared rich feature embedding from raw FOV input
    2. Computing importance scores from the embedded features (not raw pixels)
    3. Applying attention and processing attended features
    4. Being lightweight and cognitively plausible
    """
    def __init__(self, input_channels, fov_size, output_dim):
        super().__init__()
        fov_flat_size = input_channels * fov_size * fov_size
        shared_hidden_dim = 256  # A shared intermediate dimension for richer features
        
        # NEW: A shared layer to create a rich feature embedding first
        # This is like the visual cortex processing raw light into basic shapes and edges
        self.shared_embedding = nn.Sequential(
            nn.Linear(fov_flat_size, shared_hidden_dim),
            nn.ReLU()
        )

        # UPDATED: Attention network now works on the richer embedded features
        self.attention_net = nn.Sequential(
            nn.Linear(shared_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fov_size * fov_size),  # Output one weight per cell in the FOV
            nn.Softmax(dim=-1)  # Normalize weights so they sum to 1
        )
        
        # UPDATED: Feature network also works on the richer embedded features
        self.feature_net = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Flatten the input
        flat_x = x.reshape(batch_size, -1)
        actual_input_size = flat_x.shape[1]
        
        # Check if input size matches what the linear layer expects
        expected_input_size = self.shared_embedding[0].in_features
        if actual_input_size != expected_input_size:
            # Print debug info to understand the mismatch
            print(f"🔧 [DEBUG] Input size mismatch: expected {expected_input_size}, got {actual_input_size}")
            print(f"🔧 [DEBUG] Input shape: {x.shape}")
            
            # Create a new embedding layer with the correct input size
            shared_hidden_dim = self.shared_embedding[0].out_features
            device = next(self.parameters()).device
            
            self.shared_embedding = nn.Sequential(
                nn.Linear(actual_input_size, shared_hidden_dim),
                nn.ReLU()
            ).to(device)
            
            print(f"🔧 [DEBUG] Dynamically adjusted embedding layer to handle {actual_input_size} inputs")
        
        # NEW: First, create the shared feature embedding
        # This transforms raw FOV pixels into meaningful features like edges, shapes, patterns
        embedded_features = self.shared_embedding(flat_x)
        
        # UPDATED: Compute attention from the embedded features, not the raw input
        # The attention mechanism now works on processed features rather than raw pixels
        attention_weights = self.attention_net(embedded_features)
        
        # DYNAMIC ADJUSTMENT: Check if attention network output matches spatial dimensions
        spatial_size = x.shape[2] * x.shape[3]  # height * width
        if attention_weights.shape[1] != spatial_size:
            # Dynamically adjust attention network to match actual input spatial dimensions
            shared_hidden_dim = self.shared_embedding[0].out_features
            device = next(self.parameters()).device
            
            print(f"🔧 [DEBUG] Attention size mismatch: expected {spatial_size}, got {attention_weights.shape[1]}")
            print(f"🔧 [DEBUG] Adjusting attention network from {attention_weights.shape[1]} to {spatial_size} outputs")
            
            self.attention_net = nn.Sequential(
                nn.Linear(shared_hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, spatial_size),  # Match actual spatial dimensions
                nn.Softmax(dim=-1)  # Normalize weights so they sum to 1
            ).to(device)
            
            # Recompute attention weights with the new network
            attention_weights = self.attention_net(embedded_features)
        
        # Reshape weights to match the spatial dimensions of the FOV
        # attention_map shape: [batch_size, 1, height, width]
        attention_map = attention_weights.reshape(batch_size, 1, x.shape[2], x.shape[3])
        
        # Apply attention: multiply the original input by the importance map
        # This highlights important cells and suppresses irrelevant ones
        attended_x = x * attention_map
        
        # UPDATED: Now, create a new embedding from the *attended* input
        # This processes the spatially-filtered input through the shared embedding again
        attended_flat = attended_x.reshape(batch_size, -1)
        attended_embedded_features = self.shared_embedding(attended_flat)
        
        # Process the attended embedding through the feature network
        features = self.feature_net(attended_embedded_features)
        
        # CRITICAL: Normalize output to consistent range for downstream processing
        features = F.layer_norm(features, normalized_shape=[features.shape[-1]], eps=1e-6)
        features = torch.clamp(features, -1.0, 1.0)  # Consistent range for SNN processing
        
        return features


class AdaptiveLIFNode(neuron.LIFNode):
    """
    LIF neuron that adapts to different batch sizes between train/validation.
    Key improvement: Better parameter initialization for consistent spiking.
    """
    def __init__(self, tau=None, v_threshold=None, v_reset=None, surrogate_alpha=None, detach_reset=None, *args, **kwargs):
        # Use config parameters for better spiking - START WITH CONFIG THRESHOLD DIRECTLY
        config_threshold = config.get('lif_v_threshold', 0.01)  # Use our aggressive threshold from config (was 0.05)
        super().__init__(
            tau=tau if tau is not None else config.get('lif_tau', 10.0), 
            v_threshold=v_threshold if v_threshold is not None else config_threshold,  # Start with config threshold
            v_reset=v_reset if v_reset is not None else config.get('lif_v_reset', 0.0),
            detach_reset=detach_reset if detach_reset is not None else config.get('detach_reset', True),
            *args, **kwargs
        )

    def reset(self):
        """Reset to scalar voltage to adapt to new batch sizes"""
        neuron.BaseNode.reset(self)
        # Ensure v_reset and v_threshold are always tensors for consistent behavior
        if not isinstance(self.v_reset, torch.Tensor):
            self.v_reset = torch.tensor(self.v_reset, dtype=torch.float32)
        if not isinstance(self.v_threshold, torch.Tensor):
            self.v_threshold = torch.tensor(self.v_threshold, dtype=torch.float32)
        self.v = self.v_reset  # Reset to scalar for shape adaptation

    def single_step_forward(self, x):
        # Ensure v_reset is a tensor
        if not isinstance(self.v_reset, torch.Tensor):
            self.v_reset = torch.tensor(self.v_reset, device=x.device, dtype=x.dtype)
        
        # Ensure v_threshold is a tensor
        if not isinstance(self.v_threshold, torch.Tensor):
            self.v_threshold = torch.tensor(self.v_threshold, device=x.device, dtype=x.dtype)
        
        # Ensure membrane potential tensor shape matches input x
        if not isinstance(getattr(self, 'v', None), torch.Tensor) or self.v.shape != x.shape:
            self.v = self.v_reset * torch.ones_like(x, device=x.device)
        
        # Call parent forward to get output spikes
        out = super().single_step_forward(x)
        
        # STORE ACTUAL SPIKE OUTPUT FOR MONITORING
        self._last_spike_output = out.detach().clone()
        
        # SMART ADAPTIVE THRESHOLD LOGIC for normalized inputs
        if self.training:
            with torch.no_grad():  # Ensure threshold adaptation doesn't interfere with gradients
                target_rate = config.get('target_spike_rate', 0.08)  # 8% target from config
                max_spike_rate = config.get('max_spike_rate', 0.15)   # 15% maximum from config  
                min_spike_rate = config.get('min_spike_rate', 0.008)  # 0.8% minimum from config
                current_rate = out.detach().mean()
                
                # Get current membrane potential statistics
                membrane_mean = self.v.detach().mean()
                membrane_std = self.v.detach().std() + 1e-8
                
                # For NORMALIZED inputs, we expect membrane potentials in a predictable range
                # Since inputs are normalized to [-1, 1], membrane potentials should be reasonable
                
                # HEALTHY ZONE: No adjustment needed if spike rate is in acceptable range
                if min_spike_rate <= current_rate <= max_spike_rate:
                    # Spike rate is healthy - no threshold adjustment needed
                    pass
                
                # TOO FEW SPIKES: Lower threshold to increase firing
                elif current_rate < min_spike_rate:
                    # Calculate how much to lower threshold based on severity
                    spike_deficit = (min_spike_rate - current_rate) / (min_spike_rate + 1e-8)
                    
                    if current_rate == 0.0:
                        # Complete silence - emergency threshold reduction
                        new_threshold = membrane_mean - 0.5 * membrane_std
                    else:
                        # Gradual threshold reduction based on deficit
                        threshold_reduction = spike_deficit * 0.3 * membrane_std
                        new_threshold = self.v_threshold - threshold_reduction
                    
                    # Apply the adjustment with momentum (smoother adaptation)
                    adaptation_weight = min(0.5, spike_deficit)  # Max 50% adaptation per step
                    new_threshold = (1.0 - adaptation_weight) * self.v_threshold + adaptation_weight * new_threshold
                
                # TOO MANY SPIKES: Raise threshold to reduce firing  
                elif current_rate > max_spike_rate:
                    # Calculate how much to raise threshold based on severity
                    spike_excess = (current_rate - max_spike_rate) / max_spike_rate
                    
                    # Gradual threshold increase based on excess
                    threshold_increase = spike_excess * 0.5 * membrane_std
                    new_threshold = self.v_threshold + threshold_increase
                    
                    # Apply the adjustment with momentum
                    adaptation_weight = min(0.7, spike_excess)  # Max 70% adaptation for high spike rates
                    new_threshold = (1.0 - adaptation_weight) * self.v_threshold + adaptation_weight * new_threshold
                
                # Apply reasonable bounds for normalized inputs
                # With normalized inputs, thresholds should be in a reasonable range
                min_threshold = 0.01   # Very sensitive
                max_threshold = 5.0    # Very conservative (for normalized inputs)
                
                if 'new_threshold' in locals():
                    new_threshold = torch.clamp(new_threshold, min_threshold, max_threshold)
                    self.v_threshold = new_threshold.detach().clone()
        
        return out


class AttentionGate(nn.Module):
    """
    Simple attention mechanism to focus on important spatial/feature locations.
    This helps the SNN pay attention to relevant parts of the input.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention_fc = nn.Linear(input_dim, input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)
        
        # Learnable residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # Compute attention weights
        attention_weights = torch.sigmoid(self.attention_fc(x))
        # Compute gating values  
        gate_values = torch.tanh(self.gate_fc(x))
        # Apply attention: element-wise multiplication with learnable residual weight
        return x + (attention_weights * gate_values * self.residual_weight)  # Residual + attended features


class DynamicGraphSNN(base.MemoryModule):
    """
    Dynamic Graph-based Spiking Neural Network for localized decision making in MAPF.
    
    Key improvements over static version:
    - Dynamic adjacency matrix based on agent proximity and relevance
    - Learned edge weights for adaptive communication importance
    - Proximity-based agent communication
    - Temporal adaptation of graph structure
    - Enhanced conflict detection with spatial awareness
    - Batch normalization between spiking layers for stable training
    """
    def __init__(self, input_size, hidden_size, num_agents=5, cfg=None):
        super().__init__()
        
        # Store config for later use
        self.config = cfg if cfg is not None else config
        
        # Get SNN parameters from global config
        self.time_steps = config.get('snn_time_steps', 3)
        self.input_scale = config.get('input_scale', 1.0)  # Updated to match config fix
        self.num_agents = num_agents
        self.hidden_size = hidden_size
        
        # Batch normalization parameters
        bn_momentum = float(cfg.get('batch_norm_momentum', 0.1)) if cfg else 0.1
        bn_eps = float(cfg.get('batch_norm_eps', 1e-5)) if cfg else 1e-5
        
        # Dynamic graph parameters
        self.proximity_threshold = config.get('graph_proximity_threshold', 0.3)  # Threshold for agent proximity in feature space
        self.max_connections = config.get('max_connections', 5)  # Maximum connections per agent (updated to match config)
        self.edge_learning_rate = config.get('edge_learning_rate', 0.2)  # How fast edges adapt (updated to match config)
        
        # Convert hard-coded weights to learnable parameters
        self.recurrence_weight = nn.Parameter(torch.tensor(config.get('recurrence_weight', 0.4)))  # Updated to match config
        self.input_weight = nn.Parameter(torch.tensor(config.get('input_weight', 0.6)))  # Updated to match config
        self.comm_message_weight = nn.Parameter(torch.tensor(0.3))  # Learnable communication mixing weight
        
        # Hesitation parameters - controls decision confidence
        self.hesitation_weight = config.get('hesitation_weight', 0.05)  # Updated default to match config
        self.confidence_threshold = config.get('confidence_threshold', 0.9)  # Updated default to match config
        
        # Learnable gate thresholds - updated defaults to match config
        self.hesitation_gate_threshold = nn.Parameter(torch.tensor(config.get('hesitation_gate_threshold', 0.15)))  # Kept same
        
        # Learnable scaling factors
        self.uncertainty_scale = nn.Parameter(torch.tensor(2.0))  # For hesitation input scaling
        
        # Input processing with enhanced spatial attention
        self.spatial_attention = AttentionGate(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        
        # Position encoding for spatial awareness (helps with proximity calculation)
        self.position_encoder = nn.Linear(2, hidden_size // 4)  # Encode x,y positions
        
        # Dynamic graph structure learning
        self.edge_weight_fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.edge_weight_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.edge_weight_fc2 = nn.Linear(hidden_size, 1)
        
        # Enhanced graph-based communication layers
        self.agent_comm_fc = nn.Linear(hidden_size, hidden_size)
        self.agent_comm_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.message_agg_fc = nn.Linear(hidden_size, hidden_size)
        self.message_agg_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.edge_update_fc = nn.Linear(hidden_size * 2, hidden_size)  # For updating edge representations
        self.edge_update_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        
        # Proximity-aware local decision making
        self.local_decision_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.01) * config.get('lif_threshold_local_decision_scale', 2.0),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Enhanced hesitation with spatial awareness
        self.hesitation_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.hesitation_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * config.get('lif_tau_hesitation_scale', 1.5),
            v_threshold=config.get('lif_v_threshold', 0.01) * config.get('lif_threshold_hesitation_scale', 0.8),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Temporal memory with graph adaptation
        self.recurrent_fc = nn.Linear(hidden_size, hidden_size)
        self.recurrent_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.temporal_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * config.get('lif_tau_temporal_scale', 1.2),
            v_threshold=config.get('lif_v_threshold', 0.01),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Output processing
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.output_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        
        # Learnable edge weights initialization factor
        self.edge_init_weight = nn.Parameter(torch.tensor(0.5))
        
        # Initialize edge weights buffer using the learnable parameter
        self.register_buffer('edge_weights', torch.ones(num_agents, num_agents) * 0.5)
        
        # Add a buffer to store explicitly provided positions
        self.register_buffer('provided_positions', None)
        
        # Initialize batch norm running means for auxiliary heads to wake them up
        self._init_auxiliary_batch_norms()
        
        # =============================================================================
        # STDP COORDINATION SYSTEM
        # =============================================================================
        
        # STDP coordination parameters
        self.use_stdp_coordination = cfg.get('use_stdp_coordination', True) if cfg else True
        self.board_size = cfg.get('board_size', [9, 9])[0] if cfg else 9  # Assuming square board
        self.fov_size = 7  # 7x7 FOV from recent updates
        self.num_directions = 5  # Stay, Right, Up, Left, Down
        
        if self.use_stdp_coordination:
            # STDP parameters
            self.stdp_tau_facilitation = cfg.get('stdp_tau_facilitation', 50.0) if cfg else 50.0
            self.stdp_tau_inhibition = cfg.get('stdp_tau_inhibition', 20.0) if cfg else 20.0
            self.stdp_lr_facilitation = cfg.get('stdp_lr_facilitation', 2e-4) if cfg else 2e-4
            self.stdp_lr_inhibition = cfg.get('stdp_lr_inhibition', 1e-4) if cfg else 1e-4
            
            # Momentum parameters
            self.momentum_decay = cfg.get('momentum_decay', 0.8) if cfg else 0.8
            self.momentum_alignment_threshold = cfg.get('momentum_alignment_threshold', 0.6) if cfg else 0.6
            
            # Safety parameters
            self.collision_history_window = cfg.get('collision_history_window', 10) if cfg else 10
            self.safe_region_threshold = cfg.get('safe_region_threshold', 0.7) if cfg else 0.7
            
            # Coordination parameters
            self.coordination_radius = cfg.get('coordination_radius', 3.0) if cfg else 3.0
            self.flow_field_strength = cfg.get('flow_field_strength', 0.5) if cfg else 0.5
            
            # Action deltas for movement directions
            self.register_buffer('action_deltas', torch.tensor([
                [0, 0],   # 0: Stay
                [1, 0],   # 1: Right
                [0, 1],   # 2: Up  
                [-1, 0],  # 3: Left
                [0, -1],  # 4: Down
            ], dtype=torch.float32))
            
            # STDP coordination layers
            self.momentum_processor = nn.Linear(2, hidden_size // 4)  # Process momentum vectors
            self.intention_flow_processor = nn.Sequential(
                nn.Linear(self.num_directions, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4)
            )
            self.coordination_integrator = nn.Linear(hidden_size + hidden_size // 4 + hidden_size // 4, hidden_size)
            self.coordination_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
            
            # Initialize STDP state tracking
            self.reset_stdp_state()
        
        # =============================================================================
        
    def _init_auxiliary_batch_norms(self):
        """Initialize batch norm running means for auxiliary heads to wake them up"""
        # Initialize BN running-mean to 0.4 so the first forward pass can spike
        # and the learnable mean will track data thereafter
        nn.init.constant_(self.hesitation_bn.running_mean, 0.4)
    
    def reset_state(self, detach=True):
        """
        Reset all hidden states to prevent snowball effect across mini-batches.
        
        Problem: neuron voltages, graph memories, and STDP traces accumulate across 
        mini-batches because reset only happens at epoch boundaries. After ≈40 steps 
        almost every membrane sits above threshold → surrogate gradient ≈ 0.
        
        Args:
            detach: If True, detach gradients. If False, zero the values.
        """
        for m in self.modules():
            if hasattr(m, 'v'):  # LIF, ALIF, etc.
                if isinstance(m.v, torch.Tensor):
                    if detach:
                        m.v.detach_()
                    else:
                        m.v.zero_()
                else:
                    # Handle scalar voltage values
                    m.v = 0.0
        
        # Reset graph memory if it exists
        if hasattr(self, 'graph_memory') and isinstance(self.graph_memory, torch.Tensor):
            self.graph_memory.zero_()
        
        # Reset edge weights to prevent accumulation
        if hasattr(self, 'edge_weights'):
            self.edge_weights.fill_(0.5)  # Reset to initial values
        
        # Reset STDP traces if they exist
        if hasattr(self, 'stdp_facilitation_traces') and self.stdp_facilitation_traces is not None:
            self.stdp_facilitation_traces.zero_()
        if hasattr(self, 'stdp_inhibition_traces') and self.stdp_inhibition_traces is not None:
            self.stdp_inhibition_traces.zero_()
        
        # Reset momentum tracking
        if hasattr(self, 'agent_momentum') and self.agent_momentum is not None:
            self.agent_momentum.zero_()
        
        # Reset collision history
        if hasattr(self, 'collision_history') and self.collision_history is not None:
            self.collision_history.zero_()
        
    def set_positions(self, positions):
        """
        Set explicit positions for graph building during evaluation
        
        Args:
            positions: Tensor of agent positions [batch, num_agents, 2]
        """
        self.provided_positions = positions
        
    def extract_positions(self, agent_features):
        """
        Extract or estimate agent positions from features for proximity calculation.
        Uses explicitly provided positions if available (especially during evaluation)
        """
        # If positions were explicitly provided (e.g., during evaluation), use them
        if self.provided_positions is not None:
            batch_size = agent_features.size(0)
            # Calculate proximities based on Euclidean distances between provided positions
            positions = self.provided_positions
            # Compute pairwise distances
            expanded_p1 = positions.unsqueeze(2)  # [batch, num_agents, 1, 2]
            expanded_p2 = positions.unsqueeze(1)  # [batch, 1, num_agents, 2]
            distances = torch.norm(expanded_p1 - expanded_p2, dim=3)  # [batch, num_agents, num_agents]
            # Convert distances to proximities (closer = higher value)
            proximities = torch.exp(-distances / self.proximity_threshold)
            return proximities
            
        # Default behavior for training if no positions provided
        batch_size = agent_features.size(0)
        
        # Compute pairwise feature similarities as a proxy for spatial proximity
        similarities = torch.bmm(agent_features, agent_features.transpose(1, 2))
        
        # Normalize to get proximity scores
        proximities = torch.softmax(similarities / (self.hidden_size ** 0.5), dim=-1)
        
        return proximities
        
    def build_dynamic_adjacency_matrix(self, agent_features, batch_size, device, num_agents=None):
        """
        Build dynamic adjacency matrix based on agent proximity and learned edge weights.
        
        Args:
            agent_features: [batch, num_agents, hidden_size]
            batch_size: int
            device: torch device
            num_agents: Optional number of agents (defaults to self.num_agents)
            
        Returns:
            adj_matrix: [batch, num_agents, num_agents] with dynamic connections
            edge_weights: [batch, num_agents, num_agents] learned edge importance
        """
        if num_agents is None:
            num_agents = self.num_agents
            
        # Get proximity-based connections
        proximities = self.extract_positions(agent_features)
        
        # Ensure proximities match the expected number of agents
        actual_num_agents = agent_features.shape[1]
        if proximities.shape[1] != actual_num_agents or proximities.shape[2] != actual_num_agents:
            # Recalculate proximities with correct dimensions
            similarities = torch.bmm(agent_features, agent_features.transpose(1, 2))
            proximities = torch.softmax(similarities / (self.hidden_size ** 0.5), dim=-1)
        
        # Compute pairwise edge weights using learned predictor
        edge_features = []
        for i in range(actual_num_agents):  # Use actual number of agents
            for j in range(actual_num_agents):
                if i != j:
                    # Concatenate features of agent pair
                    pair_features = torch.cat([
                        agent_features[:, i, :], 
                        agent_features[:, j, :]
                    ], dim=-1)
                    edge_features.append(pair_features)
        
        if edge_features:
            edge_features = torch.stack(edge_features, dim=1)  # [batch, num_pairs, hidden*2]
            
            # Reshape for batch norm: [batch * num_pairs, hidden*2] -> [batch * num_pairs, hidden] -> [batch, num_pairs, hidden]
            batch_size, num_pairs, feature_dim = edge_features.shape
            edge_features_flat = edge_features.reshape(-1, feature_dim)  # [batch*num_pairs, hidden*2]
            
            # Apply first linear layer
            edge_hidden = self.edge_weight_fc1(edge_features_flat)  # [batch*num_pairs, hidden]
            
            # Apply batch normalization
            edge_hidden = self.edge_weight_bn(edge_hidden)  # [batch*num_pairs, hidden]
            
            # Apply ReLU activation
            edge_hidden = torch.relu(edge_hidden)
            
            # Apply second linear layer and sigmoid
            edge_weights_flat = torch.sigmoid(self.edge_weight_fc2(edge_hidden))  # [batch*num_pairs, 1]
            
            # Reshape back to original dimensions
            predicted_weights = edge_weights_flat.reshape(batch_size, num_pairs)  # [batch, num_pairs]
            
            # Reshape back to adjacency matrix format
            edge_weights = torch.zeros(batch_size, actual_num_agents, actual_num_agents, device=device)
            idx = 0
            for i in range(actual_num_agents):
                for j in range(actual_num_agents):
                    if i != j:
                        edge_weights[:, i, j] = predicted_weights[:, idx]
                        idx += 1
        else:
            edge_weights = torch.zeros(batch_size, actual_num_agents, actual_num_agents, device=device)
        
        # Combine proximity and learned weights
        combined_weights = proximities * edge_weights
        
        # Create sparse adjacency matrix - keep only top-k connections per agent
        adj_matrix = torch.zeros(batch_size, actual_num_agents, actual_num_agents, device=device)
        
        for i in range(actual_num_agents):  # Use actual number of agents
            # Get top-k most important connections for each agent
            # Ensure k doesn't exceed available connections
            max_k = min(self.max_connections, actual_num_agents - 1)
            if max_k > 0:  # Only if there are other agents to connect to
                # Create mask to exclude self-connection
                agent_weights = combined_weights[:, i, :actual_num_agents].clone()
                agent_weights[:, i] = -float('inf')  # Exclude self-connection
                
                _, top_indices = torch.topk(agent_weights, max_k, dim=-1)
                
                # Set top connections to 1
                for batch_idx in range(batch_size):
                    adj_matrix[batch_idx, i, top_indices[batch_idx]] = 1.0
        
        # Make adjacency symmetric (if agent i connects to j, j connects to i)
        adj_matrix = (adj_matrix + adj_matrix.transpose(-2, -1)) / 2
        
        # Set diagonal to 0 (no self-connections)
        for batch_idx in range(batch_size):
            for i in range(actual_num_agents):
                adj_matrix[batch_idx, i, i] = 0.0
        
        return adj_matrix, combined_weights
        
    def agent_communication(self, agent_features, adj_matrix, edge_weights, num_agents=None):
        """
        Enable agents to communicate with dynamic edge weights and adaptive message passing.
        
        Args:
            agent_features: [batch, num_agents, hidden_size]
            adj_matrix: [batch, num_agents, num_agents] - binary connectivity
            edge_weights: [batch, num_agents, num_agents] - learned importance weights
            num_agents: Optional number of agents (defaults to self.num_agents)
        
        Returns:
            communicated_features: [batch, num_agents, hidden_size]
        """
        if num_agents is None:
            num_agents = self.num_agents
            
        batch_size = agent_features.size(0)
        
        # Generate messages from each agent with batch norm
        messages = self.agent_comm_fc(agent_features.reshape(-1, self.hidden_size))
        messages = self.agent_comm_bn(messages)  # Apply batch norm
        messages = messages.reshape(batch_size, num_agents, self.hidden_size)
        
        # Weight messages by learned edge importance
        weighted_adj = adj_matrix * edge_weights
        
        # Aggregate messages based on weighted graph structure
        aggregated_messages = torch.bmm(weighted_adj, messages)  # [batch, num_agents, hidden_size]
        
        # Process aggregated messages with batch norm
        processed_messages = self.message_agg_fc(aggregated_messages.reshape(-1, self.hidden_size))
        processed_messages = self.message_agg_bn(processed_messages)  # Apply batch norm
        processed_messages = processed_messages.reshape(batch_size, num_agents, self.hidden_size)
        
        return processed_messages
        
    def forward(self, x, stdp_features=None, positions=None, goals=None):
        """
        Process input through dynamic graph-based localized decision making.
        
        Args:
            x: Input features [batch * num_agents, input_dim]
            stdp_features: Optional STDP spatial features [batch * num_agents, hidden_size] for danger computation
            positions: Optional agent positions [batch, num_agents, 2] for true spatial proximity
            
        Returns:
            output: Processed features [batch * num_agents, hidden_size]
        """
        # Reset all spiking neurons
        self.local_decision_lif.reset()
        self.hesitation_lif.reset()
        self.temporal_lif.reset()
        
        # Determine actual number of agents from positions if available, otherwise infer from input
        if positions is not None:
            actual_num_agents = positions.shape[1]
            batch_size = positions.shape[0]
        else:
            # Fallback: try to infer from input size
            total_samples = x.size(0)
            
            # Check if total_samples is divisible by self.num_agents
            if total_samples % self.num_agents == 0:
                actual_num_agents = self.num_agents
                batch_size = total_samples // self.num_agents
            else:
                # Find a better factorization - try common batch sizes
                possible_batch_sizes = [1, 2, 4, 8, 16, 32]
                found = False
                for bs in possible_batch_sizes:
                    if total_samples % bs == 0:
                        batch_size = bs
                        actual_num_agents = total_samples // bs
                        found = True
                        break
                
                if not found:
                    # Last resort: assume batch_size=1
                    batch_size = 1
                    actual_num_agents = total_samples
        
        device = x.device
        
        # Set positions if provided for true Euclidean distance computation
        if positions is not None:
            self.set_positions(positions)
        
        # Apply spatial attention to input
        attended_x = self.spatial_attention(x)
        
        # Project to hidden space with normalized input processing
        input_current = self.input_fc(attended_x)
        input_current = self.input_bn(input_current)  # Apply batch norm
        
        # CRITICAL: Normalize the input current to consistent range for adaptive thresholds
        input_current = F.layer_norm(
            input_current,
            normalized_shape=[input_current.shape[-1]],
            eps=1e-6
        )
        input_current = torch.clamp(input_current, -1.0, 1.0)  # Consistent [-1, 1] range for SNN processing
        
        # Reshape for agent-based processing using actual number of agents
        agent_features = input_current.reshape(batch_size, actual_num_agents, self.hidden_size)
        
        # Apply STDP coordination if positions and goals are provided
        stdp_coordination_metrics = {}
        if positions is not None and goals is not None and self.use_stdp_coordination:
            agent_features, stdp_coordination_metrics = self.apply_stdp_coordination(
                agent_features, positions, goals
            )
        
        # Embed positions into node features if provided
        if positions is not None:
            pos_emb = torch.relu(self.position_encoder(positions))  # [batch, actual_num_agents, hidden_size//4]
            # Pad position embeddings to match feature dimension
            pos_emb_padded = torch.zeros(batch_size, actual_num_agents, self.hidden_size, device=device)
            pos_emb_padded[:, :, :pos_emb.size(-1)] = pos_emb
            # Combine input features with position embeddings
            agent_features = agent_features + pos_emb_padded
        
        # Compute danger scores from STDP features if available
        if stdp_features is not None:
            stdp_reshaped = stdp_features.reshape(batch_size, actual_num_agents, -1)
        
        # Build dynamic communication graph
        adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(agent_features, batch_size, device, actual_num_agents)
        
        spike_accumulator = 0
        hidden_state = torch.zeros_like(agent_features)
        hesitation_state = torch.zeros_like(agent_features)
        
        # Process through multiple time steps for temporal dynamics and adaptive communication
        for t in range(self.time_steps):
            # Update dynamic graph structure based on current state
            if t > 0:
                # Rebuild graph with updated agent features for adaptation
                adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(
                    hidden_state, batch_size, device, actual_num_agents
                )
            
            # Agent-to-agent communication with dynamic weights
            comm_messages = self.agent_communication(
                agent_features.reshape(batch_size, actual_num_agents, -1),
                adj_matrix,
                edge_weights,
                actual_num_agents
            )
            
            # Temporal recurrence from previous state
            if t == 0:
                # First time step: input + communication
                total_current = (agent_features * self.input_weight + 
                               comm_messages * (1 - self.input_weight))
            else:
                # Subsequent time steps: input + communication + recurrence
                recurrent_current = self.recurrent_fc(
                    hidden_state.reshape(-1, self.hidden_size)
                )
                recurrent_current = self.recurrent_bn(recurrent_current)  # Apply batch norm
                recurrent_current = recurrent_current.reshape(batch_size, actual_num_agents, self.hidden_size)
                
                total_current = (agent_features * self.input_weight + 
                               comm_messages * self.comm_message_weight + 
                               recurrent_current * self.recurrence_weight)
            
            # Local decision making
            local_decisions = self.local_decision_lif(
                total_current.reshape(-1, self.hidden_size)
            ).reshape(batch_size, actual_num_agents, self.hidden_size)
            
            # Enhanced hesitation mechanism with learnable uncertainty scaling
            spatial_uncertainty = torch.var(edge_weights, dim=-1, keepdim=True)  # Edge weight uncertainty
            feature_uncertainty = torch.var(total_current, dim=-1, keepdim=True)  # Feature uncertainty
            combined_uncertainty = (spatial_uncertainty + feature_uncertainty).expand(-1, -1, self.hidden_size)
            
            # Apply batch norm before hesitation LIF
            hesitation_input = combined_uncertainty.reshape(-1, self.hidden_size)
            hesitation_input = self.hesitation_bn(hesitation_input) * self.uncertainty_scale
            hesitation_spikes = self.hesitation_lif(hesitation_input)
            hesitation_spikes = hesitation_spikes.reshape(batch_size, actual_num_agents, self.hidden_size)
            
            # Apply learnable hesitation gate threshold with SOFT gating
            hesitation_gate = (hesitation_spikes.mean(dim=-1, keepdim=True) > self.hesitation_gate_threshold).float()
            
            # Soft gate: 70% operation instead of complete pause to prevent neuron death
            hesitation_inhibition = hesitation_gate.expand(-1, -1, self.hidden_size) * 0.3  # 30% inhibition
            inhibited_decisions = local_decisions * (1.0 - hesitation_inhibition)
            inhibited_decisions = torch.relu(inhibited_decisions)  # Ensure non-negative
            
            # Accumulate spikes and update states
            spike_accumulator += inhibited_decisions
            hidden_state = inhibited_decisions
            hesitation_state = hesitation_spikes
        
        # Average spikes over time for rate coding
        avg_spikes = spike_accumulator / self.time_steps
        
        # Preserve batch and agent dimensions: [batch, actual_num_agents, hidden_size]
        batch_size, actual_num_agents, hidden_size = avg_spikes.shape
        
        # Reshape for processing: [batch * actual_num_agents, hidden_size]
        output_features = avg_spikes.reshape(-1, self.hidden_size)
        
        # Final output processing with batch norm
        output = self.output_fc(output_features)
        output = self.output_bn(output)  # Apply batch norm
        
        # Keep the flattened format for compatibility with Network class
        return output

    def reset(self):
        """Reset all stateful components"""
        super().reset()
        self.local_decision_lif.reset()
        self.hesitation_lif.reset()
        self.temporal_lif.reset()
        
        # Reset STDP coordination state
        self.reset_stdp_state()

    # =============================================================================
    # STDP COORDINATION METHODS
    # =============================================================================
    
    def reset_stdp_state(self):
        """Reset all STDP coordination state tracking"""
        if not self.use_stdp_coordination:
            return
            
        # STDP facilitation traces [batch, agents, board_y, board_x, directions]
        self.stdp_facilitation_traces = None
        self.stdp_inhibition_traces = None
        
        # Agent momentum vectors [batch, agents, 2] (x, y components)
        self.agent_momentum = None
        
        # Collision history [batch, agents, board_y, board_x]
        self.collision_history = None
        
        # Previous positions for momentum calculation [batch, agents, 2]
        self.prev_positions = None
        
        # Time step counter
        self.timestep = 0
    
    def initialize_stdp_traces(self, batch_size: int, device: torch.device, num_agents: int = None):
        """Initialize all STDP traces and state tensors"""
        if not self.use_stdp_coordination:
            return
        
        # Use provided num_agents or fall back to configured value
        if num_agents is None:
            num_agents = self.num_agents
            
        # STDP traces: [batch, agents, agents] - pairwise interaction traces
        self.stdp_facilitation_traces = torch.zeros(
            batch_size, num_agents, num_agents,
            device=device, dtype=torch.float32
        )
        self.stdp_inhibition_traces = torch.zeros(
            batch_size, num_agents, num_agents,
            device=device, dtype=torch.float32
        )
        
        # Agent momentum: [batch, agents, 2]
        self.agent_momentum = torch.zeros(
            batch_size, num_agents, 2, device=device, dtype=torch.float32
        )
        
        # Collision history: [batch, agents, board_y, board_x]
        self.collision_history = torch.zeros(
            batch_size, num_agents, self.board_size, self.board_size,
            device=device, dtype=torch.float32
        )
        
        # Previous positions: [batch, agents, 2]
        self.prev_positions = torch.zeros(
            batch_size, num_agents, 2, device=device, dtype=torch.float32
        )
        
    def update_momentum(self, current_positions: torch.Tensor):
        """
        Update agent momentum vectors based on position changes.
        
        Agent-level "momentum vector" tracking:
        agent_momentum[b, a] = decay * agent_momentum[b, a] + (pos_now - pos_prev)
        
        This becomes the agent's intrinsic motion intent - like a neuron's firing history,
        directionally biased.
        
        Args:
            current_positions: [batch, agents, 2] current agent positions
        """
        if not self.use_stdp_coordination:
            return
            
        batch_size, num_agents = current_positions.shape[:2]
        device = current_positions.device
        
        # Initialize or resize agent momentum if needed
        if (self.agent_momentum is None or 
            self.agent_momentum.shape[:2] != (batch_size, num_agents)):
            self.agent_momentum = torch.zeros(
                batch_size, num_agents, 2, device=device, dtype=torch.float32
            )
            
        if self.prev_positions is None:
            self.prev_positions = current_positions.clone()
            return
        
        # Resize prev_positions if needed
        if self.prev_positions.shape[:2] != (batch_size, num_agents):
            self.prev_positions = torch.zeros_like(current_positions)
        
        # Calculate position delta (movement vector)
        position_delta = current_positions - self.prev_positions  # [batch, agents, 2]
        
        # Update momentum with decay (like neuron firing history)
        # This creates intrinsic motion intent that persists over time
        self.agent_momentum = (self.momentum_decay * self.agent_momentum + 
                              (1.0 - self.momentum_decay) * position_delta)
        
        # Normalize momentum to prevent unbounded growth
        momentum_magnitude = torch.norm(self.agent_momentum, dim=-1, keepdim=True)
        max_momentum = 2.0  # Maximum momentum magnitude
        self.agent_momentum = torch.where(
            momentum_magnitude > max_momentum,
            self.agent_momentum * max_momentum / momentum_magnitude,
            self.agent_momentum
        )
        
        # Update previous positions
        self.prev_positions = current_positions.clone()
    
    def compute_goal_alignment(self, positions: torch.Tensor, goals: torch.Tensor, 
                              direction_idx: int) -> torch.Tensor:
        """
        Compute how well a movement direction aligns with the goal vector.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions  
            direction_idx: int, movement direction index
            
        Returns:
            alignment: [batch, agents] alignment score (0-1)
        """
        device = positions.device
        action_delta = self.action_deltas[direction_idx]  # [2]
        
        # Goal vector (normalized)
        goal_vector = goals - positions  # [batch, agents, 2]
        goal_distance = torch.norm(goal_vector, dim=-1, keepdim=True) + 1e-8  # [batch, agents, 1]
        goal_vector_norm = goal_vector / goal_distance  # [batch, agents, 2]
        
        # Action vector (already unit length for grid actions)
        action_vector = action_delta.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        
        # Compute dot product (cosine similarity)
        alignment = torch.sum(goal_vector_norm * action_vector, dim=-1)  # [batch, agents]
        
        # Convert to 0-1 range (from -1 to 1)
        alignment = (alignment + 1.0) / 2.0
        
        return alignment
    
    def compute_region_safety(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute safety of current regions based on collision history.
        
        Args:
            positions: [batch, agents, 2] current positions
            
        Returns:
            safety: [batch, agents] safety score (0-1, 1=safe)
        """
        if not self.use_stdp_coordination or self.collision_history is None:
            # Default to safe if no coordination or history
            return torch.ones(positions.shape[:2], device=positions.device)
            
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        safety_scores = torch.ones(batch_size, num_agents, device=device)
        
        # Get collision history at current positions
        for b in range(batch_size):
            for a in range(num_agents):
                pos = positions[b, a].long()
                # Clamp positions to board bounds
                x = torch.clamp(pos[0], 0, self.board_size - 1)
                y = torch.clamp(pos[1], 0, self.board_size - 1)
                
                # Safety is inverse of collision history (more collisions = less safe)
                collision_rate = self.collision_history[b, a, y, x]
                safety_scores[b, a] = 1.0 - torch.clamp(collision_rate, 0.0, 1.0)
        
        return safety_scores
    
    def update_collision_history(self, positions: torch.Tensor, collisions: torch.Tensor):
        """
        Update collision history based on current collisions.
        
        Args:
            positions: [batch, agents, 2] current positions
            collisions: [batch, agents] boolean tensor indicating collisions
        """
        if not self.use_stdp_coordination or self.collision_history is None:
            return
            
        batch_size, num_agents = positions.shape[:2]
        
        # Decay collision history (non-inplace operation)
        self.collision_history = self.collision_history * (1.0 - 1.0/self.collision_history_window)
        
        # Add new collision events
        for b in range(batch_size):
            for a in range(num_agents):
                if collisions[b, a]:
                    pos = positions[b, a].long()
                    # Clamp positions to board bounds
                    x = torch.clamp(pos[0], 0, self.board_size - 1)
                    y = torch.clamp(pos[1], 0, self.board_size - 1)
                    
                    # Increment collision count at this position (non-inplace operation)
                    increment = torch.zeros_like(self.collision_history)
                    increment[b, a, y, x] = 1.0/self.collision_history_window
                    self.collision_history = self.collision_history + increment
    
    def compute_nearby_momentum_alignment(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute consensus momentum direction from nearby agents within FOV.
        
        FOV + Coordination implementation:
        Each agent:
        1. Looks at nearby agents via FOV (coordination_radius)
        2. Reads their momentum (intrinsic motion intent)  
        3. Checks where they're flowing (consensus direction)
        4. Computes weighted consensus based on distance and alignment
        
        Args:
            positions: [batch, agents, 2] current positions
            
        Returns:
            consensus_momentum: [batch, agents, 2] weighted consensus momentum from FOV
        """
        if not self.use_stdp_coordination or self.agent_momentum is None:
            return torch.zeros_like(positions)
            
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        consensus_momentum = torch.zeros_like(self.agent_momentum)
        
        for b in range(batch_size):
            # Compute pairwise distances (FOV range)
            pos_b = positions[b]  # [agents, 2]
            distances = torch.cdist(pos_b, pos_b, p=2)  # [agents, agents]
            
            # Create FOV mask (agents within coordination radius)
            fov_mask = (distances <= self.coordination_radius) & (distances > 0.1)  # Exclude self
            
            for a in range(num_agents):
                nearby_agents = fov_mask[a]  # [agents] - FOV neighbors
                
                if torch.any(nearby_agents):
                    # Get momentum of nearby agents within FOV
                    nearby_momentum = self.agent_momentum[b, nearby_agents]  # [nearby_count, 2]
                    nearby_distances = distances[a, nearby_agents]  # [nearby_count]
                    
                    # Weight by inverse distance (closer agents have more influence)
                    distance_weights = 1.0 / (nearby_distances + 1e-8)  # [nearby_count]
                    
                    # Weight by momentum alignment (agents moving in similar direction)
                    current_momentum = self.agent_momentum[b, a]  # [2]
                    if torch.norm(current_momentum) > 1e-6:
                        # Compute alignment scores with nearby momentum vectors
                        momentum_dots = torch.sum(nearby_momentum * current_momentum.unsqueeze(0), dim=-1)  # [nearby_count]
                        momentum_magnitudes = torch.norm(nearby_momentum, dim=-1) * torch.norm(current_momentum)
                        alignment_scores = momentum_dots / (momentum_magnitudes + 1e-8)  # [-1, 1]
                        alignment_weights = (alignment_scores + 1.0) / 2.0  # [0, 1]
                    else:
                        alignment_weights = torch.ones_like(distance_weights)
                    
                    # Combine distance and alignment weights
                    combined_weights = distance_weights * alignment_weights
                    combined_weights = combined_weights / (torch.sum(combined_weights) + 1e-8)  # Normalize
                    
                    # Compute weighted consensus momentum (flow direction)
                    consensus_momentum[b, a] = torch.sum(
                        nearby_momentum * combined_weights.unsqueeze(-1), dim=0
                    )
                else:
                    # No nearby agents in FOV, use own momentum
                    consensus_momentum[b, a] = self.agent_momentum[b, a]
        
        return consensus_momentum
    
    def compute_intention_flow_field(self, positions: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """
        Compute intention flow field that combines individual goals with consensus momentum.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions
            
        Returns:
            flow_field: [batch, agents, num_directions] flow field scores for each direction
        """
        if not self.use_stdp_coordination:
            # Default uniform flow if no coordination
            batch_size, num_agents = positions.shape[:2]
            return torch.ones(batch_size, num_agents, self.num_directions, device=positions.device) / self.num_directions
        
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Compute consensus momentum from nearby agents
        consensus_momentum = self.compute_nearby_momentum_alignment(positions)  # [batch, agents, 2]
        
        # Initialize flow field
        flow_field = torch.zeros(batch_size, num_agents, self.num_directions, device=device)
        
        for direction_idx in range(self.num_directions):
            # Goal alignment component
            goal_alignment = self.compute_goal_alignment(positions, goals, direction_idx)  # [batch, agents]
            
            # Momentum alignment component
            action_delta = self.action_deltas[direction_idx]  # [2]
            momentum_alignment = torch.sum(consensus_momentum * action_delta.unsqueeze(0).unsqueeze(0), dim=-1)  # [batch, agents]
            momentum_alignment = torch.clamp((momentum_alignment + 1.0) / 2.0, 0.0, 1.0)  # Normalize to 0-1
            
            # Combine goal and momentum alignment
            combined_flow = (goal_alignment + self.flow_field_strength * momentum_alignment) / (1.0 + self.flow_field_strength)
            flow_field[:, :, direction_idx] = combined_flow
        
        # Normalize flow field to sum to 1 for each agent
        flow_field = F.softmax(flow_field, dim=-1)
        
        return flow_field
    
    def _update_stdp_traces_with_facilitation(self, positions: torch.Tensor, goals: torch.Tensor):
        """
        Update STDP traces with both facilitation and inhibition based on safety and goal alignment.
        
        STDP encodes:
        - Facilitation: Spikes in safe regions with goal-aligned directions
        - Inhibition: Spikes in dangerous areas
        
        Args:
            positions: [batch, agents, 2] current positions  
            goals: [batch, agents, 2] goal positions
        """
        if not self.use_stdp_coordination or self.stdp_facilitation_traces is None:
            return
            
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Decay traces (non-inplace operations)
        self.stdp_facilitation_traces = self.stdp_facilitation_traces * torch.exp(-1.0/self.stdp_tau_facilitation)
        self.stdp_inhibition_traces = self.stdp_inhibition_traces * torch.exp(-1.0/self.stdp_tau_inhibition)
        
        for b in range(batch_size):
            for a in range(num_agents):
                pos = positions[b, a]
                goal = goals[b, a]
                
                # Ensure positions are within board bounds
                x = torch.clamp(pos[0].long(), 0, self.board_size - 1)
                y = torch.clamp(pos[1].long(), 0, self.board_size - 1)
                
                # Compute region safety (low collision history)
                region_safe = self.compute_region_safety(pos.unsqueeze(0).unsqueeze(0))[0, 0] > self.safe_region_threshold
                
                # Update STDP traces for each direction
                for direction_idx in range(self.num_directions):
                    # Check if direction aligns with goal
                    goal_alignment = self.compute_goal_alignment(
                        pos.unsqueeze(0).unsqueeze(0), 
                        goal.unsqueeze(0).unsqueeze(0), 
                        direction_idx
                    )[0, 0]
                    
                    direction_aligned_with_goal = goal_alignment > 0.7  # Threshold for good alignment
                    
                    # STDP Rule Implementation (non-inplace operations):
                    if region_safe and direction_aligned_with_goal:
                        # Facilitation: encourage spikes in safe, goal-aligned directions
                        facilitation_increment = torch.zeros_like(self.stdp_facilitation_traces)
                        facilitation_increment[b, a, y, x, direction_idx] = self.stdp_lr_facilitation
                        self.stdp_facilitation_traces = self.stdp_facilitation_traces + facilitation_increment
                    else:
                        # Inhibition: discourage spikes in dangerous or misaligned directions
                        inhibition_increment = torch.zeros_like(self.stdp_inhibition_traces)
                        inhibition_increment[b, a, y, x, direction_idx] = self.stdp_lr_inhibition
                        self.stdp_inhibition_traces = self.stdp_inhibition_traces + inhibition_increment
                
                # Simulate "spike" for position update (agent moved to this position)
                self.timestep += 1
    
    def compute_intention_flow_field_with_stdp(self, positions: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """
        Enhanced intention flow field that combines STDP traces, momentum, and FOV coordination.
        
        Each agent:
        1. Looks at nearby agents via FOV (coordination radius)
        2. Reads their momentum (intrinsic motion intent)
        3. Checks consensus direction
        4. If flow aligns with STDP facilitated safe zones, spike stronger
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions
            
        Returns:
            flow_field: [batch, agents, num_directions] enhanced flow field scores
        """
        if not self.use_stdp_coordination:
            return self.compute_intention_flow_field(positions, goals)
        
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Get consensus momentum from nearby agents
        consensus_momentum = self.compute_nearby_momentum_alignment(positions)  # [batch, agents, 2]
        
        # Initialize enhanced flow field
        flow_field = torch.zeros(batch_size, num_agents, self.num_directions, device=device)
        
        for direction_idx in range(self.num_directions):
            # 1. Goal alignment component
            goal_alignment = self.compute_goal_alignment(positions, goals, direction_idx)  # [batch, agents]
            
            # 2. Momentum alignment component (consensus direction)
            action_delta = self.action_deltas[direction_idx]  # [2]
            momentum_alignment = torch.sum(consensus_momentum * action_delta.unsqueeze(0).unsqueeze(0), dim=-1)  # [batch, agents]
            momentum_alignment = torch.clamp((momentum_alignment + 1.0) / 2.0, 0.0, 1.0)  # Normalize to 0-1
            
            # 3. STDP facilitation component (safe zone bonus)
            stdp_facilitation_bonus = torch.zeros(batch_size, num_agents, device=device)
            stdp_inhibition_penalty = torch.zeros(batch_size, num_agents, device=device)
            
            if self.stdp_facilitation_traces is not None:
                for b in range(batch_size):
                    for a in range(num_agents):
                        pos = positions[b, a]
                        x = torch.clamp(pos[0].long(), 0, self.board_size - 1)
                        y = torch.clamp(pos[1].long(), 0, self.board_size - 1)
                        
                        # Get STDP trace values for this position and direction
                        facilitation = self.stdp_facilitation_traces[b, a, y, x, direction_idx]
                        inhibition = self.stdp_inhibition_traces[b, a, y, x, direction_idx]
                        
                        stdp_facilitation_bonus[b, a] = facilitation
                        stdp_inhibition_penalty[b, a] = inhibition
            
            # 4. Combine all components with STDP enhancement
            # Base flow: goal + momentum consensus
            base_flow = (goal_alignment + self.flow_field_strength * momentum_alignment) / (1.0 + self.flow_field_strength)
            
            # STDP enhancement: facilitation boosts, inhibition reduces
            stdp_enhancement = stdp_facilitation_bonus - stdp_inhibition_penalty
            
            # Final flow: base + STDP enhancement
            enhanced_flow = base_flow + 0.3 * torch.tanh(stdp_enhancement)  # Bounded STDP influence
            
            flow_field[:, :, direction_idx] = torch.clamp(enhanced_flow, 0.0, 1.0)
        
        # Normalize flow field to sum to 1 for each agent (probability distribution)
        flow_field = F.softmax(flow_field * 2.0, dim=-1)  # Temperature scaling for sharper decisions
        
        return flow_field
    
    def _compute_coordination_metrics(self, original_features: torch.Tensor, 
                                    coordinated_features: torch.Tensor,
                                    momentum_features: torch.Tensor,
                                    flow_features: torch.Tensor, 
                                    flow_field: torch.Tensor,
                                    positions: torch.Tensor, 
                                    goals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for coordination system usage for reward calculation.
        
        Args:
            original_features: Original agent features before coordination
            coordinated_features: Features after coordination enhancement
            momentum_features: Processed momentum features
            flow_features: Processed intention flow features
            flow_field: Raw intention flow field
            positions: Agent positions
            goals: Goal positions
            
        Returns:
            Dictionary of coordination metrics
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        metrics = {}
        
        # 1. Coordination Enhancement Score: how much coordination changed features
        feature_change = torch.norm(coordinated_features - original_features, dim=-1)  # [batch, agents]
        metrics['coordination_enhancement'] = feature_change
        
        # 2. Momentum Utilization: how much momentum information is being used
        momentum_magnitude = torch.norm(momentum_features, dim=-1)  # [batch, agents]
        metrics['momentum_utilization'] = momentum_magnitude
        
        # 3. Flow Field Diversity: how diverse the intention flow is (not all stay)
        flow_entropy = -torch.sum(flow_field * torch.log(flow_field + 1e-8), dim=-1)  # [batch, agents]
        max_entropy = torch.log(torch.tensor(self.num_directions, dtype=torch.float32, device=device))
        normalized_entropy = flow_entropy / max_entropy
        metrics['flow_diversity'] = normalized_entropy
        
        # 4. Goal Alignment: how well actions align with goals
        goal_distances = torch.norm(goals - positions, dim=-1)  # [batch, agents]
        metrics['goal_proximity'] = 1.0 / (1.0 + goal_distances)  # Higher when closer to goal
        
        # 5. Coordination Consensus: agreement with nearby agents
        if self.agent_momentum is not None:
            consensus_momentum = self.compute_nearby_momentum_alignment(positions)
            consensus_strength = torch.norm(consensus_momentum, dim=-1)  # [batch, agents]
            metrics['coordination_consensus'] = consensus_strength
        else:
            metrics['coordination_consensus'] = torch.zeros(batch_size, num_agents, device=device)
        
        return metrics
    
    def compute_coordination_rewards(self, positions: torch.Tensor, goals: torch.Tensor, 
                                   batch_size: int, collided_agents: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for effective STDP coordination usage.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions  
            batch_size: int, batch size
            collided_agents: [batch, agents] boolean tensor marking collided agents (no rewards)
            
        Returns:
            Dictionary containing total_reward and metrics
        """
        if not self.use_stdp_coordination:
            # Get actual number of agents from input
            actual_num_agents = positions.shape[1]
            # Return zero rewards if STDP coordination is disabled
            return {
                'total_reward': torch.zeros(batch_size, actual_num_agents, device=positions.device),
                'metrics': {
                    'coordination_efficiency': 0.0,
                    'goal_proximity_bonus': 0.0,
                    'conflict_resolution_bonus': 0.0
                }
            }
        
        # Get actual number of agents from input
        actual_num_agents = positions.shape[1]
        
        # Initialize STDP traces if needed (use actual number of agents)
        if self.stdp_facilitation_traces is None or self.stdp_facilitation_traces.shape[1] != actual_num_agents:
            self.initialize_stdp_traces(batch_size, positions.device, actual_num_agents)
        
        # Apply STDP coordination and get metrics
        dummy_features = torch.zeros(batch_size, actual_num_agents, self.hidden_size, device=positions.device)
        coordinated_features, coordination_metrics = self.apply_stdp_coordination(
            dummy_features, positions, goals
        )
        
        # Compute basic coordination rewards from metrics
        total_reward = torch.zeros(batch_size, actual_num_agents, device=positions.device)
        
        # Extract metrics and compute rewards
        enhancement_reward = coordination_metrics.get('coordination_enhancement', 
                                                    torch.zeros(batch_size, actual_num_agents, device=positions.device))
        momentum_reward = coordination_metrics.get('momentum_utilization',
                                                 torch.zeros(batch_size, actual_num_agents, device=positions.device))
        diversity_reward = coordination_metrics.get('flow_diversity',
                                                  torch.zeros(batch_size, actual_num_agents, device=positions.device))
        consensus_reward = coordination_metrics.get('coordination_consensus',
                                                  torch.zeros(batch_size, actual_num_agents, device=positions.device))
        
        # Goal proximity bonus
        goal_distances = torch.norm(goals - positions, dim=-1)  # [batch, agents]
        goal_proximity_bonus = 1.0 / (1.0 + goal_distances)  # Higher when closer to goal
        
        # Normalize and combine reward components
        enhancement_reward = torch.tanh(enhancement_reward * 0.1)
        momentum_reward = torch.tanh(momentum_reward * 0.1)  
        consensus_reward = torch.tanh(consensus_reward * 0.1)
        
        # Combine with weights
        total_reward = (
            0.4 * goal_proximity_bonus +           # Primary: goal proximity
            0.25 * enhancement_reward +            # Coordination usage
            0.2 * momentum_reward +                # Momentum consistency
            0.1 * diversity_reward +               # Flow diversity
            0.05 * consensus_reward                # Consensus with neighbors
        )
        
        # IMPORTANT: Zero out rewards for collided agents
        if collided_agents is not None:
            # Expand collided_agents to match reward dimensions if needed
            if collided_agents.dim() == 2:  # [batch, agents]
                collision_mask = ~collided_agents  # Invert: True = no collision, False = collision
                total_reward = total_reward * collision_mask.float()
                
                # Also zero out individual reward components for accurate reporting
                goal_proximity_bonus = goal_proximity_bonus * collision_mask.float()
                enhancement_reward = enhancement_reward * collision_mask.float()
                momentum_reward = momentum_reward * collision_mask.float()
                diversity_reward = diversity_reward * collision_mask.float()
                consensus_reward = consensus_reward * collision_mask.float()
                
                num_collided = torch.sum(collided_agents).item()
                print(f"🚫 Zeroed rewards for {num_collided} collided agents")
        
        # Aggregate metrics for reporting
        aggregated_metrics = {
            'coordination_efficiency': enhancement_reward.flatten().mean().item(),
            'goal_proximity_bonus': goal_proximity_bonus.flatten().mean().item(),
            'conflict_resolution_bonus': consensus_reward.flatten().mean().item()
        }
        
        return {
            'total_reward': total_reward,
            'metrics': aggregated_metrics
        }

    # =============================================================================
    def apply_stdp_coordination(self, features, positions, goals):
        """Apply STDP-based coordination to agent features."""
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Initialize STDP traces if needed - use actual number of agents
        if (self.stdp_facilitation_traces is None or 
            self.stdp_facilitation_traces.shape[1] != num_agents):
            self.initialize_stdp_traces(batch_size, device, num_agents)
        
        # Resize traces if batch size changed
        if self.stdp_facilitation_traces.shape[0] != batch_size:
            current_agents = self.stdp_facilitation_traces.shape[1]
            self.stdp_facilitation_traces = torch.zeros(
                batch_size, current_agents, current_agents, device=device
            )
            self.stdp_inhibition_traces = torch.zeros(
                batch_size, current_agents, current_agents, device=device
            )
        
        # Compute intention flow field
        intention_flow = self.compute_intention_flow_field(positions, goals)
        
        # Update agent momentum
        self.update_momentum(positions)
        
        # Compute pairwise distances for STDP updates
        pos_expanded = positions.unsqueeze(2)  # [batch, agents, 1, 2]
        pos_transposed = positions.unsqueeze(1)  # [batch, 1, agents, 2]
        distances = torch.norm(pos_expanded - pos_transposed, dim=-1)  # [batch, agents, agents]
        
        # Identify agents that have reached their goals
        goal_distances = torch.norm(positions - goals, dim=-1)  # [batch, agents]
        agents_at_goal = goal_distances <= 0.5  # [batch, agents]
        
        # Apply STDP rules based on proximity and coordination needs
        proximity_threshold = self.config.get('proximity_threshold_hardcoded', 3.0)
        close_pairs = distances < proximity_threshold
        
        # Update facilitation traces for close agents
        for agent_i in range(num_agents):
            for agent_j in range(num_agents):
                if agent_i != agent_j:
                    is_close = close_pairs[:, agent_i, agent_j]
                    
                    # Check if agents need coordination (moving toward each other or similar goals)
                    goal_similarity = torch.cosine_similarity(
                        goals[:, agent_i] - positions[:, agent_i],
                        goals[:, agent_j] - positions[:, agent_j],
                        dim=-1
                    )
                    
                    # Enhanced coordination for agents who have reached goals:
                    # They should actively help other agents by providing stronger coordination signals
                    agent_i_at_goal = agents_at_goal[:, agent_i]
                    agent_j_at_goal = agents_at_goal[:, agent_j]
                    
                    # If one agent reached goal, they should provide guidance to others
                    guidance_bonus = (agent_i_at_goal | agent_j_at_goal).float() * 0.5
                    
                    # Update facilitation trace with enhanced coordination for goal-reached agents (non-inplace)
                    coordination_need = is_close & (goal_similarity > 0.3)  # Lowered threshold for more coordination
                    coordination_strength = coordination_need.float() * (0.1 + guidance_bonus)
                    
                    facilitation_update = torch.zeros_like(self.stdp_facilitation_traces)
                    facilitation_update[:, agent_i, agent_j] = coordination_strength
                    self.stdp_facilitation_traces = self.stdp_facilitation_traces + facilitation_update
                    
                    # Update inhibition trace for conflicting movements (non-inplace)
                    # Goal-reached agents should have reduced conflict to help others
                    conflict_base = is_close & (goal_similarity < -0.3)
                    conflict_reduction = (agent_i_at_goal | agent_j_at_goal).float() * 0.5
                    conflict_strength = conflict_base.float() * (0.1 - conflict_reduction).clamp(min=0.01)
                    
                    inhibition_update = torch.zeros_like(self.stdp_inhibition_traces)
                    inhibition_update[:, agent_i, agent_j] = conflict_strength
                    self.stdp_inhibition_traces = self.stdp_inhibition_traces + inhibition_update
        
        # Apply decay to traces (non-inplace operations)
        self.stdp_facilitation_traces = self.stdp_facilitation_traces * 0.95
        self.stdp_inhibition_traces = self.stdp_inhibition_traces * 0.95
        
        # Compute coordination adjustments based on STDP traces (only for actual agents)
        facilitation_strength = self.stdp_facilitation_traces[:, :num_agents, :num_agents].sum(dim=-1, keepdim=True)  # [batch, agents, 1]
        inhibition_strength = self.stdp_inhibition_traces[:, :num_agents, :num_agents].sum(dim=-1, keepdim=True)  # [batch, agents, 1]
        
        # Enhanced coordination factor - goal-reached agents provide stronger influence
        base_coordination_factor = torch.tanh(facilitation_strength - inhibition_strength)
        
        # Amplify coordination effects when interacting with goal-reached agents
        goal_reached_influence = agents_at_goal.unsqueeze(-1).float()  # [batch, agents, 1]
        enhanced_coordination_factor = base_coordination_factor * (1.0 + 0.5 * goal_reached_influence)
        
        # Apply coordination effects to features with enhanced strength
        coordinated_features = features + 0.15 * enhanced_coordination_factor * features  # Increased from 0.1 to 0.15
        
        # Compute coordination metrics
        # Create dummy tensors for missing features
        momentum_features = torch.zeros_like(features)
        flow_features = torch.zeros_like(features)
        
        coordination_metrics = self._compute_coordination_metrics(
            features,           # original_features
            coordinated_features,  # coordinated_features
            momentum_features,  # momentum_features
            flow_features,      # flow_features
            intention_flow,     # flow_field
            positions,          # positions
            goals              # goals
        )
        
        return coordinated_features, coordination_metrics
    
class Network(nn.Module):
    """
    Main SNN Network class that wraps DynamicGraphSNN for MAPF.
    
    This network integrates:
    - Dynamic Graph SNN for localized decision making
    - STDP coordination system with facilitation/inhibition
    - Agent momentum tracking and intention flow fields
    - Transformer encoder for global context (optional)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_agents = config["num_agents"]
        self.map_shape = config["map_shape"]  # FOV size
        self.num_actions = 5
        self.num_classes = 5  # Same as num_actions for compatibility
        self.net_type = config.get("net_type", "snn")
        
        # Calculate input dimensions - determine dynamically from actual data
        self.input_channels = 2  # Based on data loader: (agents, 2, 7, 7) format
        # Map shape from config might not match actual data, so we'll determine this dynamically
        self.map_shape = config["map_shape"]  # FOV size from config
        self.fov_size = self.map_shape[0] * self.map_shape[1]
        self.expected_input_size = self.input_channels * self.fov_size
        
        # Network architecture parameters
        self.hidden_size = config.get("hidden_size", 1280)  # INCREASED from 128 to 1280 (10x) to reduce spike rates
        self.encoder_dims = config.get("encoder_dims", [64])
        
        # SNN OUTPUT LAYER: Use much larger output layer for better spiking stability
        # Instead of 5 neurons (1 per action), use 200 neurons (40 per action class)
        # This enables population coding and more stable spike patterns
        self.output_neurons_per_action = config.get("output_neurons_per_action", 40)  # INCREASED from 20
        self.snn_output_size = self.num_actions * self.output_neurons_per_action  # 5 * 40 = 200
        
        # Use transformer encoder for global context (optional)
        self.use_transformer_encoder = config.get("use_transformer_encoder", False)
        
        # --- NEW: Use the AttentionFeatureExtractor instead of dynamic input processing ---
        # This solves the spatial blindness issue with brain-like attention mechanism
        # Calculate FOV size from input_dim: but the data loader shows (agents, 2, 7, 7) format
        # So we have 2 channels and 7x7 spatial, total flattened = 2 * 7 * 7 = 98 per agent
        fov_side_length = 7  # From data loader: states shape (agents, 2, 7, 7)
        
        self.feature_extractor = AttentionFeatureExtractor(
            input_channels=self.input_channels,  # 2 channels from data loader format
            fov_size=fov_side_length,  # 7x7 FOV from data loader
            output_dim=self.hidden_size  # Output dimension matches hidden_size
        )
        
        # Keep these for compatibility, but they will be used less
        self.input_norm = None
        self.input_projection = None
        self.actual_input_size = None
        
        # Optional transformer encoder for global spatial understanding (initialized dynamically)
        if self.use_transformer_encoder:
            self.transformer_encoder = None  # Will be initialized on first forward pass
        
        # Main SNN processing block with STDP coordination
        self.snn_block = DynamicGraphSNN(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_agents=self.num_agents,
            cfg=config
        )
        
        # Add adaptive LIF nodes for spike regulation in key layers
        self.global_features_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.5),  # Fixed: was 0.01, should be 0.5
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        self.snn_output_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.5),  # Fixed: was 0.01, should be 0.5
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Output processing
        self.output_fc = nn.Linear(self.hidden_size, self.snn_output_size)  # 1280 -> 200 (40 neurons per action)
        
        # Add adaptive LIF for action logits
        self.action_logits_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.01),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Final action classifier: Pool from 200 neurons to 5 action classes
        # Use average pooling to combine spike patterns from multiple neurons per action
        self.action_classifier = nn.Linear(self.snn_output_size, self.num_actions)
        
        # Add output neuron for spiking (only during eval/inference)
        self.output_neuron = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.01),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
    def forward(self, states, return_spikes=False, positions=None, goals=None):
        """
        Forward pass through the SNN network.
        
        Args:
            states: Input states [batch, num_agents, channels, height, width]
            return_spikes: Whether to return spike information
            positions: Optional agent positions [batch, num_agents, 2] for STDP coordination
            goals: Optional goal positions [batch, num_agents, 2] for STDP coordination
            
        Returns:
            action_logits: Action logits [batch, num_agents, num_actions]
            spike_info: Optional spike information if return_spikes=True
        """
        # Handle input shape and process with AttentionFeatureExtractor
        if states.dim() == 5:
            batch_size, num_agents, channels, height, width = states.size()
            
            # --- NORMALIZED INPUT PROCESSING ---
            # Reshape for the feature extractor: [batch * agents, channels, h, w]
            extractor_input = states.reshape(batch_size * num_agents, channels, height, width)
            
            # CRITICAL: Normalize inputs to consistent range [0, 1] for stable adaptive thresholds
            # This ensures all layers receive inputs in the same magnitude range
            extractor_input_normalized = torch.clamp(extractor_input, 0.0, 1.0)  # Ensure [0, 1] range
            extractor_input_normalized = F.layer_norm(
                extractor_input_normalized.reshape(batch_size * num_agents, -1),
                normalized_shape=[extractor_input_normalized.reshape(batch_size * num_agents, -1).shape[-1]],
                eps=1e-6
            ).reshape(batch_size * num_agents, channels, height, width)
            
            # Process the spatial FOV with the Attention module (solves spatial blindness)
            global_features = self.feature_extractor(extractor_input_normalized)
            
        else:
            # Fallback for pre-flattened input - use traditional processing
            # This is a fallback case - ideally input should be 5D for spatial processing
            total_samples = states.size(0)
            actual_input_size = states.size(1)
            
            # Try to infer from goals or positions if provided
            if goals is not None:
                batch_size, num_agents = goals.shape[:2]
                if total_samples != batch_size * num_agents:
                    raise ValueError(f"Input samples {total_samples} doesn't match batch_size*num_agents {batch_size * num_agents}")
            elif positions is not None:
                batch_size, num_agents = positions.shape[:2]
                if total_samples != batch_size * num_agents:
                    raise ValueError(f"Input samples {total_samples} doesn't match batch_size*num_agents {batch_size * num_agents}")
            else:
                # Fall back to configured values - try to infer intelligently
                # Common agent counts: 2, 3, 4, 5, 6, 8, 10
                possible_agents = [2, 3, 4, 5, 6, 8, 10]
                for possible_num in possible_agents:
                    if total_samples % possible_num == 0:
                        num_agents = possible_num
                        batch_size = total_samples // possible_num
                        break
                else:
                    # Default fallback
                    batch_size = total_samples // self.num_agents
                    num_agents = self.num_agents
            
            # Initialize fallback input processing layers if not already done
            if self.input_norm is None or self.actual_input_size != actual_input_size:
                self.actual_input_size = actual_input_size
                self.input_norm = nn.LayerNorm(actual_input_size).to(states.device)
                self.input_projection = nn.Linear(actual_input_size, self.hidden_size).to(states.device)
                
                # Initialize transformer encoder if needed
                if self.use_transformer_encoder:
                    self.transformer_encoder = TransformerEncoder(
                        input_dim=actual_input_size,
                        d_model=self.hidden_size,
                        nhead=self.config.get("transformer_heads", 8),
                        num_layers=self.config.get("transformer_layers", 2),
                        dropout=self.config.get("transformer_dropout", 0.1)
                    ).to(states.device)
            
            # Traditional input processing for fallback case
            # CRITICAL: Normalize inputs to consistent range for stable adaptive thresholds
            x_norm = F.layer_norm(states, normalized_shape=[states.shape[-1]], eps=1e-6)
            x_norm = torch.clamp(x_norm, -2.0, 2.0)  # Reasonable range for layer norm output
            
            # Global context through transformer encoder (optional) or projection
            if self.use_transformer_encoder:
                global_features = self.transformer_encoder(x_norm, num_agents)
            else:
                global_features = self.input_projection(x_norm)
        
        # Process global features through configurable scaling
        # Use config factor to adjust dynamic range before spiking
        gf_scale = config.get('global_features_scale', 1.0)
        global_features = global_features * gf_scale
        # Clip to avoid extreme values and maintain some dynamic range
        global_features = torch.clamp(global_features, -1.0, 1.0)
        
        # Apply adaptive LIF to global features for spike regulation
        global_features = self.global_features_lif(global_features)
        
        # Process through Dynamic Graph SNN with STDP coordination
        snn_output = self.snn_block(
            global_features, 
            stdp_features=None,  # Could be used for additional STDP features
            positions=positions, 
            goals=goals
        )
        
        # CRITICAL: Normalize SNN output before next layer
        snn_output = F.layer_norm(
            snn_output,
            normalized_shape=[snn_output.shape[-1]],
            eps=1e-6
        )
        snn_output = torch.clamp(snn_output, -1.0, 1.0)  # Consistent range
        
        # Apply adaptive LIF to SNN output for spike regulation
        snn_output = self.snn_output_lif(snn_output)
        
        # Generate action logits from SNN output (normalized input)
        action_logits = self.output_fc(snn_output)
        
        # CRITICAL: Normalize action logits before final processing
        action_logits = F.layer_norm(
            action_logits,
            normalized_shape=[action_logits.shape[-1]],
            eps=1e-6
        )
        action_logits = torch.clamp(action_logits, -1.0, 1.0)  # Consistent range
        
        # Apply adaptive LIF to action logits for spike regulation
        action_logits = self.action_logits_lif(action_logits)
        
        # Pool from 200 neurons to 5 action classes using population coding
        # Each action class gets 40 neurons, we average their responses
        action_logits_pooled = action_logits.reshape(-1, self.num_actions, self.output_neurons_per_action)
        action_logits_pooled = action_logits_pooled.mean(dim=2)  # Average across neurons per action
        
        # Final action classification with normalized input
        final_action_logits = self.action_classifier(action_logits)
        
        # CRITICAL: Normalize final output before activation
        final_action_logits = F.layer_norm(
            final_action_logits,
            normalized_shape=[final_action_logits.shape[-1]],
            eps=1e-6
        )
        final_action_logits = torch.clamp(final_action_logits, -2.0, 2.0)  # Reasonable range for final output
        
        # Unsaturate the output: use analogue values during training, spikes during eval
        if self.training:                # no spiking during backward pass
            output_act = final_action_logits * 0.5  # Modest scaling for stable gradients
        else:                            # spiking only in eval/inference
            output_act = self.output_neuron(final_action_logits * 0.5)  # Consistent scaling
        
        # Reshape to agent-wise output using actual number of agents
        output_act = output_act.reshape(batch_size, num_agents, self.num_actions)
        
        if return_spikes:
            spike_info = {
                'global_features': global_features,     # Use the variable that holds the LIF output
                'snn_output': snn_output,               # Use the variable that holds the LIF output
                'action_logits': action_logits          # Use the variable that holds the LIF output
            }
            return output_act, spike_info
        
        return output_act
    
    def reset(self):
        """Reset all memory modules for clean state between episodes"""
        if hasattr(self.snn_block, 'reset'):
            self.snn_block.reset()
        if hasattr(self, 'output_neuron'):
            self.output_neuron.reset()
    
    def reset_state(self, detach=True):
        """
        Reset all hidden states to prevent snowball effect across mini-batches.
        
        This prevents neuron voltages from accumulating across batches, which causes
        membranes to sit above threshold and makes surrogate gradients ≈ 0.
        
        Side-effects: GPU peak memory ↓ 30-40%, runtime ↓ ~7%
        
        Args:
            detach: If True, detach gradients. If False, zero the values.
        """
        if hasattr(self.snn_block, 'reset_state'):
            self.snn_block.reset_state(detach=detach)
        
        # Reset adaptive LIF nodes
        if hasattr(self, 'global_features_lif'):
            self.global_features_lif.reset()
        if hasattr(self, 'snn_output_lif'):
            self.snn_output_lif.reset()
        if hasattr(self, 'action_logits_lif'):
            self.action_logits_lif.reset()
        
        # Reset output neuron if it exists
        if hasattr(self, 'output_neuron') and hasattr(self.output_neuron, 'v'):
            if isinstance(self.output_neuron.v, torch.Tensor):
                if detach:
                    self.output_neuron.v.detach_()
                else:
                    self.output_neuron.v.zero_()
            else:
                # Handle scalar voltage values
                self.output_neuron.v = 0.0
    
    def compute_coordination_rewards(self, positions: torch.Tensor, goals: torch.Tensor, batch_size: int, collided_agents: torch.Tensor = None) -> Dict:
        """
        Delegate coordination rewards computation to the SNN block.
        
        Args:
            positions: Agent positions [batch_size, num_agents, 2]
            goals: Agent goals [batch_size, num_agents, 2]  
            batch_size: Batch size
            collided_agents: [batch, agents] boolean tensor marking collided agents (no rewards)
            
        Returns:
            Dictionary containing total_reward and metrics
        """
        return self.snn_block.compute_coordination_rewards(positions, goals, batch_size, collided_agents)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for global spatial understanding and multi-agent awareness.
    
    Key features:
    - Global spatial context through self-attention
    - Multi-agent awareness and coordination
    - Position encoding for spatial relationships
    - Rich feature representations for downstream SNN
    """
    def __init__(self, input_dim, d_model, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for spatial awareness
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max 100 agents
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, num_agents):
        """
        Process input through transformer encoder for global context.
        
        Args:
            x: Input features [batch * num_agents, input_dim]
            num_agents: Number of agents
            
        Returns:
            encoded_features: [batch * num_agents, d_model] with global context
        """
        batch_size = x.size(0) // num_agents
        
        # Reshape for sequence processing: [batch, num_agents, input_dim]
        x_seq = x.reshape(batch_size, num_agents, self.input_dim)
        
        # Project to model dimension
        x_proj = self.input_projection(x_seq)
        
        # Add positional encoding
        x_pos = x_proj + self.pos_encoding[:, :num_agents, :]
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x_pos)
        
        # Apply layer norm and output projection
        output = self.layer_norm(encoded)
        output = self.output_projection(output)
        
        # Reshape back to flat format: [batch * num_agents, d_model]
        return output.reshape(-1, self.d_model)

