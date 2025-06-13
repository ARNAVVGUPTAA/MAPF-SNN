import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, base
from spikingjelly.activation_based.learning import STDPLearner
from config import config


class AdaptiveLIFNode(neuron.LIFNode):
    """
    LIF neuron that adapts to different batch sizes between train/validation.
    Key improvement: Better parameter initialization for consistent spiking.
    """
    def __init__(self, tau=None, v_threshold=None, v_reset=None, surrogate_alpha=None, detach_reset=None, *args, **kwargs):
        # Use config parameters for better spiking
        super().__init__(
            tau=tau if tau is not None else config.get('lif_tau', 10.0), 
            v_threshold=v_threshold if v_threshold is not None else config.get('lif_v_threshold', 0.5), 
            v_reset=v_reset if v_reset is not None else config.get('lif_v_reset', 0.0),
            detach_reset=detach_reset if detach_reset is not None else config.get('detach_reset', True),
            *args, **kwargs
        )

    def reset(self):
        """Reset to scalar voltage to adapt to new batch sizes"""
        neuron.BaseNode.reset(self)
        self.v = self.v_reset  # Reset to scalar for shape adaptation

    def single_step_forward(self, x):
        # Ensure membrane potential tensor shape matches input x
        if not isinstance(getattr(self, 'v', None), torch.Tensor) or self.v.shape != x.shape:
            self.v = self.v_reset * torch.ones_like(x, device=x.device)
        return super().single_step_forward(x)


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
        
        # Get SNN parameters from global config
        self.time_steps = config.get('snn_time_steps', 3)
        self.input_scale = config.get('input_scale', 2.0)
        self.num_agents = num_agents
        self.hidden_size = hidden_size
        
        # Batch normalization parameters
        bn_momentum = float(cfg.get('batch_norm_momentum', 0.1)) if cfg else 0.1
        bn_eps = float(cfg.get('batch_norm_eps', 1e-5)) if cfg else 1e-5
        
        # Dynamic graph parameters
        self.proximity_threshold = config.get('proximity_threshold', 0.3)  # Threshold for agent proximity
        self.max_connections = config.get('max_connections', 3)  # Maximum connections per agent
        self.edge_learning_rate = config.get('edge_learning_rate', 0.1)  # How fast edges adapt
        
        # Convert hard-coded weights to learnable parameters
        self.recurrence_weight = nn.Parameter(torch.tensor(config.get('recurrence_weight', 0.3)))
        self.input_weight = nn.Parameter(torch.tensor(config.get('input_weight', 0.7)))
        self.comm_message_weight = nn.Parameter(torch.tensor(0.3))  # Learnable communication mixing weight
        
        # Hesitation parameters - controls decision confidence
        self.hesitation_weight = config.get('hesitation_weight', 0.2)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Learnable gate thresholds
        self.conflict_gate_threshold = nn.Parameter(torch.tensor(config.get('conflict_gate_threshold', 0.2)))
        self.hesitation_gate_threshold = nn.Parameter(torch.tensor(config.get('hesitation_gate_threshold', 0.15)))
        
        # Learnable scaling factors
        self.uncertainty_scale = nn.Parameter(torch.tensor(2.0))  # For hesitation input scaling
        self.danger_scale = nn.Parameter(torch.tensor(0.3))  # For danger-aware edge boosting
        
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
            v_threshold=config.get('lif_v_threshold', 0.5),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Enhanced hesitation with spatial awareness
        self.hesitation_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * 1.5,
            v_threshold=config.get('lif_v_threshold', 0.5) * 0.8,
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Spatial conflict detection - considers agent positions
        self.spatial_conflict_detector = nn.Linear(hidden_size * 3, hidden_size)  # Include position info
        self.spatial_conflict_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.conflict_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * 0.5,
            v_threshold=config.get('lif_v_threshold', 0.5) * 0.6,
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Temporal memory with graph adaptation
        self.recurrent_fc = nn.Linear(hidden_size, hidden_size)
        self.recurrent_bn = nn.BatchNorm1d(hidden_size, momentum=bn_momentum, eps=bn_eps)
        self.temporal_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * 1.2,
            v_threshold=config.get('lif_v_threshold', 0.5),
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
        
    def build_dynamic_adjacency_matrix(self, agent_features, danger_score, batch_size, device):
        """
        Build dynamic adjacency matrix based on agent proximity, learned edge weights, and danger awareness.
        
        Args:
            agent_features: [batch, num_agents, hidden_size]
            danger_score: [batch, num_agents] - scalar danger score per agent 
            batch_size: int
            device: torch device
            
        Returns:
            adj_matrix: [batch, num_agents, num_agents] with dynamic connections
            edge_weights: [batch, num_agents, num_agents] learned edge importance
        """
        # Get proximity-based connections
        proximities = self.extract_positions(agent_features)
        
        # Compute pairwise edge weights using learned predictor
        edge_features = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
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
            edge_features_flat = edge_features.view(-1, feature_dim)  # [batch*num_pairs, hidden*2]
            
            # Apply first linear layer
            edge_hidden = self.edge_weight_fc1(edge_features_flat)  # [batch*num_pairs, hidden]
            
            # Apply batch normalization
            edge_hidden = self.edge_weight_bn(edge_hidden)  # [batch*num_pairs, hidden]
            
            # Apply ReLU activation
            edge_hidden = torch.relu(edge_hidden)
            
            # Apply second linear layer and sigmoid
            edge_weights_flat = torch.sigmoid(self.edge_weight_fc2(edge_hidden))  # [batch*num_pairs, 1]
            
            # Reshape back to original dimensions
            predicted_weights = edge_weights_flat.view(batch_size, num_pairs)  # [batch, num_pairs]
            
            # Reshape back to adjacency matrix format
            edge_weights = torch.zeros(batch_size, self.num_agents, self.num_agents, device=device)
            idx = 0
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        edge_weights[:, i, j] = predicted_weights[:, idx]
                        idx += 1
        else:
            edge_weights = torch.zeros(batch_size, self.num_agents, self.num_agents, device=device)
        
        # Danger-aware edge boosting: boost edges near dangerous agents
        # Create pairwise danger scores: average of endpoint dangers
        danger_pair = 0.5 * (danger_score.unsqueeze(2) + danger_score.unsqueeze(1))  # [batch, num_agents, num_agents]
        danger_boost = self.danger_scale * danger_pair  # λ is learnable parameter
        
        # Combine proximity, learned weights, and danger boost
        combined_weights = proximities * edge_weights + danger_boost
        
        # Create sparse adjacency matrix - keep only top-k connections per agent
        adj_matrix = torch.zeros_like(combined_weights)
        
        for i in range(self.num_agents):
            # Get top-k most important connections for each agent
            _, top_indices = torch.topk(combined_weights[:, i, :], 
                                      min(self.max_connections, self.num_agents - 1), 
                                      dim=-1)
            
            # Set top connections to 1
            for batch_idx in range(batch_size):
                adj_matrix[batch_idx, i, top_indices[batch_idx]] = 1.0
        
        # Make adjacency symmetric (if agent i connects to j, j connects to i)
        adj_matrix = (adj_matrix + adj_matrix.transpose(-2, -1)) / 2
        
        # Set diagonal to 0 (no self-connections) - handle batch dimension properly
        for batch_idx in range(batch_size):
            adj_matrix[batch_idx].fill_diagonal_(0)
        
        return adj_matrix, combined_weights
        
    def agent_communication(self, agent_features, adj_matrix, edge_weights):
        """
        Enable agents to communicate with dynamic edge weights and adaptive message passing.
        
        Args:
            agent_features: [batch, num_agents, hidden_size]
            adj_matrix: [batch, num_agents, num_agents] - binary connectivity
            edge_weights: [batch, num_agents, num_agents] - learned importance weights
        
        Returns:
            communicated_features: [batch, num_agents, hidden_size]
        """
        batch_size = agent_features.size(0)
        
        # Generate messages from each agent with batch norm
        messages = self.agent_comm_fc(agent_features.view(-1, self.hidden_size))
        messages = self.agent_comm_bn(messages)  # Apply batch norm
        messages = messages.view(batch_size, self.num_agents, self.hidden_size)
        
        # Weight messages by learned edge importance
        weighted_adj = adj_matrix * edge_weights
        
        # Aggregate messages based on weighted graph structure
        aggregated_messages = torch.bmm(weighted_adj, messages)  # [batch, num_agents, hidden_size]
        
        # Process aggregated messages with batch norm
        processed_messages = self.message_agg_fc(aggregated_messages.view(-1, self.hidden_size))
        processed_messages = self.message_agg_bn(processed_messages)  # Apply batch norm
        processed_messages = processed_messages.view(batch_size, self.num_agents, self.hidden_size)
        
        return processed_messages
        
    def detect_spatial_conflicts(self, current_features, neighbor_features, position_info):
        """
        Enhanced conflict detection that considers spatial relationships.
        
        Args:
            current_features: [batch, num_agents, hidden_size]
            neighbor_features: [batch, num_agents, hidden_size] 
            position_info: [batch, num_agents, hidden_size] - spatial context
            
        Returns:
            conflict_signals: [batch, num_agents, hidden_size]
        """
        # Concatenate current, neighbor, and spatial features for enhanced conflict detection
        conflict_input = torch.cat([current_features, neighbor_features, position_info], dim=-1)
        
        # Detect conflicts through learned spatial patterns with batch norm
        conflict_potential = self.spatial_conflict_detector(conflict_input.view(-1, self.hidden_size * 3))
        conflict_potential = self.spatial_conflict_bn(conflict_potential)  # Apply batch norm
        conflict_potential = conflict_potential.view(current_features.shape)
        
        # Generate conflict signals through spiking
        conflict_signals = self.conflict_lif(conflict_potential * self.input_scale)
        
        return conflict_signals
    
    def compute_danger_score(self, features):
        """
        Derive a scalar danger score per agent from spatial features.
        
        Args:
            features: [batch, num_agents, feature_dim] - spatial features (STDP or hidden state)
            
        Returns:
            danger_score: [batch, num_agents] - normalized danger scores in [0,1]
        """
        # Average across feature dimensions to get raw danger signal
        danger_raw = features.mean(dim=-1)  # [batch, num_agents]
        
        # Normalize to [0,1] range using sigmoid
        danger_score = torch.sigmoid(danger_raw)
        
        return danger_score
        
    def forward(self, x, stdp_features=None, positions=None):
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
        self.conflict_lif.reset()
        self.temporal_lif.reset()
        
        batch_size = x.size(0) // self.num_agents
        device = x.device
        
        # Set positions if provided for true Euclidean distance computation
        if positions is not None:
            self.set_positions(positions)
        
        # Apply spatial attention to input
        attended_x = self.spatial_attention(x)
        
        # Project to hidden space and scale for spiking with batch norm
        input_current = self.input_fc(attended_x)
        input_current = self.input_bn(input_current)  # Apply batch norm
        input_current = input_current * self.input_scale
        
        # Reshape for agent-based processing
        agent_features = input_current.view(batch_size, self.num_agents, self.hidden_size)
        
        # Embed positions into node features if provided
        if positions is not None:
            pos_emb = torch.relu(self.position_encoder(positions))  # [batch, num_agents, hidden_size//4]
            # Pad position embeddings to match feature dimension
            pos_emb_padded = torch.zeros(batch_size, self.num_agents, self.hidden_size, device=device)
            pos_emb_padded[:, :, :pos_emb.size(-1)] = pos_emb
            # Combine input features with position embeddings
            agent_features = agent_features + pos_emb_padded
        
        # Compute danger scores from STDP features if available
        if stdp_features is not None:
            stdp_reshaped = stdp_features.view(batch_size, self.num_agents, -1)
            danger_score = self.compute_danger_score(stdp_reshaped)
        else:
            # Default: no danger boost
            danger_score = torch.zeros(batch_size, self.num_agents, device=device)
        
        # Build dynamic communication graph with danger awareness
        adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(agent_features, danger_score, batch_size, device)
        
        spike_accumulator = 0
        hidden_state = torch.zeros_like(agent_features)
        hesitation_state = torch.zeros_like(agent_features)
        
        # Process through multiple time steps for temporal dynamics and adaptive communication
        for t in range(self.time_steps):
            # Update dynamic graph structure based on current state
            if t > 0:
                # Rebuild graph with updated agent features for adaptation
                # Recompute danger scores from current hidden state
                current_danger = self.compute_danger_score(hidden_state)
                adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(
                    hidden_state, current_danger, batch_size, device
                )
            
            # Agent-to-agent communication with dynamic weights
            comm_messages = self.agent_communication(
                agent_features.view(batch_size, self.num_agents, -1), 
                adj_matrix,
                edge_weights
            )
            
            # Enhanced conflict detection with spatial awareness
            current_state = hidden_state if t > 0 else agent_features
            
            # Create position context (simplified - could be enhanced with actual positions)
            position_context = torch.mean(current_state, dim=-1, keepdim=True).expand(-1, -1, self.hidden_size)
            conflict_signals = self.detect_spatial_conflicts(current_state, comm_messages, position_context)
            
            # Temporal recurrence from previous state
            if t == 0:
                # First time step: input + communication
                total_current = (agent_features * self.input_weight + 
                               comm_messages * (1 - self.input_weight))
            else:
                # Subsequent time steps: input + communication + recurrence
                recurrent_current = self.recurrent_fc(
                    hidden_state.view(-1, self.hidden_size)
                )
                recurrent_current = self.recurrent_bn(recurrent_current)  # Apply batch norm
                recurrent_current = recurrent_current.view(batch_size, self.num_agents, self.hidden_size)
                
                total_current = (agent_features * self.input_weight + 
                               comm_messages * self.comm_message_weight + 
                               recurrent_current * self.recurrence_weight)
            
            # Local decision making with learnable conflict inhibition
            # Use learnable threshold parameter
            conflict_gate = (conflict_signals.mean(dim=-1, keepdim=True) > self.conflict_gate_threshold).float()
            
            # Apply hard gate: 0 = full inhibition, 1 = normal operation
            decision_input = total_current * (1.0 - conflict_gate.expand(-1, -1, self.hidden_size))
            
            local_decisions = self.local_decision_lif(
                decision_input.view(-1, self.hidden_size)
            ).view(batch_size, self.num_agents, self.hidden_size)
            
            # Enhanced hesitation mechanism with learnable uncertainty scaling
            spatial_uncertainty = torch.var(edge_weights, dim=-1, keepdim=True)  # Edge weight uncertainty
            feature_uncertainty = torch.var(total_current, dim=-1, keepdim=True)  # Feature uncertainty
            combined_uncertainty = (spatial_uncertainty + feature_uncertainty).expand(-1, -1, self.hidden_size)
            
            hesitation_spikes = self.hesitation_lif(combined_uncertainty.view(-1, self.hidden_size) * self.uncertainty_scale)
            hesitation_spikes = hesitation_spikes.view(batch_size, self.num_agents, self.hidden_size)
            
            # Apply learnable hesitation gate threshold
            hesitation_gate = (hesitation_spikes.mean(dim=-1, keepdim=True) > self.hesitation_gate_threshold).float()
            
            # Hard gate: 0 = full pause, 1 = normal operation  
            inhibited_decisions = local_decisions * (1.0 - hesitation_gate.expand(-1, -1, self.hidden_size))
            inhibited_decisions = torch.relu(inhibited_decisions)  # Ensure non-negative
            
            # Accumulate spikes and update states
            spike_accumulator += inhibited_decisions
            hidden_state = inhibited_decisions
            hesitation_state = hesitation_spikes
        
        # Average spikes over time for rate coding
        avg_spikes = spike_accumulator / self.time_steps
        
        # Reshape back to original format
        output_features = avg_spikes.view(-1, self.hidden_size)
        
        # Final output processing with batch norm
        output = self.output_fc(output_features)
        output = self.output_bn(output)  # Apply batch norm
        
        return output

    def reset(self):
        """Reset all stateful components"""
        super().reset()
        self.local_decision_lif.reset()
        self.hesitation_lif.reset()
        self.conflict_lif.reset()
        self.temporal_lif.reset()


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
            x: Input features [batch * num_agents, actual_input_dim]
            num_agents: Number of agents
            
        Returns:
            encoded_features: [batch * num_agents, d_model] with global context
        """
        batch_size = x.size(0) // num_agents
        actual_input_dim = x.size(1)  # Get actual input dimension from tensor
        
        # Reshape for sequence processing: [batch, num_agents, actual_input_dim]
        x_seq = x.view(batch_size, num_agents, actual_input_dim)
        
        # Project to model dimension (handle dynamic input size)
        if actual_input_dim != self.input_dim:
            # Create a dynamic projection layer if input size changed
            if not hasattr(self, 'dynamic_projection') or self.dynamic_projection.in_features != actual_input_dim:
                self.dynamic_projection = nn.Linear(actual_input_dim, self.d_model).to(x.device)
            x_proj = self.dynamic_projection(x_seq)  # [batch, num_agents, d_model]
        else:
            x_proj = self.input_projection(x_seq)  # [batch, num_agents, d_model]
        
        # Add positional encoding for spatial awareness
        pos_enc = self.pos_encoding[:, :num_agents, :].expand(batch_size, -1, -1)
        x_pos = x_proj + pos_enc
        
        # Apply layer normalization
        x_norm = self.layer_norm(x_pos)
        
        # Process through transformer encoder for global attention
        encoded = self.transformer_encoder(x_norm)  # [batch, num_agents, d_model]
        
        # Output projection
        output = self.output_projection(encoded)
        
        # Reshape back to original format: [batch * num_agents, d_model]
        output_flat = output.view(-1, self.d_model)
        
        return output_flat


class Network(base.MemoryModule):
    """
    Enhanced SNN for MAPF with Transformer Encoder and Dynamic Graph SNN.
    
    Architecture: Input → Transformer Encoder → Dynamic GraphSNN → SpikingTransformer → Output
    
    Key improvements:
    1. Transformer encoder for global spatial understanding
    2. Dynamic graph SNN for adaptive agent communication
    3. Multi-step temporal processing
    4. Enhanced attention mechanisms for focus
    5. Better parameter tuning for consistent spiking
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config["num_actions"]
        self.num_agents = config.get("num_agents", 5)
        
        # Network dimensions
        input_dim = config["input_dim"] 
        hidden_dim = config["hidden_dim"]
        action_dim = config["num_actions"]
        
        # SNN parameters from config
        self.time_steps = config.get("snn_time_steps", 3)
        self.input_scale = config.get("input_scale", 2.0)
        
        # Enhanced architecture components
        
        # 0. STDP-based spatial danger detection (first layer)
        self.use_stdp_spatial_detector = config.get('use_stdp_spatial_detector', True)
        if self.use_stdp_spatial_detector:
            # Infer input channels from config
            input_channels = config.get('channels', 2)  # Default to 2 channels (agent + obstacles)
            stdp_features = config.get('stdp_feature_channels', 32)
            
            self.stdp_spatial_detector = STDPSpatialDangerDetector(
                input_channels=input_channels,
                feature_channels=stdp_features,
                kernel_size=config.get('stdp_kernel_size', 3),
                cfg=config
            )
            
            # Danger decoder to interpret STDP patterns as actionable danger signals
            self.danger_decoder = DangerDecoder(
                stdp_feature_channels=stdp_features,
                hidden_dim=hidden_dim,
                num_agents=self.num_agents,
                cfg=config
            )
            
            # Calculate enhanced input dimension after STDP processing
            # STDP output: [batch, stdp_features, 9, 9] -> flattened
            stdp_output_dim = stdp_features * 9 * 9
            effective_input_dim = stdp_output_dim
        else:
            self.stdp_spatial_detector = None
            self.danger_decoder = None
            effective_input_dim = input_dim
        
        # 1. Transformer Encoder for global spatial understanding
        self.use_transformer_encoder = config.get('use_transformer_encoder', True)
        if self.use_transformer_encoder:
            self.transformer_encoder = TransformerEncoder(
                input_dim=effective_input_dim,
                d_model=hidden_dim,
                nhead=config.get('encoder_nhead', 8),
                num_layers=config.get('encoder_layers', 2),
                dropout=config.get('encoder_dropout', 0.1)
            )
            encoder_output_dim = hidden_dim
        else:
            # Fallback: simple input processing
            self.input_norm = nn.LayerNorm(effective_input_dim)
            self.input_projection = nn.Linear(effective_input_dim, hidden_dim)
            encoder_output_dim = hidden_dim
        
        # 2. Dynamic Graph SNN for adaptive localized decisions
        self.snn_block = DynamicGraphSNN(
            input_size=encoder_output_dim,
            hidden_size=hidden_dim,
            num_agents=self.num_agents,
            cfg=config
        )
        
        # 3. Optional Spiking Transformer for enhanced attention
        self.use_spiking_transformer = config.get('use_spiking_transformer', False)
        if self.use_spiking_transformer:
            self.spiking_transformer = SpikingTransformer(
                d_model=hidden_dim,
                nhead=config.get('transformer_nhead', 4),
                dim_feedforward=config.get('transformer_ff_dim', hidden_dim * 2)
            )
        
        # 4. Output processing with attention
        self.output_attention = AttentionGate(hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, action_dim)
        
        # Learnable scaling factors for the network
        self.transformer_residual_weight = nn.Parameter(torch.tensor(0.5))  # For transformer output blending
        self.output_scale = nn.Parameter(torch.tensor(2.0))  # For final output neuron scaling
        
        # 5. Final output neuron with config parameters
        self.output_neuron = AdaptiveLIFNode(
            tau=config.get("tau_output", config.get("lif_tau", 10.0) * 0.5),  # Faster dynamics for output
            v_threshold=config.get("v_threshold_output", config.get("lif_v_threshold", 0.5) * 0.3),  # Lower threshold for output spikes
            v_reset=config.get("lif_v_reset", 0.0)
        )

    def forward(self, x, gso=None, return_spikes=False, positions=None):
        """
        Forward pass through the enhanced SNN with STDP Spatial Danger Detection.
        
        Architecture flow:
        Input → STDP Spatial Danger Detection → Transformer Encoder (global context) 
              → Dynamic Graph SNN (local decisions) → Spiking Transformer (attention) → Output
        
        Args:
            x: Input tensor [batch, agents, channels, height, width] or flattened
            gso: Graph structure (not used in this SNN)
            return_spikes: Whether to return spike statistics
            positions: Agent positions for consistent graph building in evaluation
            
        Returns:
            Action probabilities for each agent
        """
        # Reset all stateful components
        self.reset()
        
        # Handle input reshaping
        if x.dim() == 5:
            batch, agents, c, h, w = x.size()
            original_spatial_input = x  # Keep for STDP processing
            x = x.reshape(batch * agents, c * h * w)
            
            # Extract positions from input if not provided
            if positions is None and hasattr(self, 'extract_positions_from_input'):
                positions = self.extract_positions_from_input(x)
        else:
            # Flattened feature input: skip spatial reshape
            batch_agents = x.shape[0]
            original_spatial_input = None
        
        # 0. STDP-based spatial danger detection
        stdp_spatial_features = None  # Store for danger decoder
        if self.use_stdp_spatial_detector and original_spatial_input is not None:
            # Process each agent's FOV through STDP spatial danger detector
            if original_spatial_input.dim() == 5:
                batch, agents, c, h, w = original_spatial_input.size()
                spatial_input = original_spatial_input.reshape(batch * agents, c, h, w)
            else:
                spatial_input = original_spatial_input
            
            # Apply STDP spatial danger detection
            enhanced_spatial_features = self.stdp_spatial_detector(spatial_input)
            stdp_spatial_features = enhanced_spatial_features  # Store for danger decoder
            
            # Flatten enhanced features for subsequent processing
            stdp_features = enhanced_spatial_features.view(enhanced_spatial_features.size(0), -1)
            x = stdp_features  # Replace original flattened input with STDP-enhanced features
        
        # 1. Global spatial understanding through Transformer Encoder
        if self.use_transformer_encoder:
            # Process through transformer encoder for global context and multi-agent awareness
            global_features = self.transformer_encoder(x, self.num_agents)
        else:
            # Fallback: simple input processing with normalization
            x_norm = self.input_norm(x)
            global_features = self.input_projection(x_norm)
            
        # Pass positions to SNN block if available
        if positions is not None and hasattr(self.snn_block, 'set_positions'):
            self.snn_block.set_positions(positions)
        
        # Scale input to encourage spiking - use config value
        global_features = global_features * self.input_scale
        
        # 2. Adaptive localized decision making through Dynamic Graph SNN with danger awareness
        snn_output = self.snn_block(global_features, stdp_features=stdp_spatial_features, positions=positions)
        
        # 3. Optional enhanced attention through Spiking Transformer
        if self.use_spiking_transformer:
            transformer_output = self.spiking_transformer(snn_output)
            snn_output = snn_output + transformer_output * self.transformer_residual_weight  # Learnable residual connection
        
        # 4. Output attention for action selection focus
        attended_output = self.output_attention(snn_output)
        
        # 5. Generate action logits
        action_logits = self.output_fc(attended_output)
        
        # 6. Apply danger gates from STDP spatial features (if available)
        danger_info = {}
        if self.use_stdp_spatial_detector and stdp_spatial_features is not None and self.danger_decoder is not None:
            # Apply danger gating to inhibit risky actions based on STDP-learned spatial patterns
            action_logits, danger_info = self.danger_decoder(stdp_spatial_features, action_logits)
        
        # 7. Final spiking output with learnable scaling
        output_spikes = self.output_neuron(action_logits * self.output_scale)  # Learnable scaling for output spikes
        
        # Ensure output shape is correct
        if output_spikes.dim() != 2 or output_spikes.shape[-1] != self.num_classes:
            raise ValueError(f"Unexpected output shape: {output_spikes.shape}")
        
        if return_spikes:
            # Return spike statistics for analysis
            spike_info = {
                'global_features': global_features,
                'snn_block_spikes': snn_output,
                'output_spikes': output_spikes,
                'attended_output': attended_output
            }
            # Add danger information if available
            if danger_info:
                spike_info['danger_info'] = danger_info
            return output_spikes, spike_info
        
        return output_spikes

    def reset(self):
        """Reset all memory modules for clean state between episodes"""
        super().reset()
        if hasattr(self, 'stdp_spatial_detector') and self.stdp_spatial_detector is not None:
            self.stdp_spatial_detector.reset()
        if hasattr(self, 'spiking_transformer') and self.use_spiking_transformer:
            self.spiking_transformer.reset()
        # Transformer encoder doesn't need reset as it's stateless
        
    def get_attention_weights(self, x):
        """
        Debug method to visualize attention weights.
        Useful for understanding what the network focuses on.
        """
        if x.dim() == 5:
            batch, agents, c, h, w = x.size()
            x = x.reshape(batch * agents, c * h * w)
        
        if self.use_transformer_encoder:
            # For transformer encoder, return global features
            global_features = self.transformer_encoder(x, self.num_agents)
            return {
                'global_features': global_features,
                'transformer_encoded': True
            }
        else:
            # For simple input processing
            x_norm = self.input_norm(x)
            projected = self.input_projection(x_norm)
            
            # Get attention weights from spatial attention in SNN block
            spatial_attention_weights = torch.sigmoid(
                self.snn_block.spatial_attention.attention_fc(projected)
            )
            
            return {
                'spatial_attention': spatial_attention_weights,
                'projected_features': projected
            }


class SpikingTransformer(base.MemoryModule):
    """
    Simplified Spiking Transformer (Resformer) for SNNs.
    Uses spiking neurons for attention computation and feedforward layers.
    Much simpler than full transformer - focuses on core attention mechanism.
    """
    def __init__(self, d_model, nhead=4, dim_feedforward=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        dim_feedforward = dim_feedforward or d_model * 2
        
        # Batch normalization parameters - get from config if available
        bn_momentum = float(config.get('batch_norm_momentum', 0.1))
        bn_eps = float(config.get('batch_norm_eps', 1e-5))
        
        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        
        # Learnable scaling factors for LIF neurons
        self.attn_tau_scale = nn.Parameter(torch.tensor(0.5))  # Faster for attention
        self.attn_threshold_scale = nn.Parameter(torch.tensor(0.7))  # Lower threshold
        self.ff_input_scale = nn.Parameter(torch.tensor(config.get('input_scale', 2.0)))  # Input scaling for FF
        self.attn_input_scale = nn.Parameter(torch.tensor(config.get('input_scale', 2.0)))  # Input scaling for attention
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_proj_bn = nn.BatchNorm1d(d_model, momentum=bn_momentum, eps=bn_eps)
        
        # Spiking neurons for attention processing
        self.attn_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * 0.5,  # Use fixed values initially  
            v_threshold=config.get('lif_v_threshold', 0.5) * 0.7,
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Feedforward network with spiking and batch norm
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.BatchNorm1d(dim_feedforward, momentum=bn_momentum, eps=bn_eps),
            AdaptiveLIFNode(
                tau=config.get('lif_tau', 10.0),
                v_threshold=config.get('lif_v_threshold', 0.5),
                v_reset=config.get('lif_v_reset', 0.0)
            ),
            nn.Linear(dim_feedforward, d_model),
            nn.BatchNorm1d(d_model, momentum=bn_momentum, eps=bn_eps)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _update_lif_parameters(self):
        """Update LIF neuron parameters based on learnable scaling factors"""
        base_tau = config.get('lif_tau', 10.0)
        base_threshold = config.get('lif_v_threshold', 0.5)
        self.attn_lif.tau = base_tau * self.attn_tau_scale
        self.attn_lif.v_threshold = base_threshold * self.attn_threshold_scale
        
    def forward(self, x):
        """
        Simple spiking self-attention.
        x: [batch_size, seq_len, d_model] or [batch_size, d_model] for single token
        """
        # Update LIF parameters with current learnable values
        self._update_lif_parameters()
        
        # Handle single token case
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_model]
            single_token = True
        else:
            single_token = False
            
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual connection
        attn_input = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(attn_input).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(attn_input).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(attn_input).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Transpose for attention computation: [batch, nhead, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with proper masking for evaluation consistency
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply consistent attention masking if needed (to ensure deterministic behavior)
        # This ensures same attention patterns during both training and evaluation
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output.view(-1, d_model))
        attn_output = self.out_proj_bn(attn_output)  # Apply batch norm
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Pass through spiking neuron and add residual with learnable scaling
        spiking_attn = self.attn_lif(attn_output * self.attn_input_scale)
        x = x + spiking_attn
        
        # Feedforward with residual connection and learnable scaling
        ff_input = self.norm2(x)
        ff_output = self.ff_network(ff_input.view(-1, d_model) * self.ff_input_scale)
        ff_output = ff_output.view(batch_size, seq_len, d_model)
        x = x + ff_output
        
        # Return to original shape if needed
        if single_token:
            x = x.squeeze(1)  # [batch, d_model]
            
        return x

    def reset(self):
        """Reset spiking neuron states"""
        super().reset()  # Call parent MemoryModule reset
        self.attn_lif.reset()
        for module in self.ff_network:
            if hasattr(module, 'reset'):
                module.reset()


class STDPSpatialDangerDetector(base.MemoryModule):
    """
    STDP-based spatial danger detection layer for learning obstacles, traps, and tight corridors.
    
    This layer processes field-of-view input to learn spike patterns for spatial dangers through
    unsupervised STDP learning, then provides enhanced features to higher layers.
    
    Architecture:
    - Conv2d layers for spatial feature extraction
    - STDPLearner for unsupervised learning of danger patterns
    - Spiking neurons for temporal processing
    """
    
    def __init__(self, input_channels, feature_channels=32, kernel_size=3, cfg=None):
        super().__init__()
        
        # Get STDP parameters from config (ensure numeric types)
        self.tau_pre = float(cfg.get('stdp_tau_pre', 20.0)) if cfg else 20.0
        self.tau_post = float(cfg.get('stdp_tau_post', 20.0)) if cfg else 20.0
        self.learning_rate_pre = float(cfg.get('stdp_lr_pre', 1e-4)) if cfg else 1e-4
        self.learning_rate_post = float(cfg.get('stdp_lr_post', 1e-4)) if cfg else 1e-4
        self.stdp_enabled = bool(cfg.get('enable_stdp_learning', True)) if cfg else True
        
        # Learnable scaling factors for LIF neurons
        self.tau_scale_1 = nn.Parameter(torch.tensor(0.8))  # For spatial detection speed
        self.tau_scale_2 = nn.Parameter(torch.tensor(0.8))
        self.threshold_scale_1 = nn.Parameter(torch.tensor(0.7))  # For sensitivity
        self.threshold_scale_2 = nn.Parameter(torch.tensor(0.7))
        
        # Spatial feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, feature_channels, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        
        # Store base parameters for dynamic LIF configuration
        self.base_tau = (cfg.get('lif_tau', 10.0) if cfg else 10.0)
        self.base_threshold = (cfg.get('lif_v_threshold', 0.5) if cfg else 0.5)
        self.base_reset = cfg.get('lif_v_reset', 0.0) if cfg else 0.0
        
        # Spiking neurons for each conv layer - will be dynamically configured
        self.sn1 = AdaptiveLIFNode(
            tau=self.base_tau * 0.8,  # Use fixed values initially
            v_threshold=self.base_threshold * 0.7,
            v_reset=self.base_reset
        )
        
        self.sn2 = AdaptiveLIFNode(
            tau=self.base_tau * 0.8,
            v_threshold=self.base_threshold * 0.7,
            v_reset=self.base_reset
        )
        
        # Initialize STDP learners for unsupervised learning
        if self.stdp_enabled:
            # STDP for first layer - learns basic spatial patterns
            self.stdp1 = STDPLearner(
                step_mode='s',
                synapse=self.conv1,
                sn=self.sn1,
                tau_pre=self.tau_pre,
                tau_post=self.tau_post,
                f_pre=lambda w: torch.full_like(w, self.learning_rate_pre),  # constant pre-learning rate tensor
                f_post=lambda w: torch.full_like(w, self.learning_rate_post)  # constant post-learning rate tensor
            )
            
            # STDP for second layer - learns complex danger patterns
            self.stdp2 = STDPLearner(
                step_mode='s',
                synapse=self.conv2,
                sn=self.sn2,
                tau_pre=self.tau_pre * 1.5,
                tau_post=self.tau_post * 1.5,
                f_pre=lambda w: torch.full_like(w, self.learning_rate_pre * 0.8),  # constant pre-learning rate tensor
                f_post=lambda w: torch.full_like(w, self.learning_rate_post * 0.8)  # constant post-learning rate tensor
            )
        else:
            self.stdp1 = None
            self.stdp2 = None
        
        # Output processing to provide enhanced features
        self.feature_aggregator = nn.Conv2d(feature_channels, feature_channels, 
                                          kernel_size=1, bias=True)
        self.danger_threshold = float(cfg.get('danger_threshold', 0.3)) if cfg else 0.3
        
        # Adaptive pooling to provide consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((9, 9))  # Match expected FOV size

    def _update_lif_parameters(self):
        """Update LIF neuron parameters based on learnable scaling factors"""
        # Update tau and threshold dynamically using learnable parameters
        self.sn1.tau = self.base_tau * self.tau_scale_1
        self.sn1.v_threshold = self.base_threshold * self.threshold_scale_1
        self.sn2.tau = self.base_tau * self.tau_scale_2
        self.sn2.v_threshold = self.base_threshold * self.threshold_scale_2
        
    def forward(self, x):
        """
        Process input through STDP-based spatial danger detection.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Enhanced spatial features with learned danger patterns
        """
        # Update LIF parameters with current learnable values
        self._update_lif_parameters()
        
        # Reset spiking neurons
        self.sn1.reset()
        self.sn2.reset()
        
        # First layer: basic spatial feature extraction with STDP
        # Apply synapse (conv) and spiking neuron separately
        conv1_out = self.conv1(x)
        spikes1 = self.sn1(conv1_out)
        
        # Apply STDP learning during training (if enabled)
        if self.stdp_enabled and self.training and self.stdp1 is not None:
            # STDP learning step - updates weights based on pre/post spike timing
            self.stdp1.step(on_grad=True)
        
        # Second layer: complex pattern detection with STDP
        # Apply synapse (conv) and spiking neuron separately
        conv2_out = self.conv2(spikes1)
        spikes2 = self.sn2(conv2_out)
        
        # Apply STDP learning during training (if enabled)
        if self.stdp_enabled and self.training and self.stdp2 is not None:
            # STDP learning step - updates weights based on pre/post spike timing
            self.stdp2.step(on_grad=True)
        
        # Aggregate features to highlight learned danger patterns
        enhanced_features = self.feature_aggregator(spikes2)
        
        # Ensure consistent output size
        enhanced_features = self.adaptive_pool(enhanced_features)
        
        return enhanced_features
    
    def reset(self):
        """Reset all stateful components"""
        super().reset()
        self.sn1.reset()
        self.sn2.reset()
        if hasattr(self.stdp1, 'reset') and self.stdp1 is not None:
            self.stdp1.reset()
        if hasattr(self.stdp2, 'reset') and self.stdp2 is not None:
            self.stdp2.reset()
    
    def set_stdp_learning(self, enabled):
        """Enable/disable STDP learning"""
        self.stdp_enabled = enabled


class DangerDecoder(nn.Module):
    """
    Danger decoder module that interprets STDP spatial features as actionable danger signals.
    
    This module bridges the gap between unsupervised STDP spatial learning and supervised
    action selection by:
    1. Analyzing STDP spike patterns to detect danger signals
    2. Computing global and per-agent danger levels  
    3. Applying danger gates to inhibit risky actions
    
    The key insight is that STDP learns spatial danger patterns unsupervised, but we need
    a supervised module to interpret these patterns and influence action decisions.
    """
    
    def __init__(self, stdp_feature_channels=32, hidden_dim=128, num_agents=5, cfg=None):
        super().__init__()
        
        # STDP spatial features are [batch*agents, stdp_feature_channels, 9, 9]
        self.stdp_input_dim = stdp_feature_channels * 9 * 9
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Get danger parameters from config
        self.danger_threshold = float(cfg.get('danger_threshold', 0.3)) if cfg else 0.3
        self.danger_gate_strength = float(cfg.get('danger_gate_strength', 0.8)) if cfg else 0.8
        self.enable_danger_inhibition = bool(cfg.get('enable_danger_inhibition', True)) if cfg else True
        
        # Danger pattern analysis layers
        self.danger_analyzer = nn.Sequential(
            nn.Linear(self.stdp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single danger score per agent
        )
        
        # Global danger aggregation across agents
        self.global_danger_aggregator = nn.Sequential(
            nn.Linear(num_agents, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Global danger level
        )
        
        # Danger gate computation
        self.danger_gate_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # Individual + global danger
            nn.ReLU(), 
            nn.Linear(hidden_dim // 4, 1),  # Gate strength
            nn.Sigmoid()  # Gate value between 0 and 1
        )
        
        # Learnable danger sensitivity parameters
        self.register_parameter('danger_sensitivity', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('gate_bias', nn.Parameter(torch.tensor(0.1)))
        
    def forward(self, stdp_features, action_logits):
        """
        Decode danger signals from STDP features and apply danger gates to action logits.
        
        Args:
            stdp_features: STDP spatial features [batch*agents, feature_channels, 9, 9]
            action_logits: Action logits before danger gating [batch*agents, num_actions]
            
        Returns:
            gated_action_logits: Action logits with danger inhibition applied
            danger_info: Dictionary with danger analysis information
        """
        if not self.enable_danger_inhibition:
            return action_logits, {'danger_gates': torch.ones_like(action_logits[:, 0:1])}
        
        batch_agents = stdp_features.size(0)
        batch_size = batch_agents // self.num_agents
        
        # Flatten STDP spatial features for analysis
        flattened_stdp = stdp_features.view(batch_agents, -1)  # [batch*agents, stdp_input_dim]
        
        # 1. Analyze individual agent danger levels from STDP patterns
        individual_danger_scores = self.danger_analyzer(flattened_stdp)  # [batch*agents, 1]
        individual_danger_scores = torch.sigmoid(individual_danger_scores) * self.danger_sensitivity
        
        # 2. Compute global danger level across all agents
        # Reshape to group by batch for global analysis
        individual_reshaped = individual_danger_scores.view(batch_size, self.num_agents)  # [batch, agents]
        global_danger_scores = self.global_danger_aggregator(individual_reshaped)  # [batch, 1]
        
        # Broadcast global danger back to individual agents
        global_danger_broadcast = global_danger_scores.repeat(1, self.num_agents).view(batch_agents, 1)  # [batch*agents, 1]
        
        # 3. Compute danger gates combining individual and global danger
        combined_danger = torch.cat([individual_danger_scores, global_danger_broadcast], dim=1)  # [batch*agents, 2]
        danger_gates = self.danger_gate_fc(combined_danger)  # [batch*agents, 1]
        
        # Add learnable bias to prevent complete action blocking
        danger_gates = danger_gates + self.gate_bias
        danger_gates = torch.clamp(danger_gates, 0.0, 1.0)
        
        # 4. Apply danger gates to action logits
        # Gate strength determines how much actions are inhibited in dangerous situations
        # danger_gate close to 0 = high danger, strongly inhibit actions
        # danger_gate close to 1 = low danger, allow normal actions
        gate_multiplier = 1.0 - (1.0 - danger_gates) * self.danger_gate_strength
        gated_action_logits = action_logits * gate_multiplier
        
        # Collect danger information for analysis/debugging
        danger_info = {
            'individual_danger_scores': individual_danger_scores,
            'global_danger_scores': global_danger_scores,
            'danger_gates': danger_gates,
            'gate_multiplier': gate_multiplier,
            'mean_danger_level': individual_danger_scores.mean().item(),
            'max_danger_level': individual_danger_scores.max().item()
        }
        
        return gated_action_logits, danger_info
    
    def get_danger_analysis(self, stdp_features):
        """
        Get detailed danger analysis without affecting action logits.
        Useful for visualization and debugging.
        """
        with torch.no_grad():
            batch_agents = stdp_features.size(0)
            batch_size = batch_agents // self.num_agents
            
            flattened_stdp = stdp_features.view(batch_agents, -1)
            individual_danger_scores = torch.sigmoid(self.danger_analyzer(flattened_stdp)) * self.danger_sensitivity
            
            individual_reshaped = individual_danger_scores.view(batch_size, self.num_agents)
            global_danger_scores = self.global_danger_aggregator(individual_reshaped)
            
            return {
                'individual_danger_scores': individual_danger_scores,
                'global_danger_scores': global_danger_scores,
                'danger_threshold': self.danger_threshold,
                'mean_danger': individual_danger_scores.mean().item(),
                'agents_in_danger': (individual_danger_scores > self.danger_threshold).sum().item()
            }

