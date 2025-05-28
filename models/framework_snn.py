import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, base
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


class AttentionGate(nn.Module):
    """
    Simple attention mechanism to focus on important spatial/feature locations.
    This helps the SNN pay attention to relevant parts of the input.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention_fc = nn.Linear(input_dim, input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Compute attention weights
        attention_weights = torch.sigmoid(self.attention_fc(x))
        # Compute gating values  
        gate_values = torch.tanh(self.gate_fc(x))
        # Apply attention: element-wise multiplication
        return x + (attention_weights * gate_values * 0.5)  # Residual + attended features


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
        
        # Recurrence weights for temporal dynamics
        self.recurrence_weight = config.get('recurrence_weight', 0.3)
        self.input_weight = config.get('input_weight', 0.7)
        
        # Hesitation parameters - controls decision confidence
        self.hesitation_weight = config.get('hesitation_weight', 0.2)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
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
        
        # Learnable edge weights (will be updated dynamically)
        self.register_buffer('edge_weights', torch.ones(num_agents, num_agents) * 0.5)
        
    def extract_positions(self, agent_features):
        """
        Extract or estimate agent positions from features for proximity calculation.
        This is a simplified version - in practice, positions might be explicitly provided.
        """
        batch_size = agent_features.size(0)
        
        # For now, use feature similarity as a proxy for spatial proximity
        # In a real implementation, you'd extract actual positions from the input
        
        # Compute pairwise feature similarities
        similarities = torch.bmm(agent_features, agent_features.transpose(1, 2))
        
        # Normalize to get proximity scores
        proximities = torch.softmax(similarities / (self.hidden_size ** 0.5), dim=-1)
        
        return proximities
        
    def build_dynamic_adjacency_matrix(self, agent_features, batch_size, device):
        """
        Build dynamic adjacency matrix based on agent proximity and learned edge weights.
        
        Args:
            agent_features: [batch, num_agents, hidden_size]
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
        
        # Combine proximity and learned weights
        combined_weights = proximities * edge_weights
        
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
        
    def forward(self, x):
        """
        Process input through dynamic graph-based localized decision making.
        
        Args:
            x: Input features [batch * num_agents, input_size]
            
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
        
        # Apply spatial attention to input
        attended_x = self.spatial_attention(x)
        
        # Project to hidden space and scale for spiking with batch norm
        input_current = self.input_fc(attended_x)
        input_current = self.input_bn(input_current)  # Apply batch norm
        input_current = input_current * self.input_scale
        
        # Reshape for agent-based processing
        agent_features = input_current.view(batch_size, self.num_agents, self.hidden_size)
        
        # Build dynamic communication graph based on current agent states
        adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(agent_features, batch_size, device)
        
        spike_accumulator = 0
        hidden_state = torch.zeros_like(agent_features)
        hesitation_state = torch.zeros_like(agent_features)
        
        # Process through multiple time steps for temporal dynamics and adaptive communication
        for t in range(self.time_steps):
            # Update dynamic graph structure based on current state
            if t > 0:
                # Rebuild graph with updated agent features for adaptation
                adj_matrix, edge_weights = self.build_dynamic_adjacency_matrix(
                    hidden_state, batch_size, device
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
                               comm_messages * 0.3 + 
                               recurrent_current * self.recurrence_weight)
            
            # Local decision making with enhanced conflict awareness
            decision_input = total_current - conflict_signals * 0.5  # Conflicts inhibit decisions
            local_decisions = self.local_decision_lif(
                decision_input.view(-1, self.hidden_size)
            ).view(batch_size, self.num_agents, self.hidden_size)
            
            # Enhanced hesitation mechanism with spatial uncertainty
            spatial_uncertainty = torch.var(edge_weights, dim=-1, keepdim=True)  # Edge weight uncertainty
            feature_uncertainty = torch.var(total_current, dim=-1, keepdim=True)  # Feature uncertainty
            combined_uncertainty = (spatial_uncertainty + feature_uncertainty).expand(-1, -1, self.hidden_size)
            
            hesitation_spikes = self.hesitation_lif(combined_uncertainty.view(-1, self.hidden_size) * 2.0)
            hesitation_spikes = hesitation_spikes.view(batch_size, self.num_agents, self.hidden_size)
            
            # Apply hesitation as inhibition to decisions
            hesitation_inhibition = hesitation_spikes * self.hesitation_weight
            inhibited_decisions = local_decisions - hesitation_inhibition
            inhibited_decisions = torch.relu(inhibited_decisions)  # Ensure non-negative
            
            # Accumulate spikes and update states
            spike_accumulator += inhibited_decisions
            hidden_state = inhibited_decisions
            hesitation_state = hesitation_spikes
        
        # Average spikes over time steps for rate coding
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
            x: Input features [batch * num_agents, input_dim]
            num_agents: Number of agents
            
        Returns:
            encoded_features: [batch * num_agents, d_model] with global context
        """
        batch_size = x.size(0) // num_agents
        
        # Reshape for sequence processing: [batch, num_agents, input_dim]
        x_seq = x.view(batch_size, num_agents, self.input_dim)
        
        # Project to model dimension
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
        
        # 1. Transformer Encoder for global spatial understanding
        self.use_transformer_encoder = config.get('use_transformer_encoder', True)
        if self.use_transformer_encoder:
            self.transformer_encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=config.get('encoder_nhead', 8),
                num_layers=config.get('encoder_layers', 2),
                dropout=config.get('encoder_dropout', 0.1)
            )
            encoder_output_dim = hidden_dim
        else:
            # Fallback: simple input processing
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_projection = nn.Linear(input_dim, hidden_dim)
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
        
        # 5. Final output neuron with config parameters
        self.output_neuron = AdaptiveLIFNode(
            tau=config.get("tau_output", config.get("lif_tau", 10.0) * 0.5),  # Faster dynamics for output
            v_threshold=config.get("v_threshold_output", config.get("lif_v_threshold", 0.5) * 0.3),  # Lower threshold for output spikes
            v_reset=config.get("lif_v_reset", 0.0)
        )

    def forward(self, x, gso=None, return_spikes=False):
        """
        Forward pass through the enhanced SNN with Transformer Encoder and Dynamic Graph.
        
        Architecture flow:
        Input → Transformer Encoder (global context) → Dynamic Graph SNN (local decisions) 
              → Spiking Transformer (attention) → Output
        
        Args:
            x: Input tensor [batch, agents, channels, height, width] or flattened
            gso: Graph structure (not used in this SNN)
            return_spikes: Whether to return spike statistics
            
        Returns:
            Action probabilities for each agent
        """
        # Reset all stateful components
        self.reset()
        
        # Handle input reshaping
        if x.dim() == 5:
            batch, agents, c, h, w = x.size()
            x = x.reshape(batch * agents, c * h * w)
        
        # 1. Global spatial understanding through Transformer Encoder
        if self.use_transformer_encoder:
            # Process through transformer encoder for global context and multi-agent awareness
            global_features = self.transformer_encoder(x, self.num_agents)
        else:
            # Fallback: simple input processing with normalization
            x_norm = self.input_norm(x)
            global_features = self.input_projection(x_norm)
        
        # Scale input to encourage spiking - use config value
        global_features = global_features * self.input_scale
        
        # 2. Adaptive localized decision making through Dynamic Graph SNN
        snn_output = self.snn_block(global_features)
        
        # 3. Optional enhanced attention through Spiking Transformer
        if self.use_spiking_transformer:
            transformer_output = self.spiking_transformer(snn_output)
            snn_output = snn_output + transformer_output * 0.5  # Residual connection
        
        # 4. Output attention for action selection focus
        attended_output = self.output_attention(snn_output)
        
        # 5. Generate action logits
        action_logits = self.output_fc(attended_output)
        
        # 6. Final spiking output (rate-coded action values)
        output_spikes = self.output_neuron(action_logits * 2.0)  # Aggressive scaling for output spikes
        
        # Ensure output shape is correct
        if output_spikes.dim() != 2 or output_spikes.shape[-1] != self.num_classes:
            raise ValueError(f"Unexpected output shape: {output_spikes.shape}")
        
        if return_spikes:
            # Return spike statistics for analysis
            return output_spikes, {
                'global_features': global_features,
                'snn_block_spikes': snn_output,
                'output_spikes': output_spikes,
                'attended_output': attended_output
            }
        
        return output_spikes

    def reset(self):
        """Reset all memory modules for clean state between episodes"""
        super().reset()
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
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_proj_bn = nn.BatchNorm1d(d_model, momentum=bn_momentum, eps=bn_eps)
        
        # Spiking neurons for attention processing
        self.attn_lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0) * 0.5,  # Faster for attention
            v_threshold=config.get('lif_v_threshold', 0.5) * 0.7,  # Lower threshold
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
        
    def forward(self, x):
        """
        Simple spiking self-attention.
        x: [batch_size, seq_len, d_model] or [batch_size, d_model] for single token
        """
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
        
        # Scaled dot-product attention (simplified)
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output.view(-1, d_model))
        attn_output = self.out_proj_bn(attn_output)  # Apply batch norm
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Pass through spiking neuron and add residual
        spiking_attn = self.attn_lif(attn_output * config.get('input_scale', 2.0))
        x = x + spiking_attn
        
        # Feedforward with residual connection
        ff_input = self.norm2(x)
        ff_output = self.ff_network(ff_input.view(-1, d_model) * config.get('input_scale', 2.0))
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
