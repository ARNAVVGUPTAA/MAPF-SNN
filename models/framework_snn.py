"""
Receptor-Based SNN Framework for MAPF
=====================================

Biologically-inspired implementation with proper neurotransmitter dynamics:
- AMPA/NMDA receptors for fast/slow excitation
- GABAA/GABAB receptors for fast/slow inhibition
- Dopamine modulation of excitatory transmission
- GABA modulation of inhibitory transmission
- Conductance-based synaptic currents
- Simplified architecture for clarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from spikingjelly.activation_based import base
from config import config
from models.receptor_dynamics import ReceptorLIFNeuron, create_receptor_layer


class ReceptorLayer(base.MemoryModule):
    """
    Simplified layer using proper neurotransmitter receptor dynamics.
    
    Replaces the complex EILIFLayer with a cleaner receptor-based approach:
    - AMPA: Fast excitation (τ=2ms)
    - NMDA: Slow excitation (τ=50ms) with voltage-dependent gating
    - GABAA: Fast inhibition (τ=6ms)
    - GABAB: Slow inhibition (τ=150ms)
    
    This is biologically accurate and simpler than the old multiplicative inhibition.
    """
    
    def __init__(self, input_dim, output_dim, config_dict=None, layer_name=None, recurrent=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_name = layer_name
        self.recurrent = recurrent
        
        # Create receptor-based neuron layer
        self.neuron_layer = create_receptor_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            config=config_dict
        )
        
        # Optional recurrent connections (simple linear feedback)
        if self.recurrent:
            self.recurrent_weight = nn.Parameter(torch.randn(output_dim, output_dim) * 0.02)
            self.prev_spikes = None
        
        # Neuromodulator levels (set by external controller)
        self.dopamine_level = 1.0
        self.gaba_level = 1.0
        
        # Store spikes for monitoring
        self.last_spikes = None
    
    def forward(self, x):
        """
        Forward pass with receptor dynamics.
        
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            spikes: Output spikes [batch, output_dim]
        """
        # Forward through receptor-based neurons
        spikes = self.neuron_layer(
            x, 
            dopamine_level=self.dopamine_level,
            gaba_level=self.gaba_level
        )
        
        # Add recurrent feedback if enabled (after neuron layer)
        if self.recurrent and self.prev_spikes is not None:
            recurrent_input = torch.matmul(self.prev_spikes, self.recurrent_weight)
            spikes = spikes + recurrent_input
        
        # Store spikes for recurrent and monitoring
        if self.recurrent:
            self.prev_spikes = spikes.detach()
        self.last_spikes = spikes
        
        return spikes
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Set neuromodulator levels"""
        self.dopamine_level = dopamine
        self.gaba_level = gaba
    
    def reset_recurrent_state(self):
        """Reset recurrent state"""
        if self.recurrent:
            self.prev_spikes = None
    
    def reset(self):
        """Reset all states"""
        self.neuron_layer.reset_state()
        if self.recurrent:
            self.prev_spikes = None
    
    def get_spike_rate(self) -> float:
        """Get spike rate for monitoring"""
        return self.neuron_layer.get_spike_rate()
    
    def get_ei_ratio(self) -> float:
        """Get E/I ratio for health monitoring"""
        return self.neuron_layer.get_ei_ratio()
    
    def get_receptor_stats(self) -> Dict[str, float]:
        """Get detailed receptor statistics"""
        return self.neuron_layer.get_receptor_stats()


class AttentionFeatureExtractor(base.MemoryModule):
    """Simplified attention-based feature extractor using receptor dynamics"""
    
    def __init__(self, input_channels, fov_size, output_dim, config_dict=None):
        super().__init__()
        # Calculate actual FOV flat size
        self.input_channels = input_channels
        self.fov_size = fov_size
        fov_flat_size = input_channels * fov_size * fov_size
        
        print(f"🔧 [AttentionFeatureExtractor] Expected input: {input_channels}x{fov_size}x{fov_size} = {fov_flat_size}")
        
        # Simplified layers using receptor dynamics
        embed_dim = 256
        attn_dim = 128
        attn_target_size = fov_size * fov_size
        
        # Use ReceptorLayer instead of EILIFLayer
        self.embedding = ReceptorLayer(fov_flat_size, embed_dim, config_dict, layer_name="embedding")
        self.attention1 = ReceptorLayer(embed_dim, attn_dim, config_dict, layer_name="attention1", recurrent=True)
        # Use regular linear layer for exact attention output size
        self.attention2 = nn.Linear(attn_dim, attn_target_size)
        
        # Feature processing
        self.feature_net = ReceptorLayer(embed_dim, output_dim, config_dict, layer_name="feature_net", recurrent=True)
        
        self.fov_size = fov_size
        
    def forward(self, fov):
        batch_size, num_agents = fov.shape[:2]
        # Flatten FOV
        fov_flat = fov.view(batch_size * num_agents, -1)
        # Shared embedding
        embedded = self.embedding(fov_flat)
        embedded = embedded.view(batch_size, num_agents, -1)
        # Compute attention weights
        attn1 = self.attention1(embedded.view(batch_size * num_agents, -1))
        attn_weights = torch.sigmoid(self.attention2(attn1))
        attn_weights = attn_weights.view(batch_size, num_agents, self.fov_size, self.fov_size)
        # Apply attention to FOV
        fov_attended = fov * attn_weights.unsqueeze(2)
        # Final feature extraction
        fov_attended_flat = fov_attended.view(batch_size * num_agents, -1)
        embedded_attended = self.embedding(fov_attended_flat)
        features = self.feature_net(embedded_attended)
        features = features.view(batch_size, num_agents, -1)
        return features, attn_weights
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels to all layers"""
        self.embedding.set_neuromodulators(dopamine, gaba)
        self.attention1.set_neuromodulators(dopamine, gaba)
        self.feature_net.set_neuromodulators(dopamine, gaba)
    
    def reset_recurrent_state(self):
        """Reset recurrent state for attention layers"""
        self.attention1.reset_recurrent_state()
        self.feature_net.reset_recurrent_state()


class DynamicGraphSNN(base.MemoryModule):
    """Simple dynamic graph SNN processor using receptor dynamics"""
    
    def __init__(self, feature_dim, hidden_dim, config_dict=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Node processing (with recurrent memory for temporal integration)
        self.node_processor = ReceptorLayer(feature_dim, hidden_dim, config_dict, layer_name="node_processor", recurrent=True)
        
        # Edge processing  
        self.edge_processor = ReceptorLayer(feature_dim * 2, hidden_dim, config_dict, layer_name="edge_processor")
        
        # Message passing - input is hidden_dim + hidden_dim (with recurrent memory)
        self.message_net = ReceptorLayer(hidden_dim + hidden_dim, hidden_dim, config_dict, layer_name="message_net", recurrent=True)
        
        # Output (with recurrent memory for temporal decisions)
        self.output_net = ReceptorLayer(hidden_dim, feature_dim, config_dict, layer_name="output_net", recurrent=True)
        
    def forward(self, node_features):
        batch_size, num_agents, feature_dim = node_features.shape
        
        # Process nodes - flatten batch and agent dims
        node_flat = node_features.view(batch_size * num_agents, feature_dim)
        node_hidden = self.node_processor(node_flat)
        # ADD NORMALIZATION
        node_hidden = F.layer_norm(node_hidden, [node_hidden.shape[-1]])
        node_hidden = node_hidden.view(batch_size, num_agents, self.hidden_dim)
        
        # Create simple fully-connected graph
        edges = []
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    edge_feat = torch.cat([node_features[:, i], node_features[:, j]], dim=-1)
                    edge_hidden = self.edge_processor(edge_feat)
                    # ADD NORMALIZATION
                    edge_hidden = F.layer_norm(edge_hidden, [edge_hidden.shape[-1]])
                    edges.append((i, j, edge_hidden))
        
        # Message passing
        messages = torch.zeros_like(node_hidden)
        for src, dst, edge_feat in edges:
            msg_input = torch.cat([node_hidden[:, src], edge_feat], dim=-1)
            message = self.message_net(msg_input)
            messages[:, dst] += message
        
        # Update nodes - flatten batch and agent dims
        # ADD NORMALIZATION
        messages = F.layer_norm(messages, [messages.shape[-1]])
        messages_flat = messages.view(batch_size * num_agents, self.hidden_dim)
        updated_features = self.output_net(messages_flat)
        updated_features = updated_features.view(batch_size, num_agents, self.feature_dim)
        
        return updated_features
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels to all layers"""
        self.node_processor.set_neuromodulators(dopamine, gaba)
        self.edge_processor.set_neuromodulators(dopamine, gaba)
        self.message_net.set_neuromodulators(dopamine, gaba)
        self.output_net.set_neuromodulators(dopamine, gaba)
    
    def reset_recurrent_state(self):
        """Reset recurrent state for all layers"""
        self.node_processor.reset_recurrent_state()
        self.message_net.reset_recurrent_state()
        self.output_net.reset_recurrent_state()


class SpikingRecurrentMemory(base.MemoryModule):
    """Pure SNN recurrent memory using receptor-based neurons with feedback"""
    
    def __init__(self, input_dim, hidden_dim, config_dict=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input to hidden (receptor layer)
        self.input_layer = ReceptorLayer(input_dim, hidden_dim, config_dict, layer_name="recurrent_input")
        
        # Recurrent connection (hidden to hidden) - pure spiking
        self.recurrent_layer = ReceptorLayer(hidden_dim, hidden_dim, config_dict, layer_name="recurrent_hidden", recurrent=True)
        
        # Memory state (accumulated spikes over time)
        self.memory_state = None
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, A, D]
        Returns:
            output: Spike output with memory [B, A, D]
        """
        B, A, D = x.shape
        x_flat = x.reshape(B * A, D)
        
        # Process input through spiking layer
        input_spikes = self.input_layer(x_flat)  # [B*A, hidden_dim]
        
        # Add recurrent connection if memory exists
        if self.memory_state is not None:
            # Recurrent spikes from previous state
            recurrent_spikes = self.recurrent_layer(self.memory_state)
            # Combine input and recurrent spikes
            combined = input_spikes + recurrent_spikes
        else:
            combined = input_spikes
            
        # Update memory state (leaky integration of spikes)
        if self.memory_state is None:
            self.memory_state = combined.detach()
        else:
            # Leaky integration: decay old memory, add new spikes
            self.memory_state = 0.9 * self.memory_state + 0.1 * combined.detach()
        
        return combined.reshape(B, A, self.hidden_dim)
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels"""
        self.input_layer.set_neuromodulators(dopamine, gaba)
        self.recurrent_layer.set_neuromodulators(dopamine, gaba)
    
    def reset_memory(self):
        """Reset memory state for new sequence"""
        self.memory_state = None
        
    def reset(self):
        """Reset all neuron states"""
        self.input_layer.reset()
        self.recurrent_layer.reset()
        self.reset_memory()


class SpikingAgentCommunication(base.MemoryModule):
    """Pure SNN message passing between agents using receptor-based spikes"""
    
    def __init__(self, feature_dim, communication_dim, num_hops, config_dict=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.communication_dim = communication_dim
        self.num_hops = num_hops
        
        # Message generation (receptor-based)
        self.message_gen = ReceptorLayer(feature_dim, communication_dim, config_dict, layer_name="comm_gen")
        
        # Message aggregation (receptor-based)
        self.message_agg = ReceptorLayer(communication_dim, feature_dim, config_dict, layer_name="comm_agg")
        
        # Update gate (receptor-based) - controls how much to incorporate messages
        self.update_gate = ReceptorLayer(feature_dim + communication_dim, feature_dim, config_dict, layer_name="comm_gate")
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: [B, A, D] agent features
            adjacency_matrix: [B, A, A] agent connectivity
        Returns:
            updated_features: [B, A, D] with communication
        """
        B, A, D = node_features.shape
        
        current_features = node_features
        
        # Multiple hops of message passing
        for hop in range(self.num_hops):
            # Generate messages from each agent (spiking)
            messages = self.message_gen(current_features.reshape(B * A, D))  # [B*A, comm_dim]
            messages = messages.reshape(B, A, self.communication_dim)
            
            # Aggregate messages from neighbors (weighted by adjacency)
            # Expand for broadcasting: [B, A, 1, comm_dim] * [B, 1, A, 1]
            messages_expanded = messages.unsqueeze(2)  # [B, A, 1, comm_dim]
            adjacency_expanded = adjacency_matrix.unsqueeze(-1)  # [B, A, A, 1]
            
            # Weighted sum of neighbor messages
            received_messages = (messages_expanded * adjacency_expanded).sum(dim=1)  # [B, A, comm_dim]
            
            # Aggregate received messages through spiking layer
            aggregated = self.message_agg(received_messages.reshape(B * A, self.communication_dim))
            aggregated = aggregated.reshape(B, A, D)
            
            # Update features with gating mechanism (spiking)
            combined = torch.cat([current_features, received_messages], dim=-1)  # [B, A, D+comm_dim]
            update = self.update_gate(combined.reshape(B * A, D + self.communication_dim))
            current_features = update.reshape(B, A, D)
        
        return current_features
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels"""
        self.message_gen.set_neuromodulators(dopamine, gaba)
        self.message_agg.set_neuromodulators(dopamine, gaba)
        self.update_gate.set_neuromodulators(dopamine, gaba)
    
    def reset(self):
        """Reset all neuron states"""
        self.message_gen.reset()
        self.message_agg.reset()
        self.update_gate.reset()


class MultiHeadSpikingAttention(base.MemoryModule):
    """Pure SNN multi-head attention using receptor-based neurons"""
    
    def __init__(self, feature_dim, num_heads, config_dict=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections (receptor-based)
        self.query_proj = ReceptorLayer(feature_dim, feature_dim, config_dict, layer_name="attn_query")
        self.key_proj = ReceptorLayer(feature_dim, feature_dim, config_dict, layer_name="attn_key")
        self.value_proj = ReceptorLayer(feature_dim, feature_dim, config_dict, layer_name="attn_value")
        
        # Output projection (receptor-based)
        self.output_proj = ReceptorLayer(feature_dim, feature_dim, config_dict, layer_name="attn_output")
        
    def forward(self, x):
        """
        Args:
            x: [B, A, D] agent features
        Returns:
            output: [B, A, D] attended features
        """
        B, A, D = x.shape
        x_flat = x.reshape(B * A, D)
        
        # Generate Q, K, V through spiking layers
        Q = self.query_proj(x_flat).reshape(B, A, self.num_heads, self.head_dim)
        K = self.key_proj(x_flat).reshape(B, A, self.num_heads, self.head_dim)
        V = self.value_proj(x_flat).reshape(B, A, self.num_heads, self.head_dim)
        
        # Transpose for attention: [B, num_heads, A, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores (spike-based correlation)
        # Use spike counts as similarity measure
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, num_heads, A, head_dim]
        
        # Reshape and project output through spiking layer
        attended = attended.transpose(1, 2).reshape(B * A, D)
        output = self.output_proj(attended).reshape(B, A, D)
        
        return output
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels"""
        self.query_proj.set_neuromodulators(dopamine, gaba)
        self.key_proj.set_neuromodulators(dopamine, gaba)
        self.value_proj.set_neuromodulators(dopamine, gaba)
        self.output_proj.set_neuromodulators(dopamine, gaba)
    
    def reset(self):
        """Reset all neuron states"""
        self.query_proj.reset()
        self.key_proj.reset()
        self.value_proj.reset()
        self.output_proj.reset()


class SpikingPredictiveModel(base.MemoryModule):
    """Pure SNN forward model for multi-step prediction using receptor dynamics"""
    
    def __init__(self, feature_dim, action_dim, prediction_horizon, config_dict=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        
        # State + action encoder (receptor-based)
        self.state_action_encoder = ReceptorLayer(
            feature_dim + action_dim, 
            feature_dim, 
            config_dict, 
            layer_name="pred_encoder"
        )
        
        # Prediction layers for each step (receptor-based)
        self.prediction_layers = nn.ModuleList([
            ReceptorLayer(feature_dim, feature_dim, config_dict, layer_name=f"pred_step_{i}")
            for i in range(prediction_horizon)
        ])
        
    def forward(self, current_state, action_logits):
        """
        Args:
            current_state: [B, A, D] current agent states
            action_logits: [B, A, action_dim] action predictions
        Returns:
            predictions: list of [B, A, D] predicted states for each horizon step
        """
        B, A, D = current_state.shape
        
        # Combine state and action
        combined = torch.cat([current_state, action_logits], dim=-1)  # [B, A, D+action_dim]
        combined_flat = combined.reshape(B * A, D + self.action_dim)
        
        # Encode state-action pair through spikes
        encoded = self.state_action_encoder(combined_flat).reshape(B, A, D)
        
        # Predict multiple steps ahead
        predictions = []
        current_pred = encoded
        
        for step_layer in self.prediction_layers:
            # Predict next state through spiking layer
            current_pred = step_layer(current_pred.reshape(B * A, D)).reshape(B, A, D)
            predictions.append(current_pred)
        
        return predictions
    
    def set_neuromodulators(self, dopamine: float, gaba: float):
        """Propagate neuromodulator levels"""
        self.state_action_encoder.set_neuromodulators(dopamine, gaba)
        for layer in self.prediction_layers:
            layer.set_neuromodulators(dopamine, gaba)
    
    def reset(self):
        """Reset all neuron states"""
        self.state_action_encoder.reset()
        for layer in self.prediction_layers:
            layer.reset()


class Network(base.MemoryModule):
    """Enhanced pure SNN network with multiple processing layers"""
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        self.input_dim = config.get('input_dim', 147)
        self.hidden_dim = config.get('hidden_dim', 128) 
        self.num_actions = config.get('num_actions', 5)
        self.num_agents = config.get('num_agents', 5)
        
        # Enhanced architecture flags
        self.use_recurrent_memory = config.get('use_recurrent_memory', False)
        self.use_agent_communication = config.get('use_agent_communication', False)
        self.use_multi_head_attention = config.get('num_attention_heads', 1) > 1
        self.use_predictive_model = config.get('use_predictive_model', False)
        self.num_snn_blocks = config.get('num_snn_blocks', 1)
        self.use_residual = config.get('use_residual_connections', False)
        
        # Feature extraction
        self.feature_extractor = AttentionFeatureExtractor(
            input_channels=2,
            fov_size=7,  # Match dataset FOV size (7x7)
            output_dim=self.hidden_dim,
            config_dict=config
        )
        
        # ENHANCEMENT 1: Recurrent Memory Module (pure SNN)
        if self.use_recurrent_memory:
            self.recurrent_memory = SpikingRecurrentMemory(
                input_dim=self.hidden_dim,
                hidden_dim=config.get('recurrent_memory_dim', 128),
                config_dict=config
            )
            print("✅ Enhanced: Spiking Recurrent Memory enabled")
        
        # ENHANCEMENT 2: Multi-Layer SNN Processing (stacked blocks)
        self.snn_blocks = nn.ModuleList([
            DynamicGraphSNN(
                feature_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                config_dict=config
            )
            for _ in range(self.num_snn_blocks)
        ])
        print(f"✅ Enhanced: {self.num_snn_blocks} SNN processing blocks (was 1)")
        
        # ENHANCEMENT 3: Agent Communication Module (pure SNN message passing)
        if self.use_agent_communication:
            self.agent_communication = SpikingAgentCommunication(
                feature_dim=self.hidden_dim,
                communication_dim=config.get('communication_dim', 64),
                num_hops=config.get('communication_hops', 2),
                config_dict=config
            )
            print(f"✅ Enhanced: Spiking Agent Communication with {config.get('communication_hops', 2)} hops")
        
        # ENHANCEMENT 4: Multi-Head Attention (pure SNN)
        if self.use_multi_head_attention:
            num_heads = config.get('num_attention_heads', 4)
            self.multi_head_attention = MultiHeadSpikingAttention(
                feature_dim=self.hidden_dim,
                num_heads=num_heads,
                config_dict=config
            )
            print(f"✅ Enhanced: Multi-Head Spiking Attention with {num_heads} heads")
        
        # ENHANCEMENT 5: Predictive Forward Model (pure SNN)
        if self.use_predictive_model:
            self.predictive_model = SpikingPredictiveModel(
                feature_dim=self.hidden_dim,
                action_dim=self.num_actions,
                prediction_horizon=config.get('prediction_horizon', 3),
                config_dict=config
            )
            print(f"✅ Enhanced: Spiking Predictive Model ({config.get('prediction_horizon', 3)} steps ahead)")
        
        # Output layer
        self.output_layer = ReceptorLayer(
            input_dim=self.hidden_dim,
            output_dim=self.num_actions,
            config_dict=config,
            layer_name="output_layer",
            recurrent=True
        )
        
        # Neuromodulation tracking
        self.dopamine_level = 1.0
        self.gaba_level = 1.0
        
        print("✅ Network initialized with receptor-based dynamics (AMPA/NMDA/GABAA/GABAB)")
    
    def set_neuromodulators(self, dopamine, gaba):
        """Set neuromodulator levels and propagate to all layers"""
        self.dopamine_level = dopamine
        self.gaba_level = gaba
        
        # Propagate to all modules
        self.feature_extractor.set_neuromodulators(dopamine, gaba)
        
        if self.use_recurrent_memory:
            self.recurrent_memory.set_neuromodulators(dopamine, gaba)
        
        for snn_block in self.snn_blocks:
            snn_block.set_neuromodulators(dopamine, gaba)
        
        if self.use_agent_communication:
            self.agent_communication.set_neuromodulators(dopamine, gaba)
        
        if self.use_multi_head_attention:
            self.multi_head_attention.set_neuromodulators(dopamine, gaba)
        
        if self.use_predictive_model:
            self.predictive_model.set_neuromodulators(dopamine, gaba)
        
        self.output_layer.set_neuromodulators(dopamine, gaba)
        
    def forward(self, fov, positions=None, goals=None):
        """Enhanced forward pass with all pure SNN modules"""
        # Handle both input formats:
        # Case 1: [B*agents, channels, height, width] - needs reshaping
        # Case 2: [B, agents, channels, height, width] - already correct
        
        original_was_4d = len(fov.shape) == 4
        self._original_input_was_4d = original_was_4d
        
        if original_was_4d:
            # Case 1: Reshape from [B*agents, C, H, W] to [B, agents, C, H, W]
            batch_agents, channels, height, width = fov.shape
            num_agents = self.num_agents
            batch_size = batch_agents // num_agents
            fov = fov.view(batch_size, num_agents, channels, height, width)
        
        # Extract features with attention
        features, attention_weights = self.feature_extractor(fov)
        
        # ENHANCEMENT 1: Apply recurrent memory (pure SNN)
        if self.use_recurrent_memory:
            features = self.recurrent_memory(features)
        
        # ENHANCEMENT 2: Multi-layer SNN processing with residual connections
        current_features = features
        for i, snn_block in enumerate(self.snn_blocks):
            block_output = snn_block(current_features)
            
            # Residual connection if enabled
            if self.use_residual and i > 0:
                current_features = current_features + block_output
            else:
                current_features = block_output
        
        graph_features = current_features
        
        # ENHANCEMENT 3: Agent communication (pure SNN message passing)
        if self.use_agent_communication:
            # Create simple fully-connected adjacency matrix
            batch_size, num_agents, _ = graph_features.shape
            adjacency = torch.ones(batch_size, num_agents, num_agents, device=graph_features.device)
            # Remove self-connections
            adjacency = adjacency * (1 - torch.eye(num_agents, device=graph_features.device).unsqueeze(0))
            
            graph_features = self.agent_communication(graph_features, adjacency)
        
        # ENHANCEMENT 4: Multi-head attention (pure SNN)
        if self.use_multi_head_attention:
            graph_features = self.multi_head_attention(graph_features)
        
        # Generate action logits through spiking output layer
        batch_size, num_agents, hidden_dim = graph_features.shape
        graph_features_flat = graph_features.view(batch_size * num_agents, hidden_dim)
        action_logits = self.output_layer(graph_features_flat)
        action_logits = action_logits.view(batch_size, num_agents, self.num_actions)
        
        # ENHANCEMENT 5: Predictive forward model (pure SNN)
        self.future_predictions = None
        if self.use_predictive_model:
            self.future_predictions = self.predictive_model(graph_features, action_logits)
        
        # Reshape back to [B*agents, num_actions] to match expected output
        if original_was_4d:
            batch_agents = action_logits.shape[0] * action_logits.shape[1]
            action_logits = action_logits.view(batch_agents, self.num_actions)
        
        return action_logits
    
    def get_spike_rates(self):
        """Get spike rates for monitoring with receptor-based system"""
        spike_rates = {}
        
        # Collect spike rates from ReceptorLayers
        for name, module in self.named_modules():
            if isinstance(module, ReceptorLayer):
                spike_rate = module.get_spike_rate()
                if spike_rate > 0:
                    spike_rates[name] = spike_rate
                
                # Also get E/I ratio for health monitoring
                ei_ratio = module.get_ei_ratio()
                spike_rates[f"{name}_ei_ratio"] = ei_ratio
        
        return spike_rates
    
    def get_receptor_health_report(self) -> Dict[str, float]:
        """Get comprehensive health report of receptor dynamics"""
        health_report = {
            'total_spike_rate': 0.0,
            'avg_ei_ratio': 0.0,
            'avg_ampa': 0.0,
            'avg_nmda': 0.0,
            'avg_gabaa': 0.0,
            'avg_gabab': 0.0,
            'num_layers': 0
        }
        
        for name, module in self.named_modules():
            if isinstance(module, ReceptorLayer):
                stats = module.get_receptor_stats()
                health_report['total_spike_rate'] += stats['spike_rate']
                health_report['avg_ei_ratio'] += stats['ei_ratio']
                health_report['avg_ampa'] += stats['g_ampa_mean']
                health_report['avg_nmda'] += stats['g_nmda_mean']
                health_report['avg_gabaa'] += stats['g_gabaa_mean']
                health_report['avg_gabab'] += stats['g_gabab_mean']
                health_report['num_layers'] += 1
        
        # Average values
        if health_report['num_layers'] > 0:
            for key in ['total_spike_rate', 'avg_ei_ratio', 'avg_ampa', 'avg_nmda', 'avg_gabaa', 'avg_gabab']:
                health_report[key] /= health_report['num_layers']
        
        return health_report
        
    def reset_state(self):
        """Reset SNN state including all enhancements"""
        # Reset all SNN neuron states
        for module in self.modules():
            if hasattr(module, 'reset'):
                module.reset()
        
        # Reset recurrent states
        self.feature_extractor.reset_recurrent_state()
        for snn_block in self.snn_blocks:
            snn_block.reset_recurrent_state()
        self.output_layer.reset_recurrent_state()
        
        # Reset enhancement modules
        if self.use_recurrent_memory:
            self.recurrent_memory.reset_memory()
    
    def reset_recurrent_state(self):
        """Reset only recurrent states for new sequence"""
        self.feature_extractor.reset_recurrent_state()
        for snn_block in self.snn_blocks:
            snn_block.reset_recurrent_state()
        self.output_layer.reset_recurrent_state()
        
        # Reset enhancement modules
        if self.use_recurrent_memory:
            self.recurrent_memory.reset_memory()

