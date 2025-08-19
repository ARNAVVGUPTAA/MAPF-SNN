"""Neuromodulated SNN framework (clean release).

Only depends on torch + spikingjelly. No global config usage to stay self‑contained.
"""
from spikingjelly.activation_based import neuron, base
import torch, torch.nn as nn

class EILIFLayer(base.MemoryModule):
    def __init__(self, input_dim, output_dim, cfg=None, tau=2.0, v_threshold=1.0, layer_name=None, recurrent=False):
        super().__init__()
        self.recurrent = recurrent
        if cfg is not None:
            base_tau = float(cfg.get('lif_tau', tau))
            base_thr = float(cfg.get('lif_v_threshold', v_threshold))
            self.input_scale = float(cfg.get('input_scale', 1.0))
            if layer_name:
                tau *= float(cfg.get(f'lif_tau_scale_{layer_name}', 1.0))
                v_threshold = base_thr * float(cfg.get(f'lif_threshold_scale_{layer_name}', 1.0))
                self.output_scale = float(cfg.get(f'output_scale_{layer_name}', 1.0))
            else:
                tau = base_tau; v_threshold = base_thr; self.output_scale = 1.0
        else:
            self.input_scale = 1.0; self.output_scale = 1.0
        internal = max(output_dim*2,16)
        e_n = int(0.8*internal); i_n = internal - e_n
        self.lin_e = nn.Linear(input_dim, e_n)
        self.lin_i = nn.Linear(input_dim, i_n)
        self.lif_e = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=0.)
        self.lif_i = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=0.)
        self.i2e = nn.Parameter(torch.randn(i_n, e_n)*0.1)
        if self.recurrent:
            self.rec_e = nn.Parameter(torch.randn(e_n, e_n)*0.05)
            self.rec_i = nn.Parameter(torch.randn(i_n, i_n)*0.05)
            self.prev_e = None; self.prev_i=None
        self.out = nn.Linear(e_n, output_dim)
        self.lif_out = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=0.)
    def forward(self,x):
        x = x * self.input_scale
        e_in = self.lin_e(x); i_in = self.lin_i(x)
        if self.recurrent and self.prev_e is not None:
            e_in = e_in + self.prev_e @ self.rec_e
            i_in = i_in + self.prev_i @ self.rec_i
        e_sp_raw = self.lif_e(e_in); i_sp = self.lif_i(i_in)
        if self.recurrent:
            self.prev_e = e_sp_raw.detach(); self.prev_i = i_sp.detach()
        inhib = i_sp @ self.i2e
        e_sp = e_sp_raw * (1.0 - 0.3*torch.sigmoid(inhib))
        out = self.out(e_sp)
        sp_out = self.lif_out(out) * self.output_scale
        self.spike_e = e_sp_raw; self.spike_i = i_sp; self.spike_out = sp_out
        return sp_out
    def reset_recurrent_state(self):
        if self.recurrent: self.prev_e=None; self.prev_i=None

class AttentionFeatureExtractor(base.MemoryModule):
    def __init__(self, in_ch, fov, out_dim, cfg=None):
        super().__init__(); self.fov=fov
        f_flat = in_ch*fov*fov
        self.embed = EILIFLayer(f_flat,256,cfg,layer_name="embedding")
        self.attn1 = EILIFLayer(256,128,cfg,layer_name="attention1",recurrent=True)
        self.attn2 = nn.Linear(128,fov*fov)
        self.feature = EILIFLayer(256,out_dim,cfg,layer_name="feature_net",recurrent=True)
    def forward(self,fov):
        b,a = fov.shape[:2]
        flat = fov.view(b*a,-1)
        emb = self.embed(flat).view(b,a,-1)
        a1 = self.attn1(emb.view(b*a,-1))
        weights = torch.sigmoid(self.attn2(a1)).view(b,a,self.fov,self.fov)
        attended = fov * weights.unsqueeze(2)
        feat = self.feature(self.embed(attended.view(b*a,-1))).view(b,a,-1)
        return feat, weights
    def reset_recurrent_state(self):
        self.attn1.reset_recurrent_state(); self.feature.reset_recurrent_state()

class DynamicGraphSNN(base.MemoryModule):
    def __init__(self, feat_dim, hid, cfg=None):
        super().__init__(); self.hid=hid; self.feat_dim=feat_dim
        self.node = EILIFLayer(feat_dim,hid,cfg,layer_name="node_processor",recurrent=True)
        self.edge = EILIFLayer(feat_dim*2,hid,cfg,layer_name="edge_processor")
        self.msg = EILIFLayer(hid+hid,hid,cfg,layer_name="message_net",recurrent=True)
        self.out = EILIFLayer(hid,feat_dim,cfg,layer_name="output_net",recurrent=True)
    def forward(self,x):
        b,a,_=x.shape
        node_h = self.node(x.view(b*a,-1)).view(b,a,self.hid)
        msgs = torch.zeros_like(node_h)
        for i in range(a):
            for j in range(a):
                if i==j: continue
                e_feat = torch.cat([x[:,i], x[:,j]], dim=-1)
                e_h = self.edge(e_feat)
                m_in = torch.cat([node_h[:,i], e_h], dim=-1)
                msgs[:,j]+= self.msg(m_in)
        upd = self.out(msgs.view(b*a,self.hid)).view(b,a,self.feat_dim)
        return upd
    def reset_recurrent_state(self):
        self.node.reset_recurrent_state(); self.msg.reset_recurrent_state(); self.out.reset_recurrent_state()

class NeuromodulatedSNN(base.MemoryModule):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.hidden = cfg.get('hidden_dim',128); self.num_actions=cfg.get('num_actions',5); self.num_agents=cfg.get('num_agents',5)
        self.feat = AttentionFeatureExtractor(2,5,self.hidden,cfg)
        self.graph = DynamicGraphSNN(self.hidden,self.hidden,cfg)
        self.out = EILIFLayer(self.hidden,self.num_actions,cfg,layer_name="output_layer",recurrent=True)
    def forward(self,fov,positions=None,goals=None):  # positions/goals kept for future extension
        if len(fov.shape)==4:
            ba,c,h,w = fov.shape
            b = ba//self.num_agents
            fov = fov.view(b,self.num_agents,c,h,w)
        feats,_ = self.feat(fov)
        g = self.graph(feats)
        logits = self.out(g.view(g.shape[0]*g.shape[1], g.shape[2])).view(g.shape[0],g.shape[1],self.num_actions)
        if hasattr(self,'_orig4d') and self._orig4d:
            logits = logits.view(-1,self.num_actions)
        return logits
    def set_neuromodulators(self,d,g): self.dopamine_level=d; self.gaba_level=g
    def reset_state(self):
        for m in self.modules():
            if hasattr(m,'reset'): m.reset()
        self.feat.reset_recurrent_state(); self.graph.reset_recurrent_state(); self.out.reset_recurrent_state()
    def reset_recurrent_state(self):
        self.feat.reset_recurrent_state(); self.graph.reset_recurrent_state(); self.out.reset_recurrent_state()
