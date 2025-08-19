#!/usr/bin/env python3
import torch, torch.nn as nn

class NeuromodulatedRLLoss(nn.Module):
    def __init__(self,cfg=None):
        super().__init__(); self.cfg=cfg or {}
        g=self.cfg.get
        self.goal_reward=g('goal_reward',10.0)
        self.collision_punishment=g('collision_punishment',-8.0)
        self.step_penalty=g('step_penalty',-0.5)
        self.cooperation_bonus=g('cooperation_bonus',3.0)
        self.reward_scale=g('reward_scale',0.2)
        self.punishment_scale=g('punishment_scale',0.2)
        self.trace_len=g('trace_length',10)
        self.register_buffer('reward_trace', torch.zeros(self.trace_len))
        self.register_buffer('punish_trace', torch.zeros(self.trace_len))
        self.step_count=0
    def compute(self,positions,goals,actions,collisions=None,goal_reached=None):
        B,A=positions.shape[:2]
        if goal_reached is not None:
            goal_r = goal_reached.float()*self.goal_reward
        else:
            dist = torch.norm(positions-goals,dim=-1)
            maxd = torch.sqrt(torch.tensor(2.0,device=dist.device))
            goal_r = (maxd-dist)/maxd * self.goal_reward*0.1
        if collisions is not None:
            col_p = collisions.float().abs()*abs(self.collision_punishment)
        else:
            col_p = torch.zeros_like(goal_r)
        stay = (actions==0).float()
        stay_p = stay * 5.0
        step_p = torch.ones_like(goal_r)*(-self.step_penalty)
        total_reward = goal_r
        total_punish = col_p + stay_p + step_p
        dopamine = torch.clamp(0.5 + torch.sigmoid(total_reward*self.reward_scale),0.1,2.1)
        gaba = torch.clamp(0.5 + torch.sigmoid(total_punish*self.punishment_scale),0.1,2.1)
        return dict(goal_rewards=goal_r, punishments=total_punish, dopamine=dopamine, gaba=gaba,
                    total_reward=total_reward, total_punishment=total_punish)
    def compute_loss(self, logits, target, dopa, gaba):
        probs = torch.softmax(logits,dim=-1).clamp(1e-7,1-1e-7)
        one_hot = torch.zeros_like(probs); one_hot.scatter_(-1,target.unsqueeze(-1),1.0)
        ce = -(one_hot*torch.log(probs)).sum(-1)
        base = ce * ((dopa+0.5)/1.5)
        activity = torch.abs(logits).mean(-1)
        reg = (gaba-0.5)*activity*0.05
        loss = base + torch.abs(reg)*0.05
        norm = (loss - loss.mean().detach())/(loss.std().detach()+1e-6)
        return torch.abs(norm)+0.1
    def forward(self, model_outputs, target_actions, positions, goals, collisions=None, goal_reached=None):
        r = self.compute(positions, goals, target_actions, collisions, goal_reached)
        loss = self.compute_loss(model_outputs, target_actions, r['dopamine'], r['gaba'])
        return {'loss': loss, 'dopamine': r['dopamine'], 'gaba': r['gaba'], 'rewards': r['total_reward'], 'punishments': r['total_punishment']}

def create_neuromodulated_loss(cfg=None): return NeuromodulatedRLLoss(cfg)

def detect_collisions(positions, threshold=0.5):
    B,A,_=positions.shape
    mask = torch.zeros(B,A,device=positions.device)
    for b in range(B):
        for i in range(A):
            for j in range(i+1,A):
                if torch.norm(positions[b,i]-positions[b,j])<threshold:
                    mask[b,i]=mask[b,j]=1
    return mask

def detect_goal_reached(positions, goals, threshold=0.5):
    return (torch.norm(positions-goals,dim=-1)<threshold).float()

class NeuromodulatedTrainingMode:
    def __init__(self,total_epochs): self.total=total_epochs; self.epoch=0
    def update_epoch(self,e): self.epoch=e
    def get_training_phase(self): p=self.epoch/self.total; return 'exploration' if p<0.3 else ('exploitation' if p<0.8 else 'stabilization')
    def get_neuromodulator_adjustments(self):
        ph=self.get_training_phase()
        if ph=='exploration': return {'dopamine_boost':0.3,'gaba_reduction':-0.1,'reward_scale':1.2,'punishment_scale':0.8}
        if ph=='exploitation': return {'dopamine_boost':0.0,'gaba_reduction':0.0,'reward_scale':1.0,'punishment_scale':1.0}
        return {'dopamine_boost':-0.1,'gaba_reduction':0.2,'reward_scale':0.9,'punishment_scale':1.1}

def create_training_mode_controller(total_epochs): return NeuromodulatedTrainingMode(total_epochs)
