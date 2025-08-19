# Minimal optimizer / scheduler helpers
import torch

def build_optimizer(model,cfg):
    optim_cfg=cfg.get('optimizer',{})
    lr=optim_cfg.get('lr',1e-3)
    wd=optim_cfg.get('weight_decay',0.0)
    opt_type=optim_cfg.get('type','adam').lower()
    params=model.parameters()
    if opt_type=='adam':
        opt=torch.optim.Adam(params,lr=lr,weight_decay=wd,betas=optim_cfg.get('betas',(0.9,0.999)))
    elif opt_type=='adamw':
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=wd)
    elif opt_type=='sgd':
        opt=torch.optim.SGD(params,lr=lr,momentum=optim_cfg.get('momentum',0.9),weight_decay=wd,nesterov=True)
    else:
        raise ValueError(f'Unknown optimizer type {opt_type}')
    sched_cfg=cfg.get('scheduler',{})
    scheduler=None
    if sched_cfg.get('type')=='cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=sched_cfg.get('t_max',100))
    elif sched_cfg.get('type')=='step':
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=sched_cfg.get('step_size',50),gamma=sched_cfg.get('gamma',0.5))
    return opt,scheduler
