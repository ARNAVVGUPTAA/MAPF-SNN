# Clean release training script (no visualization)
import argparse, torch
from config import load_config
from data_loader import SNNDataLoader
from models.framework_snn import NeuromodulatedSNN
from neuromodulated_loss import NeuromodulatedRLLoss
from optimizer_utils import build_optimizer


def train(cfg):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=SNNDataLoader(cfg)
    model=NeuromodulatedSNN(cfg).to(device)
    criterion=NeuromodulatedRLLoss(cfg).to(device)
    optimizer,scheduler=build_optimizer(model,cfg)
    epochs=cfg.get('epochs',1)
    log_interval=cfg.get('log_interval',10)
    for epoch in range(1,epochs+1):
        model.train()
        for batch_idx,(states,traj,gso,idx) in enumerate(data.train_loader):
            states=states.to(device); traj=traj.to(device)
            optimizer.zero_grad()
            outputs=model(states)
            loss=criterion(outputs,traj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('grad_clip',1.0))
            optimizer.step()
            if batch_idx % log_interval==0:
                print(f'Epoch {epoch} [{batch_idx}/{len(data.train_loader)}] loss={loss.item():.4f}')
        if scheduler: scheduler.step()
        if getattr(data,'valid_loader',None):
            model.eval(); total=0; count=0
            with torch.no_grad():
                for states,traj,gso,idx in data.valid_loader:
                    states=states.to(device); traj=traj.to(device)
                    out=model(states)
                    l=criterion(out,traj); total+=l.item(); count+=1
            print(f'Validation loss: {total/max(count,1):.4f}')
    save_path=cfg.get('save_path','model.pt')
    torch.save({'model':model.state_dict(),'cfg':cfg},save_path)
    print('Saved model to',save_path)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='clean_release/configs/config_snn.yaml')
    args=ap.parse_args()
    cfg=load_config(args.config)
    train(cfg)
