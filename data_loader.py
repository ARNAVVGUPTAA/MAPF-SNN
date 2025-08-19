# Clean release data loader (subset)
import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class MAPFBaseDataset(Dataset):
    def __init__(self,cfg,mode):
        self.cfg = cfg[mode]; root=self.cfg['root_dir']
        self.dir = os.path.join(root,'train' if mode=='train' else 'valid')
        self.cases=[d for d in os.listdir(self.dir) if d.startswith('case_') and all(os.path.exists(os.path.join(self.dir,d,f)) for f in ['states.npy','trajectory_record.npy','gso.npy'])]
        self.count=0
        T=self.cfg['min_time']; A=self.cfg['nb_agents']
        self.states=np.zeros((len(self.cases),T,A,2,5,5),dtype=np.float32)
        self.trajectories=np.zeros((len(self.cases),T,A),dtype=np.float32)
        self.gsos=np.zeros((len(self.cases),T,A,A),dtype=np.float32)
        for i,case in enumerate(self.cases):
            try:
                st=np.load(os.path.join(self.dir,case,'states.npy'))[1:T+1]
                tr=np.load(os.path.join(self.dir,case,'trajectory_record.npy'))[:,:T]
                gso=np.load(os.path.join(self.dir,case,'gso.npy'))[:T,0]+np.eye(A)
                if st.shape[0]<T or tr.shape[1]<T: continue
                self.states[i]=st; self.trajectories[i]=tr.T; self.gsos[i]=gso; self.count+=1
            except: continue
        self.states=self.states[:self.count]; self.trajectories=self.trajectories[:self.count]; self.gsos=self.gsos[:self.count]
        if self.count==0: raise RuntimeError('No data loaded')
    def __len__(self): return self.count
    def __getitem__(self,i):
        return torch.from_numpy(self.states[i]).float(), torch.from_numpy(self.trajectories[i]).float(), torch.from_numpy(self.gsos[i]).float(), i

class CreateSNNDataset(MAPFBaseDataset):
    def __init__(self,cfg,mode): super().__init__(cfg,mode)

class SNNDataLoader:
    def __init__(self,cfg):
        use_pin = torch.cuda.is_available()
        train_set=CreateSNNDataset(cfg,'train')
        self.train_loader=DataLoader(train_set,batch_size=cfg['train']['batch_size'],shuffle=True,pin_memory=use_pin,drop_last=True,num_workers=cfg['train'].get('num_workers',0))
        if 'valid' in cfg:
            valid_set=CreateSNNDataset(cfg,'valid')
            self.valid_loader=DataLoader(valid_set,batch_size=cfg['valid'].get('batch_size',1),shuffle=False,pin_memory=use_pin,num_workers=cfg['valid'].get('num_workers',0))
        else:
            self.valid_loader=None
