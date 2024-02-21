import torch
import numpy as np
from tqdm import tqdm
from src.utils import recall_at_k, ndcg_at_k
from src.dataset import HMDataset

def parmas_indexing(params, idx, batch_size):
    new_param = []
    for param in params:
        if param.shape[0] > batch_size:
            param = param[idx]
        new_param.append(param)
    return new_param

def train(model, optimizer, scheduler, dataloader, criterion, device):
    model.train()
    total_loss = 0

    for user, pos, neg in tqdm(dataloader):
        user = user.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        
        diff, pos_params, neg_params = model(user, pos, neg)
        loss = criterion(diff, pos_params, neg_params)

        model.zero_grad()
        loss.backward()
        optimizer.step()        
        total_loss += loss.item()
        
    scheduler.step()
    return total_loss/len(dataloader)

def eval(model, mode, sample_size, dataloader, criterion, candidate_items_each_user, device):
    model.eval()
    metrics = {'R10':[], 'R20':[], 'R40':[], 'N10':[], 'N20':[], 'N40':[]}
    total_loss = 0
    target_idx = list(range(sample_size, (sample_size+1)*dataloader.batch_size, (sample_size+1)))
    sample_size = sample_size+1 

    with torch.no_grad():
        for user, *args in tqdm(dataloader):
            if mode == "valid":
                target, neg = args
                target = target.to(device)
                neg = neg.to(device)

            if mode == "test":
                target = args[0].to(device)
            
            # get metric
            user = user.to(device)
            candidate_items = torch.tensor([], dtype=int).to(device)
            users = torch.tensor([], dtype=int).to(device)
            
            for i in range(user.shape[0]):
                u = user[i].item()
                items = candidate_items_each_user[u].to(device) # get user's candidate (target is included) => sample+target
                u_ids = torch.tensor(np.full(sample_size, u)).to(device) # make same shape with items
                candidate_items = torch.cat([candidate_items, items], dim=0) 
                users = torch.cat([users, u_ids], dim=0)
            
            # dim : batch_size * (sample_size+1)
            # e.g. bc:512, sample:1000 => dim:512512
            candidate_out, candidate_parms = model.cal_each(users, candidate_items) 

            for i, t in enumerate(target):
                idx = target_idx[i] + 1
                if idx-sample_size<0:
                    print("ERROR: idx is larger than total length")
                    break

                # get each user's top_k result and metric
                t = t.unsqueeze(0)
                user_res = candidate_out[idx-sample_size:idx]
                user_candidate = candidate_items[idx-sample_size:idx]
                sorted_idx = user_res.argsort(descending=True)
                sorted_item = user_candidate[sorted_idx]
                
                for k in [10, 20, 40]:
                    metrics['R' + str(k)].append(recall_at_k(k, t, sorted_item))
                    metrics['N' + str(k)].append(ndcg_at_k(k, t, sorted_item))
                        
            # get loss
            if mode == "valid":
                neg_out, neg_params = model.cal_each(user, neg)
                pos_out = candidate_out[target_idx[:user.shape[0]]]
                pos_params = parmas_indexing(candidate_parms, target_idx[:user.shape[0]], dataloader.batch_size)
                loss = criterion(pos_out-neg_out, pos_params, neg_params)
                total_loss += loss.item()
              
        for k in [10, 20, 40]:
            metrics['R' + str(k)] = round(np.asarray(metrics['R' + str(k)]).mean(), 5)   
            metrics['N' + str(k)] = round(np.asarray(metrics['N' + str(k)]).mean(), 5)
        
        if mode == "valid":
            return total_loss/len(dataloader), metrics 
    return metrics