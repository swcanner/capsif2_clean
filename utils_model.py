import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from scipy.spatial import distance_matrix as dm
from scipy.spatial.transform import Rotation as R
from utils import *
#from torchinfo import summary


carb_dict = {
    'abequose':0,
    'arabinose':1,
    'fucose':2,
    'galactosamine':3,
    'galactose':4,
    'galacturonic acid':5,
    'glucosamine':6,
    'glucose':7,
    'glucuronic acid':8,
    'mannosamine':9,
    'mannose':10,
    'neuraminic acid':11,
    'quinovose':12,
    'rhamnose':13,
    'ribose':14,
    'xylose':15
}

def model_train_sm(loader, model, optimizer, loss_fn, scaler, DEVICE, fake_batch_size=1):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []

    curr_loss = 0
    fake_batch_size=fake_batch_size
    my_iter = 0;

    model.train()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):
        #print(node_feat,coor)

        #nodes = torch.ones(len(coor),1)
        #nodes = nodes.to(device=DEVICE,dtype=torch.int)
        coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
        #print(coor.shape,nodes.shape)
        #exit the fail_state
        if len(coor.shape) < 2:
            #print('skip')
            continue;

        #if (coor.shape[1] != node_feat.shape[1]):
            #print('skip1')
            #continue


        node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
        #edge_feat = edge_feat.to(device=DEVICE,dtype=torch.float)
        #edges = edges.to(device=DEVICE,dtype=torch.int)

        label_res = label_res > 0.5
        label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()


        #exit the fail_state
        if len(coor.shape) < 2:
            continue;
        if len(label_res) != len(node_feat):
            #print('label-node mismatch')
            continue;

        #forward
        optimizer.zero_grad()
        #summary(model,node_feat,coor,edges,edge_feat)
        #exit();

        #FIX EDGE INFORMATION HERE
        #edges, edge_feat = fix_edges(edges, edge_feat, n_edge)

        pred_res = model(node_feat, coor, edges, edge_feat,
                        is_batch=False, n_res=n_res, n_edge=n_edge)

        #label_res = label_res > 0.5
        #print(pred_res)

        #print("PREDRES SIZE:",pred_res.size())
        pred_res = torch.squeeze(pred_res)
        if torch.any(torch.isnan(pred_res)):
            continue;
        #print(pred_res[:50,0],'\n',label_res[:50])
        #print(pred_res.size(),label_res.size())

        curr_loss += loss_fn(pred_res,label_res)
        all_loss.append(curr_loss)
        #print(curr_loss)
        curr_loss.backward()
        optimizer.step()
        #reset
        curr_loss = 0;
        my_iter += 1;

        #print(torch.sum(pred_res[:50,0]),torch.sum(label_res[:50]))
        #print(curr_loss)

        temp_loss = torch.FloatTensor(all_loss)
        #print(temp_loss)
        temp_loss = torch.sum(temp_loss)/len(all_loss)
        #print(temp_loss)
        loop.set_postfix (loss = temp_loss.item())

    model.eval();
    return temp_loss.item()


def model_train_two(loader, model, optimizer, loss_fn, scaler, DEVICE, fake_batch_size=1, epoch=0,
                    my_coef=[[.99,.5],[0.01,.5]], my_loss_epochs=100):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []

    curr_loss = 0
    fake_batch_size=fake_batch_size
    my_iter = 0;

    model.train()
    
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        with torch.autograd.set_detect_anomaly(True):

            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
            #print(coor.shape,nodes.shape)
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()

            label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
            label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze()


            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            if len(label_res) != len(node_feat):
                #print('label-node mismatch')
                continue;

            #forward
            optimizer.zero_grad()

            #summary(model,input_size=(node_feat,coor,edges,edge_feat))
            #summary(model)
            debug_mode=False

            if debug_mode==True:
                n_p = 0;
                for name, param in model.named_parameters():
                    #print(n_p,len(param))
                    n_p += 1
                    m,s,n,x = np.mean(param.cpu().detach().numpy()),np.std(param.cpu().detach().numpy()),np.min(param.cpu().detach().numpy()),np.max(param.cpu().detach().numpy())
                    #if s != 0.0:
                    #    print(name, n_p,len(param),m,s,n,x)
                    if n_p == 231:
                        print(param.cpu())

            pred_prot, pred_res = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            if type(pred_prot) == type(0):
                continue
            #print("PREDRES SIZE:",pred_res.size())
            pred_res = torch.squeeze(pred_res)
            #print(pred_res[:50,0],'\n',label_res[:50])
            #print(pred_res.size(),label_res.size())
            if torch.any(torch.isnan(pred_prot)):
                print("prot: nan")
                continue;
            if torch.any(torch.isnan(pred_res)):
                print('res: nan')
                continue;            


            loss_coef=get_loss_coef(epoch,coef1=my_coef[0],coef2=my_coef[1],n_epochs=my_loss_epochs)
            curr_loss = loss_fn(pred_prot,label_prot,pred_res,label_res,loss_coef)[0]

            #Update Loss

            all_loss.append(curr_loss)
            curr_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
    

            #print(torch.sum(pred_res[:50,0]),torch.sum(label_res[:50]))
            #print(curr_loss)

            temp_loss = torch.FloatTensor(all_loss)
            temp_loss = torch.sum(temp_loss)/len(all_loss)
            loop.set_postfix (loss = temp_loss.item())


    model.eval();
    return temp_loss.item()



def model_train_two_prot(loader, model, optimizer, loss_fn, scaler, DEVICE, epoch=0, MODIFIER=1.0):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []

    curr_loss = 0
    my_iter = 0;

    model.train()
    
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):

        with torch.autograd.set_detect_anomaly(True):

            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
            #print(coor.shape,nodes.shape)
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()

            label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
            label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze() * MODIFIER


            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            if len(label_res) != len(node_feat):
                #print('label-node mismatch')
                continue;

            #forward
            optimizer.zero_grad()

            #summary(model,input_size=(node_feat,coor,edges,edge_feat))
            #summary(model)
            debug_mode=False

            if debug_mode==True:
                n_p = 0;
                for name, param in model.named_parameters():
                    #print(n_p,len(param))
                    n_p += 1
                    m,s,n,x = np.mean(param.cpu().detach().numpy()),np.std(param.cpu().detach().numpy()),np.min(param.cpu().detach().numpy()),np.max(param.cpu().detach().numpy())
                    #if s != 0.0:
                    #    print(name, n_p,len(param),m,s,n,x)
                    if n_p == 231:
                        print(param.cpu())

            pred_prot = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            if type(pred_prot) == type(0):
                continue

            if torch.any(torch.isnan(pred_prot)):
                print("prot: nan")
                continue;

            pred_prot = pred_prot.squeeze()
            #print(pred_prot,label_prot)

            curr_loss = loss_fn(pred_prot,label_prot)

            #Update Loss

            all_loss.append(curr_loss)
            curr_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
    

            #print(torch.sum(pred_res[:50,0]),torch.sum(label_res[:50]))
            #print(curr_loss)

            temp_loss = torch.FloatTensor(all_loss)
            temp_loss = torch.sum(temp_loss)/len(all_loss)
            loop.set_postfix (loss = temp_loss.item())


    model.eval();
    return temp_loss.item()



def model_val_sm(loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []
    clusters = []


    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):
        #print(node_feat,coor)

        with torch.no_grad():

            #nodes = torch.ones(len(coor),1)
            #nodes = nodes.to(device=DEVICE,dtype=torch.int)
            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
            #print(coor.shape,nodes.shape)
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;


            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
            label_res = label_res > 0.5
            label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()

            if len(label_res) != len(node_feat):
                #print('label-node mismatch')
                continue;
            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            #summary(model,node_feat,coor,edges,edge_feat)

            pred_res = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            #print("PREDRES SIZE:",pred_res.size())
            pred_res = torch.squeeze(pred_res)

            curr_loss = loss_fn(pred_res,label_res)

            all_loss.append(curr_loss)

            temp_loss = torch.FloatTensor(all_loss)
            temp_loss = torch.sum(temp_loss)/len(all_loss)
            loop.set_postfix (loss = temp_loss.item())

    return temp_loss.item()


def model_val_two(loader, model, optimizer, loss_fn, scaler, DEVICE, final_coef=[.35,.65],epoch=0):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []
    all_track_loss = []
    clusters = []


    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):
        #print(node_feat,coor)

        with torch.no_grad():
            with torch.autograd.set_detect_anomaly(True):

                #nodes = torch.ones(len(coor),1)
                #nodes = nodes.to(device=DEVICE,dtype=torch.int)
                coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
                #print(coor.shape,nodes.shape)
                #exit the fail_state
                if len(coor.shape) < 2:
                    #print('skip')
                    continue;

                node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
                label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
                label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze()

                #exit the fail_state
                if len(coor.shape) < 2:
                    continue;
                if label_res.size()[0] != node_feat.size()[0]:
                    #print('label-node mismatch')
                    continue;
                #print(label_res.size()[0],node_feat.size()[0])

                pred_prot, pred_res = model(node_feat, coor, edges, edge_feat,
                                is_batch=False, n_res=n_res, n_edge=n_edge)
                
                #print(pred_prot[:50])
                #print("PREDRES SIZE:",pred_res.size())
                if type(pred_prot) == type(0):
                    continue;
                pred_res = torch.squeeze(pred_res)
                #print(pred_prot)
                #print(pred_prot.device,carb_binder.device,pred_res.device,loss_coef.device)
                #loss_coef = [.5,.5]
                if torch.any(torch.isnan(pred_prot)):
                    print("prot: nan")
                    continue;
                if torch.any(torch.isnan(pred_res)):
                    print('res: nan')
                    continue; 

                curr_loss, loss_tracks = loss_fn(pred_prot,label_prot,pred_res,label_res,final_coef)

                all_loss.append(curr_loss)

                all_track_loss.append(loss_tracks)
                temp_loss = torch.FloatTensor(all_loss)
                temp_loss = torch.sum(temp_loss)/len(all_loss)
                loop.set_postfix (loss = temp_loss.item())


    all_track_loss = np.array(all_track_loss)
    #print(all_track_loss)

    dv = [];
    for ii in range(len(all_track_loss)):
        val = all_track_loss[ii,1]
        if val != -1:
            dv.append(val)

    return temp_loss.item(), np.mean(all_track_loss[:,0]), np.mean(dv)



def model_val_two_prot(loader, model, optimizer, loss_fn, scaler, DEVICE,epoch=0):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    all_loss = []
    all_track_loss = []
    clusters = []

    cm = np.zeros((2,2))


    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge) in enumerate(loop):
        #print(node_feat,coor)

        with torch.no_grad():
            with torch.autograd.set_detect_anomaly(True):

                #nodes = torch.ones(len(coor),1)
                #nodes = nodes.to(device=DEVICE,dtype=torch.int)
                coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()
                #print(coor.shape,nodes.shape)
                #exit the fail_state
                if len(coor.shape) < 2:
                    #print('skip')
                    continue;

                node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
                label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
                label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze()

                #exit the fail_state
                if len(coor.shape) < 2:
                    continue;
                if label_res.size()[0] != node_feat.size()[0]:
                    #print('label-node mismatch')
                    continue;
                #print(label_res.size()[0],node_feat.size()[0])

                pred_prot = model(node_feat, coor, edges, edge_feat,
                                is_batch=False, n_res=n_res, n_edge=n_edge)
                
                #print(pred_prot[:50])
                #print("PREDRES SIZE:",pred_res.size())
                if type(pred_prot) == type(0):
                    continue;
                #print(pred_prot)
                #print(pred_prot.device,carb_binder.device,pred_res.device,loss_coef.device)
                #loss_coef = [.5,.5]
                if torch.any(torch.isnan(pred_prot)):
                    print("prot: nan")
                    continue;

                pred_prot = pred_prot.squeeze()

                if pred_prot.item() > .5:
                    if label_prot.item() > .5:
                        cm[0,0] += 1;
                    else:
                        cm[0,1] += 1
                else:
                    if label_prot.item() > .5:
                        cm[1,0] += 1;
                    else:
                        cm[1,1] += 1

                curr_loss = loss_fn(pred_prot,label_prot)

                all_loss.append(curr_loss)

                temp_loss = torch.FloatTensor(all_loss)
                temp_loss = torch.sum(temp_loss)/len(all_loss)
                loop.set_postfix (loss = temp_loss.item())


    all_track_loss = np.array(all_track_loss)
    #print(all_track_loss)

    return temp_loss.item(), cm



def dice_loss(pred,true,eps=1e-5):
    pred = torch.squeeze(pred)
    true = torch.squeeze(true)
    tp = torch.mul(pred,true);
    #print(pred.shape,true.shape,tp.shape)
    #print(pred[:50],true[:50],tp)
    tp = torch.sum(tp)
    #print(torch.sum(tp),torch.sum(pred),torch.sum(true))

    d = (2 * torch.sum(tp) + eps) / (torch.sum(pred) + torch.sum(true) + eps)
    return 1 - d;

def mcc_metric(y_pred,y_true,cutoff=0.5,eps=1e-5):
    y_pred = y_pred > cutoff;
    y_true = y_true > cutoff;
    tp = np.sum(y_pred * y_true)
    fp = np.sum(y_pred > y_true)
    fn = np.sum(y_pred < y_true)
    tn = np.sum(y_pred == y_true) - tp

    mcc = tn * tp - fn * fp
    mcc /= np.sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    return mcc

def acc_metric(y_pred,y_true,cutoff=0.5,eps=1e-5):
    #way to get it done 1x6 x 6x1
    y_pred = np.transpose(y_pred[:,0] > cutoff);
    y_true = np.array(y_true[:,0] > cutoff,dtype=float);
    #print(np.shape(y_pred))
    tp = np.matmul(y_pred, y_true)
    acc = tp / np.size(y_pred)
    print(tp,np.size(y_pred),acc)
    return acc

def print_metrics(epoch,step_loss,v_loss,v_pred,v_true,cutoff=0.5):
    #print(np.shape(v_pred),np.shape(v_true))
    v_pred = np.matrix(v_pred)
    v_true = np.matrix(v_true)
    y_pred = v_pred > cutoff;
    y_true = v_true > cutoff;
    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    for jj in range(len(v_pred[:,0])):
        if y_true[jj,0] == 1:
            if (y_true[jj,0] == y_pred[jj,0]):
                if y_true[jj,0] == 1:
                    tp += 1;
                else:
                    tn += 1;
            else:
                if y_true[jj,0] == 1:
                    fn += 1;
                else:
                    fp += 1;
    acc_1 = float( (tp + tn) / (tp + tn + fn + fp) )
    dice_1 = float(2*tp / (2*tp + tn + fn))
    acc_res_met = [];
    dice_res_met = [];
    for ii in range(len(carb_dict)):
        tp = 0;
        fp = 0;
        fn = 0;
        tn = 0;

        for jj in range(len(v_pred)):
            if y_true[jj,0] == 1:
                if (y_true[jj,ii+1] == y_pred[jj,ii+1]):
                    if y_true[jj,ii+1] == 1:
                        tp += 1;
                    else:
                        tn += 1;
                else:
                    if y_true[jj,ii+1] == 1:
                        fn += 1;
                    else:
                        fp += 1;

        acc_res_met.append((tp + tn) / (tp + tn + fn + fp))
        dice_res_met.append(2*tp / (2*tp + tn + fn))

    o = str(epoch) + " " + str(step_loss) + " " + str(v_loss) + " " + str(acc_1) + " ";
    for ii in range(len(acc_res_met)):
        o += str(acc_res_met[ii]) + " "
    for ii in range(len(dice_res_met)):
        o += str(dice_res_met[ii]) + " "
    print(o)
    return

if __name__ == "__main__":
    print("main")
