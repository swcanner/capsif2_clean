import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from scipy.spatial import distance_matrix as dm
from scipy.spatial.transform import Rotation as R
#from torchsummary import summary

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

class CSV_Dataset(Dataset):
    def __init__(self, cluster_file,  pdb_file, root_dir, nn=[6,12,18,24], train=False,
                    use_clusters=False,val=False,use_af2=True,af2_likelihood=.65,return_name=False, return_pdb_ref=False):
        """
        Arguments:
            cluster_file (string): Path to csv file of cluster/pdb annotations
            pdb_file (string): Path to the csv file with pdb file annotations
            root_dir (string): Directory to where csv_files were initialized from (home dir)
            nn (array): list of nearest neighbors to be used
            train (bool): if we are in evaluation of performance at all - includes validation and test (return label)
            use_clusters (bool): do we evaluate the sequences on cluster based information
            val (bool): if we are in validation / test mode -> returns the cluster and pdb_name
        """
        self.pdb_data = pd.read_csv(pdb_file,header=None)
        self.cluster_data = pd.read_csv(cluster_file)
        #print(self.data)
        self.root_dir = root_dir
        self.train = train;
        self.val = val;

        self.use_clusters = use_clusters;
        self.clusters_epoch = [];

        #nearest neighbors
        self.nn = nn;
        self.use_af2 = use_af2
        self.af2_likelihood = af2_likelihood

        self.return_pdb_ref = return_pdb_ref;

        #distance encodings RBF = gaus(phi) = exp( (eps * (x - u) )^2 )
        self.rbf_dist_means = np.linspace(0,20,16)
        self.rbf_eps = (self.rbf_dist_means[-1] - self.rbf_dist_means[0]) / len(self.rbf_dist_means);

        self.return_name = return_name

        self.fail_state = [0,0,0,0,0,0,0,0,0]
        if self.return_name:
            self.fail_state.append(0)

    def __len__(self):
        if (self.use_clusters):
            return len(self.cluster_data)
        else:
            return len(self.pdb_data)

    def __getitem__(self,idx,pad=True):
        """
        Arguments:
            idx (int): CSV file index for training/testing
            pad (bool): whether to pad the output or not
        Returns:
            esm_ (array): ESM features in all chains
            cb (array): coordinates of CBs in all chains
            edges (array of arrays (4xn) ): Edges of NN with varying cutoffs
            edge_feats (array of arrays (4xn) ): Edge features of each nearest neighbors
                    RBF of distance, Direction encoding, orientation encoding
            binder_type (array): Is_smallMol_binder , is_carb_binder
            carbs_bound (array): array of all bound carbs if present in struct
        """

        if torch.is_tensor(idx):
            idx = idx.tolist();



        clust = ""
        pdb_name = ""
        coor_files = ""
        esm_files = ""
        af_files = ""
        sm_binder = False
        carb_binder = False
        ref_pdb = []

        #training - get the pdb thru clusters
        if self.use_clusters:
            clust = self.cluster_data.iloc[idx,0]
            pdbs = self.cluster_data.iloc[idx,1].split('|')[:-1] #ends with a | so we remove the null

            #get random pdb
            r = np.random.randint(0,len(pdbs),size=1)

            #print(r)
            #r = [0]
            pdb_name = pdbs[r[0]]
            

            print(clust, pdb_name)
            try:
                new_df = self.pdb_data[self.pdb_data[1] == pdb_name].values[0]
                #print(new_df,clust)
                if clust != new_df[0]:
                    print("WHAT THE FUCK?!?!?! CLUSTERS: ",clust,new_df[0],pdb_name)

                coor_files = new_df[2]
                esm_files = new_df[3]
                af_files = new_df[4]
                carb_binder = new_df[5]
                sm_binder = new_df[6]

                #print(coor_files)
                #print(esm_files)
                #print(af_files)
            except:
                print("Failure to find: ",clust,pdb_name)
                return self.fail_state
        else:
            clust = self.pdb_data.iloc[idx,0]
            pdb_name = self.pdb_data.iloc[idx,1]
            coor_files = self.pdb_data.iloc[idx,2]
            esm_files = self.pdb_data.iloc[idx,3]
            af_files = self.pdb_data.iloc[idx,4]
            carb_binder = self.pdb_data.iloc[idx,5]
            sm_binder = self.pdb_data.iloc[idx,6]


        #print(idx,self.cluster_data.iloc[idx,:])

        #"Cluster,PDB,coor_files,esm_files,AF2_files,carb,sm";

        #load coor files together
        #try:
        ca,cb,frame,res_label = [],[],[],[]
        ca,cb,frame,res_label = self.load_coor(coor_files)

        if self.return_pdb_ref:
            ref_pdb = self.load_ref(coor_files)

        if type(res_label) == type(-1):
            return self.fail_state

        if (self.use_af2):
            #if af files exist
            if type(af_files) != type(np.nan):

                #see if we use the af file
                r = np.random.rand(0)
                if r < self.af2_likelihood:
                    ca_af,cb_af,frame_af = self.load_coor_af(coor_files)

                    if (len(ca) != len(ca_af)):
                        print("AF and PDB do not match!!!! ",pdb_name)
                    else:
                        ca,cb,frame = ca_af,cb_af,frame_af

        #print(np.shape(res_label))
        #print(res_label[:50])
        #print(np.shape(res_label.shape())
        if bool(carb_binder):
            if len(np.shape(res_label)) > 1:
                res_label = torch.unsqueeze(torch.from_numpy(res_label[:,0]),1)
            else:

                res_label = torch.unsqueeze(torch.from_numpy(res_label),1)
        else:
            res_label = torch.unsqueeze(torch.from_numpy(res_label),1)
        esm_ = self.load_esm(esm_files)

        #except:
        #    return self.fail_state;
        #print(pdb_name,len(cb))
        self.n_res = esm_.shape[0]

        #get the neighbors!
        edges,edge_feat = self.get_knn_info(cb,frame,self.nn)

        #make it torchy
        esm_ = torch.FloatTensor(esm_)
        #carb_oneHot = torch.IntTensor(carb_oneHot)
        cb = torch.FloatTensor(cb)
        #type_label = torch.IntTensor( [small_mol, is_na_binder, is_carb] )

        #return what is needed
        #return edge_feat
        if self.return_pdb_ref:
            return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge, pdb_name, ref_pdb

        if self.return_name:
            return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge, pdb_name

        return esm_, cb, edges, edge_feat, carb_binder, sm_binder, res_label, self.n_res, self.n_edge

    def load_carbs_oneHot(self,carbs):
        """
        Arguments:
            carbs (string): carb numbers seperated by "|"
        Returns:
            oneHot (np.array): array of carbohydrates bound in one-hot encoding
        """
        oneHot = np.zeros((len(carb_dict),))

        #print(carbs,type(carbs))
        if type(carbs) == type(np.nan):
            return oneHot
        #if np.isnan(carbs):
        #    return oneHot
        if carbs == "":
            return oneHot

        l = carbs.split('|')
        if len(l) == 0:
            return oneHot

        for ii in l:
            oneHot[int(ii)] = 1;
        return oneHot

    def load_esm(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to esm embedding files
        Returns:
            esm_ (np.array): array of all esm embeddings for all proteins
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        esm_ = [];

        for ii in l:
            #print(len(ii))
            curr = np.load(self.root_dir + ii)
            for jj in curr:
                esm_.append(jj)
        return np.array(esm_)

    def load_coor(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ca (np.array): array of all CA coor
            cb (np.array): array of all CB coor
            frame (np.array): array of all oriented frames
            label (np.array): Nres x 17 of all carbs bound
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        ca = [];
        cb = [];
        frame = []
        label = [];

        for ii in l:
            #print(self.root_dir + ii)
            curr = np.load(self.root_dir + ii)
            ca_c = curr['ca']
            cb_c = curr['cb']
            frame_c = curr['frame']

            if self.train:
                try:
                    label_c = curr['label']
                except:
                    label_c = np.zeros((cb_c.shape[0],1))

            for jj in range(len(ca_c)):
                #REMOVE DUPLICATES!!!!
                #this is a very lazy unoptimized way to do this but it works so fuck it
                skip_round = False;
                for kk in range(len(ca)):
                    if ca_c[jj][0] == ca[kk][0]:
                        if ca_c[jj][1] == ca[kk][1]:
                            if ca_c[jj][2] == ca[kk][2]:
                                skip_round=True;
                                break;
                if skip_round:
                    continue;

                ca.append(ca_c[jj])
                cb.append(cb_c[jj])
                frame.append(frame_c[jj])
                if self.train:
                    #messed up on adding 0 to end :)
                    if jj >= len(label_c):
                        #print("ADDED 0")
                        label.append(0)
                    else:

                        #print(label_c[jj])
                        #if jj < 3:
                            #print(label_c[jj],'\n',label_c[jj].tolist())
                        #print(ca_c[jj],label_c[jj])
                        #if len(label_c[jj]) != 17:
                        #    print('penis',len(label_c[jj]))
                        label.append(label_c[jj])
                        #print(len(label[jj]))

        #print(label[:10])

        ca = np.array(ca)
        cb = np.array(cb)
        frame = np.array(frame)
        try:
            label = np.stack(label)
        except:
            if self.train:
                return -1, -1, -1 ,-1
            else:
                return -1, -1, -1

        if self.train:
            return ca, cb, frame, label
        return ca, cb, frame


    def load_ref(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ref (np.array): reference pdb information
        """
        l = files.split('|')
        if "" in l:
            l.remove("")
        ref = []



        for ii in l:
            #print(self.root_dir + ii)
            curr = np.load(self.root_dir + ii)


            #print(curr)

            #for k in curr.files:
            #    print(k)
            c_ref = curr['ref']
            

            for jj in range(len(c_ref)):
                
                ref.append(c_ref[jj])
                
        
        return ref


    def load_coor_af(self,files):
        """
        Arguments:
            files (string): file names seperated by "|" to coordinate files
        Returns:
            ca (np.array): array of all CA coor
            cb (np.array): array of all CB coor
            frame (np.array): array of all local frames
        """
        l = files.split('|')
        l.remove("")
        ca = [];
        cb = [];
        frame = []

        for ii in l:
            #print(self.root_dir + ii)
            curr = np.load(self.root_dir + ii)
            ca_c = curr['ca']
            cb_c = curr['cb']
            frame_c = curr['frame']

            for jj in range(len(ca_c)):
                ca.append(ca_c[jj])
                cb.append(cb_c[jj])
                frame.append(frame_c[jj])

        ca = np.array(ca)
        cb = np.array(cb)
        frame = np.array(frame)

        return ca, cb, frame

    def get_knn_info(self,coor,frame,num_neigh, eps=[1e-5,1e-5,1e-5] ):
        """
        Arguments:
            coor (arr): coordinates to be analyzed
            frame (arr): local frame information per residue
            num_neigh (arr): array of number of nearest neighbors
        Returns:
            edges (2d array): Edges of all nodes - first index is array num-neigh related
            edge_feats (2d array): Edge features of each edge above - first index is array num-neigh related
        """

        #print(coor[:12,:])
        dist = dm(coor,coor);
        #print(dist[:12,:12])
        dist_sort = np.argsort(dist)
        #print(dist_sort[:12,:12])
        edge1 = [];
        edge2 = [];
        feats = [];

        max_neigh = np.max(num_neigh)

        for i in range(len(num_neigh)):
            edge1.append([])
            edge2.append([])
            feats.append([])

        #go thru all coordinates
        for i in range(len(coor)):

            #1 - range = skip self
            for kk in range(1,max_neigh+1):

                #assert i != dist_sort[i,kk]
                if kk >= len(dist_sort[i]):
                    continue;
                #dont include self
                #Skip self if we get self - just double down and make sure
                if (dist[i,dist_sort[i,kk]] == 0):
                    continue;

                #get RBF
                #distances.append(dist_sort[i,kk])
                dist_from_rbf = dist[i,dist_sort[i,kk]] - self.rbf_dist_means;
                my_rbf = np.exp( -( dist_from_rbf / self.rbf_eps )**2 )

                #get orientation
                orient = np.matmul( frame[i],np.transpose(frame[kk]) )
                o = R.from_matrix(orient)
                quat = o.as_quat()
                #ori.append(o.as_quat())

                #get direction
                vec = coor[dist_sort[i,kk]] - coor[i] + eps;
                vec /= np.linalg.norm(vec);
                direct = np.matmul( frame[i], vec)
                #dir.append(direct);

                #put all info into a single array;
                val = [];
                for jj in my_rbf:
                    val.append(jj)
                for jj in quat:
                    val.append(jj);
                for jj in direct:
                    val.append(jj);

                #append our neighborhood info
                for jj in range(len(num_neigh)):
                    if kk <= num_neigh[jj]:
                        edge1[jj].append(i)
                        edge2[jj].append(dist_sort[i,kk])
                        feats[jj].append(val)

        fake_val = list( np.zeros((23,)) )
        #print(np.shape(fake_val),np.shape(feats[jj]))
        #concatenate them and make them torch-worthy
        #get the num_edges

        n_edges = []
        for ii in edge1:
            n_edges.append(np.shape(ii)[0])
        self.n_edge = n_edges
        #print(n_edges)

        edges = [];
        for jj in range(len(num_neigh)):
            edges.append(torch.stack([ torch.LongTensor(edge1[jj]), torch.LongTensor(edge2[jj]) ]))
            feats[jj] = torch.FloatTensor(feats[jj])

        return edges, feats


def get_loaders(train_file_cluster, train_file_pdb, test_file_cluster, test_file_pdb, root_dir="../",
                batch_size=1, num_workers=0, train_cluster=True, val_cluster=False,
                knn=[6,12,18,24], pin_memory=True, use_pad=False, pad_size=1750):

    train_ds = CSV_Dataset( train_file_cluster,  train_file_pdb, root_dir=root_dir, train=1, use_clusters=train_cluster,nn=knn)
    train_loader = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True )

    val_ds = CSV_Dataset( test_file_cluster,  test_file_pdb, root_dir=root_dir, train=1, use_clusters=val_cluster,nn=knn, val=True)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return train_loader, val_loader


def fix_edges(edges,feats,n_edges):
    """
    Arguments:
        edges (arr): list of padded edges (n_block, n_batch, 2, n_edge)
        feats (arr): list of edge_feats (n_block, n_batch, 2, n_edge)
        n_edges (arr): length of unpadded edges (n_block,n_edge)
    Returns:
        edges (2d array): Edges of all nodes - first index is array num-neigh related
        edge_feats (2d array): Edge features of each edge above - first index is array num-neigh related
    """

    new_edges = []
    new_feats = []

    sz_block = np.shape(edges)
    sz_batch = np.shape(edges[0])
    #print(sz_batch, sz_block, np.shape(n_edges))
    #Go thru each elem in batch
    for ii in range(sz_batch[0]):

        # go thru each block in batch
        new_edges.append([])
        new_feats.append([])

        #Go thru each block
        for jj in range(sz_block[0]):
            #print(np.shape(edges[jj]), edges[jj][ii,:,:])
            new_edges[ii].append(edges[jj][ii,:,:n_edges[jj][ii]])
            new_feats[ii].append(feats[jj][ii,:n_edges[jj][ii]])
    #print(np.shape(new_edges),np.shape(new_feats))
    return new_edges, new_feats


def get_test_loader(test_file_cluster,  test_file_pdb, root_dir="../", train=0,
                batch_size=1, num_workers=0, test_cluster=False,
                knn=[6,12,18,24], pin_memory=True,return_pdb_ref=False):

    val_ds = CSV_Dataset(  test_file_cluster,  test_file_pdb, root_dir=root_dir, train=1,
        use_clusters=test_cluster,nn=knn, val=True, return_name=True, return_pdb_ref=return_pdb_ref)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return val_loader

def f1_metric(y_pred,y_true,cutoff=0.5):
    y_pred = y_pred > cutoff;
    y_true = y_true > cutoff;
    tp = np.sum(y_pred * y_true)
    fp = np.sum(y_pred > y_true)
    fn = np.sum(y_pred < y_true)
    tn = np.sum(y_pred == y_true) - tp

    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1

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
