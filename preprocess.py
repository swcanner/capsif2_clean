#preprocess all files in input_dir to output_dir
input_dir = "./input_pdb/"
output_dir = './pre_pdb/'


#load libraries
import os
import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

from pyrosetta.rosetta.protocols.carbohydrates import *
from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.simple_metrics.metrics import *
from pyrosetta.rosetta.core.simple_metrics.composite_metrics import *
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import *

options = """
-beta
-ignore_unrecognized_res
-out:level 100
"""

#-out:level 100

pyrosetta.init(" ".join(options.split('\n')))

#required functions
def get_chain_seq(pose):
    """
    Gets pose sequence by cahin
    Args:
        pose : pyrosetta pose
    Returns:
        chains (arr) : list of all sequences by internal ordering
    """
    chains = [];
    for ii in range(pose.num_chains()):
        r = pose.chain_begin(ii+1)
        c = pose.pdb_info().pose2pdb(r)
        c = c.split(' ')
        while '' in c:
            c.remove('')
        c = c[-1]
        chains.append(c)
    return chains

def get_protchainXYZ(pose,chain_num):

    """
    Args:
        pose : rosetta pose
        chain_num (int): chain number (rosetta numbering)
    Returns:
        p_coor (arr): array of protein coordinates
        p_label (arr): array with residue numbering (pose numbering)
    """

    p_coor = []; #protein coordinates
    p_label = []; # [RESIDUE_NUMBER] - rosetta numbering

    for jj in range(pose.chain_begin(chain_num),pose.chain_end(chain_num)+1):

        res_number = jj;
        num_of_atoms = pose.residue(res_number).natoms()
        for i in range(num_of_atoms):
            atom_name = pose.residue(res_number).atom_name(i+1).strip()
            if atom_name.count('H')> 0:
                continue
            if atom_name.startswith('V')> 0:
                continue
            curr = np.array(pose.residue(res_number).atom(i+1).xyz())
            p_coor.append(curr)
            p_label.append(res_number)

    return p_coor, p_label

def get_chain_coor(pose,chain):

    """
    function to get all CB and CA atom positions of all residues and their local frame of reference

    Args:
        pose : rosetta pose
        chain : rosetta pose chain number (1,2,...)
    Returns:
        cb (arr): array of all CB coordinates - glycine just CA
        ca (arr): array of all CA coordinates
        frame (arr) : array of all local frame ~
            x' = ca - n , y' = (ca - n) x (ca - c) , z' = x' x y'
        ref (arr): array of PDB nomenclature for each residue
    """

    start = pose.chain_begin(chain)
    end = pose.chain_end(chain)

    cb = np.zeros((end-start+1,3))
    ca = np.zeros((end-start+1,3))
    ref_pdb = []
    beta = 0
    num_res = 0

    frame = np.zeros((end-start+1,3,3))

    for ii in range(start,end+1):

        res = pose.residue(ii);
        beta += pose.pdb_info().temperature(ii,1)
        num_res += 1

        if (res.is_protein() == False):
            return [],[],[],[];

        ref_pdb.append(pose.pdb_info().pose2pdb(ii))


        #get atom coordinates
        xyz = res.xyz('N')
        n = np.array([xyz[0],xyz[1],xyz[2]])
        xyz = res.xyz('CA')
        a = np.array([xyz[0],xyz[1],xyz[2]])
        xyz = res.xyz('C')
        c = np.array([xyz[0],xyz[1],xyz[2]])
        b = a

        name = res.name1();
        if name != "G":
            xyz = res.xyz('CB')
            b = np.array([xyz[0],xyz[1],xyz[2]])

        #get reference frame
        ca_n = a - n;
        ca_n /= np.linalg.norm(ca_n);
        x_prime = ca_n;
        ca_c = a - c;
        ca_c /= np.linalg.norm(ca_c);
        y_prime = np.cross(ca_n,ca_c);
        y_prime /= np.linalg.norm(y_prime);
        z_prime = np.cross(x_prime,y_prime);
        z_prime /= np.linalg.norm(z_prime);

        #explcitly define as
        #        [ -x'- ]
        # ref =  [ -y'- ]
        #        [ -z'- ]
        ref = np.zeros((3,3));
        ref[0,:] = x_prime;
        ref[1,:] = y_prime;
        ref[2,:] = z_prime;

        #update
        cb[ii-start,:] = b;
        ca[ii-start,:] = a;
        frame[ii-start,...] = ref

    return cb, ca, frame, ref_pdb, beta / float(num_res)

def rosetta_preprocess(f,output_dir):

    """
    function preprocess a specific file using pyrosetta to get coordinates and sequence
    outputs a file

    Args:
        f : pdb file (str)
        output_dir : where the output file will be dumped to (str)
    Returns:
        out_fasta : fasta sequence of all chains (arr)
    """

    pose = pose_from_file(f)

    chains = get_chain_seq(pose)
    p = f.split('/')[-1].split('.')[0] #get the name of the file

    #print(ii,len(pdbs),f)
    out_fasta = []
    #go thru all protein chains
    nc = pose.num_chains();
    num_res = 0;
    beta = [];

    for c in range(1,nc+1):

        #only protein chains allowed
        if pose.residue(pose.chain_begin(c)).is_protein() == False:
            continue;

        

        coor, label = get_protchainXYZ(pose,c)
        cb, ca, frame, ref_pdb, b = get_chain_coor(pose,c)
        beta.append(b)

        if len(cb) == 0:
            continue;

        seq = pose.chain_sequence(c)
        n = p + "_" + str(c)
        out_fasta.append([n,seq])

        

        #output the coor file
        np.savez(output_dir + n + ".npz",ca=ca,cb=cb,frame=frame,ref=ref_pdb)

    return out_fasta, beta

def esm_preprocess(fa,model,alphabet,batch_converter,output_dir):

    """
    function preprocess a series of sequences with ESM

    Args:
        fa : fasta information [[name,seq]] (arr)
        model : ESM-2 Model (model)
        alphabet : ESM-2 alphabet (alphabet)
        batch_converter : ESM-2 preprocessor (converter)
        output_dir : where the output file will be dumped to (str)
    Returns:
        void
    """

    for ii in range(len(fa)):
        #print(ii,fa[ii])

        batch_labels, batch_strs, batch_tokens = batch_converter([fa[ii]])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        seq_rep = []
        for i, tokens_len in enumerate(batch_lens):
            seq_rep.append(token_representations[i, 1 : tokens_len - 1])

        #output to file
        name = fa[ii][0]
        np.save(output_dir + name + "_esm.npz",seq_rep[0].numpy())

    return;

if __name__ == '__main__':

    #load ESM Model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    #get all the files
    ls = os.listdir(input_dir)

    if not os.path.isfile(output_dir + 'files_done.txt'):
        f = open(output_dir + 'files_done.txt','w+')
        f.write("")
        f.close()

    #only ouptut files that haven't been made yet
    f = open(output_dir + 'files_done.txt','r+')
    done = []
    l = f.readlines()
    for ii in l:
        done.append(ii)
    f.close()

    f = open(output_dir + 'files_done.txt','a+')
    fasta = open(output_dir + 'fasta.fa','a+')

    print('preprocessing...')

    for i in tqdm(range(len(ls))):
        ii = ls[i] 

        if ('.pdb' in ii):
            p = ii.split('/')[-1].split('.')[0] #get the name of the file

            #only do new ones
            if p in done:
                continue;
            done.append(p)
            f.write(p + '\n')
            print(p)
            
            try:
                fa, beta = rosetta_preprocess(input_dir + ii, output_dir)
                esm_preprocess(fa,model,alphabet,batch_converter,output_dir)


                for i in range(len(fa)):
                    fasta.write('>' + fa[i][0] + '|' + str(beta[i]) + '\n' + fa[i][1] + '\n')

            
            except:
                print("unable: ",ii)


    f.close();
    fasta.close()

    print('making CSVs for file input')

    #only ouptut files that haven't been made yet
    ls = os.listdir(output_dir)
    #print(ls)

    #np.savez(output_dir + n + ".npz",ca=ca,cb=cb,frame=frame,ref=ref_pdb)
    #np.save(output_dir + name + "_esm.npz",seq_rep[0].numpy())

    #out = "Cluster,PDB,coor_files,esm_files,AF2_files,carb,sm\n"
    out = ''

    cl = 'CLUST,PDB1|PDB2\n'

    done = [];
    for p in ls:
        #only grab npz
        if '.npz' not in p:
            continue;
        if 'esm' in p:
            continue;
        if 'DS_Store' in p:
            continue;
        #just double down
    
        name = p.split('.')[0];
        short_name = name[:name.rfind('_')]
        
        if short_name in done:
            continue;
        done.append(short_name)


        chains = []
        for ii in ls:
            #print(ii)
            if 'esm' in ii:
                continue;
            if short_name in ii:
                n = ii.split('.')[0];
                c = n[n.rfind('_'):];
                chains.append(c)

        #print(chains)

        cl += name + '|' + name + '\n'

        #out += name + ',' + name + ',' + output_dir + name + '.npz,'
        #out += output_dir + name + '_esm.npz.npy,,,\n'

        out += name + ',' + name + ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '.npz'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','
        for jj in range(len(chains)):
            out += output_dir + short_name + chains[jj] + '_esm.npz.npy'
            if jj + 1 == len(chains):
                continue;
            out += '|'
        out += ','

        out += ',,\n'

    output_file = output_dir + 'dataset'
    f = open(output_file + "_pdb.csv",'w+')
    f.write(out)
    f.close()

    f = open(output_file + "_clust.csv",'w+')
    f.write(cl)
    f.close();

    print('outputted to: ', output_file + "_pdb  and " + output_file + "_clust .csv")

