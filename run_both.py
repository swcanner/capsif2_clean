## Prediction Utils ##
## Makes it easy for us to just predict literally everything in a single command :) ##

"""
Created on Jan 14 2025

@author: swcanner

 Current settings for B Factor visualization:
 `BFactor =  0.0` : Nonbinder
 `BFactor = 99.9` : CAPSIF2 Predicted binding residue


Usage: python run_both.py
    --capsif_only [run capsif2 only | default: False]
    --picap_only [run picap only | default: False]
    --high_plddt [run only on high_plddt residues (with > `plddt_cutoff`) only | default: False]
    --plddt_cutoff [cutoff for when `--high_plddt` invoked | default: 70]
Returns: `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions
    `./output_data/predictions_prot.tsv` - tsv of picap predictions
    `./output_data/all_predictions.tsv` - tsv of capsif2 and picap predictions together
                only done if picap and capsif2 are run together
    and `output_data/*_predictions.pdb` with the pdbs with the BFactor identifying the binding CAPSIF2 Residues

"""

RUN_PICAP = True;
RUN_CAP = True;
HIGH_PL = False;
PL_CUT = 70;

OUTPUT_INT_TO_CMD = False;
OUTPUT_CMD = True;
OUTPUT_CAP2_PDBS = True;
SINGLE = False;


import sys
import os

def str_to_bool(a):
    #Changes string to bool for flags
    a = a.upper()
    b = True
    if (a == "0" or a == "F" or a == "FALSE" or a == "N" or a == "NO"):
        b = False;
    return b

def manage_flags(flags):
    #Get all flags all organized
    n = len(flags)

    input_flags = ["--capsif_only","--picap_only",'--high_plddt','--plddt_cutoff','--single','--help']

    RUN_PICAP = True;
    RUN_CAP = True;
    HIGH_PL = False;
    PL_CUT = 70;
    SINGLE = False;

    if n > 1:
        for kk in input_flags:
            if kk in flags:
                ind = flags.index(kk);

                if (kk == '--capsif_only'):
                    RUN_PICAP = False
                if (kk == '--picap_only'):
                    RUN_CAP = False
                if (kk == '--high_plddt'):
                    HIGH_PL = True
                if (kk == '--plddt_cutoff'):
                    a = flags[ind+1]
                    a = float(a)
                    PL_CUT = a
                if (kk == '--single'):
                    SINGLE = True

                if (kk == '--help'):
                    print("""
PiCAP and CAPSIF2 help:

    Usage: python run_both.py
        --capsif_only [run capsif2 only | default: False]
        --picap_only [run picap only | default: False]
        --high_plddt [run only on high_plddt residues (with > `plddt_cutoff`) only | default: False]
        --plddt_cutoff [cutoff for when `--high_plddt` invoked | default: 70]
    Returns: `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions
        `./output_data/predictions_prot.tsv` - tsv of picap predictions
        `./output_data/all_predictions.tsv` - tsv of capsif2 and picap predictions together
                    only done if picap and capsif2 are run together
        and `output_data/*_predictions.pdb` with the pdbs with the BFactor identifying the binding CAPSIF2 Residues
                    """)
                    exit()

    print("Running with the following flags: ")
    print("Run PiCAP : ",RUN_PICAP)
    print("Run CAPSIF2: ",RUN_CAP)
    print("Run High pLDDT only: ",HIGH_PL)
    if HIGH_PL:
        print("pLDDT cutoff: ",PL_CUT)

    return RUN_PICAP, RUN_CAP, HIGH_PL, PL_CUT, SINGLE



from preprocess import *
from predict_res import *
from predict_prot import *



#init(" ".join(options.split('\n')))
import os
import numpy as np
import pandas as pd
from utils import *
from egnn.egnn import *
from utils_model import *
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor

import os

SPECIES = 'TEST_FILE'
TEST_PDB =   './pre_pdb/dataset_pdb.csv'
TEST_CLUST = './pre_pdb/dataset_clust.csv'

NUM_WORKERS = 0; #This is for cpu use; will auto change to 8 if you are using gpu

RUN_PICAP = True;
RUN_CAP = True;
HIGH_PL = False;
OUTPUT_INT_TO_CMD = True;
OUTPUT_CMD = True;
OUTPUT_CAP2_PDBS = True;
SINGLE = False; #for use only by the notebook!

if SINGLE:
    TEST_PDB =   './pre_pdb/dataset_single_pdb.csv'
    TEST_CLUST = './pre_pdb/dataset_single_clust.csv'

def run_capsif2(TEST_PDB,TEST_CLUST):

    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        res_label (2d arr, string): predicted residues of the associated pdbs
    Outputs:
        `./output_data/predictions_res.tsv` - tsv of capsif2 residue predictions

    """

    print("Loading Capsif2...")

    BATCH_SIZE = 1;
    NUM_WORKERS = 0;

    #Hyper parameters!
    #LOSS_FN = dice_ent_loss
    #loss_str = "_loss-dbce"
    LOSS_FN = nn.BCELoss()
    KNN = [16,16,16,16]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    loss_str = ''
    CUTOFF = 0.001

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)

    #os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    #Load Test Dataset
    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN, return_pdb_ref=True)


    model = CAPSIF2_RES2(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,normalize=False,
                    device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()
    #print("Model loaded")
    #my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) #+ '_coef_' + str(MY_COEF[0][0]) + "-" + str(MY_LOSS_EPOCHS) + "_all"

    my_model_name = 'capsif2'

    #print(my_model_name)

    if DEVICE == 'cuda':
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
    else:
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(checkpoint['info'])

    print("Capsif2 loaded")

    model.eval()
    pred_res, names, res_label  = model_test_res_env(test_loader, model, DEVICE=DEVICE)

    names = np.array(names)

    file = "./output_data/predictions_res.tsv"
    print('\n\t------Capsif2 results-------')

    f = open(file,'a+')
    for ii in range(len(names)):
        #print(names[ii][0])
        f.write(names[ii][0] + '\t')
        if OUTPUT_INT_TO_CMD:
            print(names[ii][0],end=":")
        for jj in range(len(res_label[ii])):
            if OUTPUT_INT_TO_CMD:
                print(res_label[ii][jj][0], end=",")
            f.write(res_label[ii][jj][0] + ',')
        f.write('\n')
        if OUTPUT_INT_TO_CMD:
            print()

        if OUTPUT_CAP2_PDBS:
            #output the CAPSIF2 predicted residues pdb

            #need to get full name of the input pdb
            ls = os.listdir('./input_pdb/')
            the_input_pdb_file = ''
            #print(names[ii][0])
            #print(names[ii][0].split('_'))
            #print(ls)
            for jj in ls:
                if names[ii][0].split('_')[0] in jj:
                    the_input_pdb_file = jj
                    break;

            if the_input_pdb_file == '':
                print("something messed up in output of:",names[ii].split('_')[0])
            output_structure_bfactor(file='./input_pdb/' + the_input_pdb_file,res=pred_res_to_str(res_label[ii]),
                         out_file= './output_data/' + names[ii][0] + '_predictions.pdb')

    f.close()


    return names, res_label

def run_picap(TEST_PDB,TEST_CLUST):
    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        prot_pred (arr, float): predicted probability of carb binding
    Outputs:
        "./output_data/predictions_prot.tsv" - tsv of picap predictions
    """

    print("\n\n\nloading PiCAP")
    NUM_WORKERS = 0;

    #Hyper parameters!
    KNN = [10,20,40,60]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    ADAPOOL_SIZE = (150,HIDDEN_NF)
    loss_str = ''
    NUM = ''

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)
    print(DEVICE)

    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN)

    model = PICAP(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,
                    device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    my_model_name = 'picap'

    if DEVICE == 'cuda':
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
    else:
        checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("PiCAP loaded")

    print("Running Predictions...")

    model.eval()
    prot_pred, names  = model_test_prot_env(test_loader, model, DEVICE=DEVICE)

    prot_pred = np.array(prot_pred)
    prot_pred = prot_pred.reshape((-1))

    names = np.array(names)

    file = "./output_data/predictions_prot.tsv"
    #out = 'PDB_NAME,pred\n'
    print('\n\t------PiCAP results-------')
    out = ''
    #print(file)
    for ii in range(len(names)):
        if OUTPUT_INT_TO_CMD:
            print(names[ii][0],',', str(prot_pred[ii]))
        #print(prot_pred[ii])
        out += str(names[ii][0]) + '\t' + str(round(prot_pred[ii],4)) + '\n'
    if not os.path.exists(file):
        out = 'PDB_NAME\tpred\n'
    f = open(file,'a+')
    f.write(out)
    f.close()
    return names, prot_pred

def run_it_all(RUN_CAP=True,RUN_PICAP=True,single=False):

    """
    Runs all arguments in single function to do capsif2 and picap
    Arguments:
        single: used by notebook for running a single file
    Returns:
        names_cap (arr, string): all the input pdb names from capsif2
        names_pi  (arr, string): all the input pdb names from picap
        cap_pred (2d arr, string): predicted residues of the associated pdbs
        pi_pred (arr, float): predicted probability of protein-carb binding
    """

    TEST_PDB =   './pre_pdb/dataset_pdb.csv'
    TEST_CLUST = './pre_pdb/dataset_clust.csv'
    if single:
        TEST_PDB =   './pre_pdb/dataset_single_pdb.csv'
        TEST_CLUST = './pre_pdb/dataset_single_clust.csv'

    names_cap, cap_pred = [], []
    names_pi, pi_pred = [], []
    if RUN_CAP:
        names_cap, cap_pred = run_capsif2(TEST_PDB,TEST_CLUST)
    if RUN_PICAP:
        names_pi, pi_pred = run_picap(TEST_PDB,TEST_CLUST)

    print('\n\n\n')

    #remove intermediate files
    int_dir = './pre_pdb/'
    ls = os.listdir(int_dir)
    for ii in ls:
        filename, file_extension = os.path.splitext(int_dir + ii)
        if '.npy' == file_extension or '.npz' == file_extension:
            os.remove(int_dir + ii)

    file = './output_data/all_predictions'
    if single:
        file += '_single'
    file += '.tsv'
    txt = ''
    if not os.path.exists(file):
        txt = 'NAME\tBinder_pred\tRes_pred\n'

    if RUN_CAP and RUN_PICAP:
        for ii in range(len(names_pi)):
            for jj in range(len(names_cap)):
                if names_pi[ii][0] == names_cap[jj][0]:
                    txt += names_pi[ii][0] + '\t'
                    txt += str(round(pi_pred[ii],4)) + '\t'
                    for kk in range(len(cap_pred[jj])):
                        txt += cap_pred[jj][kk][0] + ','
                    txt += '\n'
                    break;
    if not single:
        f = open(file,'a+')
        f.write(txt)
        f.close()

    if OUTPUT_CMD:
        print("Total output:")
        print(txt)

    print("\nFin.")

    return names_cap, names_pi, cap_pred, pi_pred

    #return;

if __name__ == "__main__":

    #Maintain for reproducible values across both cpu and gpu
    torch.backends.cuda.matmul.allow_tf32 = True


    RUN_PICAP, RUN_CAP, HIGH_PL, PL_CUT,  SINGLE = manage_flags(sys.argv)

    run_preprocess(HIGH_PL,PL_CUT)
    print("Preprocessing complete\n\n")
    _, _, _, _ = run_it_all(RUN_CAP,RUN_PICAP)
