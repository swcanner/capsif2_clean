# capsif2_clean
# CAPSIF2 and PiCAP #

![PiCAP](./picap_visual_abstract.jpg)

This is the first iteration and very rough. Will add more later

Libraries needed:
Conda environment with pyrosetta, pytorch, tqdm, numpy, esm, and more - will make a yml soon...


## THIS NEXT STEP MIGHT WORK OR NOT - HAVENT TESTED IT YET ##
### Quick Setup Guide ###
```
We suggest using a conda environment for the installation.
Steps:
>> conda create -n picap
>> conda install python=3.9 -c conda-forge
>> conda install pytorch torchvision cudatoolkit=11.5 -c pytorch -c nvidia -c conda-forge
>> conda install biopython pandas colorama scikit-learn matplotlib tqdm py3dmol -c conda-forge
>> pip install esm
To install pyrosetta, create a ~/.conda (.condarc) file with the following content:
--------------------------------------------------------
channels:
- https://USERNAME:PASSWORD@conda.rosettacommons.org
- defaults
--------------------------------------------------------
>> conda install pyrosetta


The weights of each model are stored on our remote server `data.graylab.jhu.edu/picap/`

Download `picap.pt` and `capsif2.pt` to `capsif2_clean/models_DL/`


## How to run ##
Put all PDB files into the `input_pdb/` directory

>> python preprocess.py

If using PiCAP:
>> python predict_prot.py

If using CAPSIF2:
>> python predict_res.py

the predictions will then be outputted to `output_data/predictions_prot.csv` and `output_data/predictions_res.csv` for PiCAP and CAPSIF2, respectively