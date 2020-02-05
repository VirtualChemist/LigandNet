# LigandNet
LigandNet, a tool which combines different machine learning models into one platform for the prediction of the state of the ligands either actives or inactives for a particular proteins.

# Setup
Create a conda environment using `environment.yml`. Run the following
```bash
conda env create -f environment.yml
```

# Run
Use `ligandnet.py` to run predictions. To see the available options, run `python ligandnet.py --help` which shows the following:

```bash
usage: ligandnet.py [-h] [--sdf SDF] [--smiles SMILES]
                    [--confidence CONFIDENCE]

Ligand activity prediction using LigandNet

optional arguments:
  -h, --help            show this help message and exit
  --sdf SDF             SDF file location
  --smiles SMILES       SMILES
  --confidence CONFIDENCE
                        Minimum confidence to consider for prediction. Default
                        is 0.5
```

For example, `python ligandnet.py --smiles CCCC` will run all the LigandNet models on the compound `CCCC`. For an sdf file as input, run `python ligandnet.py --sdf samples/AAAAML.xaa.sdf`. The parameter `confidence` is the minimum probability for which a model will consider a ligand as an active.

An example output is the following
```
{'Cmpd1': {'Q9Y2D0': 0.65, 'P14867': 0.67, 'P51787': 0.54, 'P07288': 0.52, 'P47869': 0.67, 'P47870': 0.87, 'P34903': 0.93, 'Q99685': 0.99, 'O75762': 0.99, 'Q05823': 0.6, 'Q12884': 0.53, 'Q16445': 0.98, 'P02708': 0.63, 'Q13882': 0.53, 'P08922': 0.51, 'P22748': 0.74}}
```
This is a dictionary with the compound id `Cmpd1` and it's activity to proteins. Proteins are listed with their uniprot ids and the corresponding probability/confidence for activity prediction.

# Decoys
To get the decoys used for training the LigandNet models, run 

```bash
1. bash get_decoys.sh
2. tar xvf decoys.tar.gz
```

# Web server
A web interface for ligand activity prediction using the LigandNet models is available at [LigandNet](https://drugdiscovery.utep.edu/ligandnet)


# Docker
LigandNet is available on [DockerHub](https://hub.docker.com/repository/docker/sirimullalab/ligandnet). To run LigandNet, do

```bash
1. docker pull sirimullalab/ligandnet
2. docker run sirimullalab/ligandnet --help
```
