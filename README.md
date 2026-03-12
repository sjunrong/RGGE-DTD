# RGGE-DTD: A Unified Model for Simultaneous Prediction of Drug-Target Interactions and Drug-Disease Associations in Drug Repositioning

RGGE-DTD is a model based on RGCN and Gelato for predicting novel DTIs and DDAs.

![image](./overview.jpg)

# Requirements

The project is written in Python 3.7, and all experiments were conducted on a Windows server with an Intel 13th Gen Core i7-13700KF (3.44GHz, 64G RAM) and Nvidia RTX 4080s GPU (16G GPU RAM). For a faster training process, training on a GPU is necessary, but a standard computer without a GPU also works (though it will consume much more training time). We recommend the following hardware configuration:

- RAM= 16G
- VRAM >= 16G

All implementations of RGGE-DTD are based on PyTorch and PyTorch Geometric. RGGE-DTD requires the following dependencies:

- python == 3.10.10
- numpy == 1.24.3
- pandas == 2.2.2
- gensim==4.3.3
- torch == 2.5.0+cu124
- torchvision==0.20.0+cu124
- torchaudio==2.5.0+cu124
- torch_geometric == 2.6.1

# Code and data

## Code

- `hyperparameter.py` : includes all hyperparameters of the model
- `node2vec.py` : manually implemented node2vec algorithm
- `preprocess.py` : preprocessing functions calculate the preprocessed data that needs to be input into the model.
- `RDTD.py` : predicting drug-target interaction and drug-disease association
- `util.py` : User-defined functions
- `model/RDTD.py` : RDTD Model Definition and Implementation
- `eval/evalRDTDAb.py` : Validation and testing functions for the model

## `Luo/`directory

- `drug.txt` : list of drug names
- `protein.txt` : list of protein names
- `disease.txt` : list of disease names
- `drug_smiles.csv` :Drug names and corresponding SMILES
- `protein_fasta.csv`  :Target UniProt ID and amino acid sequence
- `drug_dict_map` : a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map` : a complete ID mapping between protein names and UniProt ID
- `drug_target.dat` : Index list and binary association scores of the Drug-Target interaction matrix
- `drug_dis.dat` : Index list and binary association scores of the Drug-Disease association matrix
- `drug_drug.dat` : Index list and binary association scores of the Drug-Drug interaction
- `protein_dis.dat` : Index list and binary association scores of the Protein-Disease association matrix
- `pro_pro.dat` : Index list and binary association scores of the Protein-Protein interaction matrix

# Hyperparameter

| α    | β    | η    | lr    | RGCN_layer | MLP_layer | Input_dim | Hidden_dim | Out_dim |
| ---- | ---- | ---- | ----- | ---------- | --------- | --------- | ---------- | ------- |
| 0.0  | 1.0  | 0.0  | 0.001 | 2          | 3         | 128       | 128        | 64      |

## Seed

```
523
```

## Runtime complexity

| RGCN               | Gelato | RGGE-DTD           |
| ------------------ | ------ | ------------------ |
| O(∣V∣⋅d^2+\|E\|⋅d) | O(d^2) | *O*(∣V∣⋅d^2+∣E∣⋅d) |

∣V∣ is the total number of nodes, ∣E∣ is the total number of edges across all relation types, and d*d* is the hidden embedding dimension.

# Run RGGE-DTD

Obtain the preprocessing file and execute the following command.

```
python preprocess.py
```

Run the RGGE-DTD model and execute the following command.

```
python main.py
```

# **Code Availability**

The source code and data used in this study are available at https://github.com/sjunrong/RGGE-DTD under the MIT license.

# Contacts

If you have any questions or comments, please feel free to email Junrong Song(zz2144@ynufe.edu.cn).