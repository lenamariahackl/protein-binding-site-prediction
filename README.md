# Protein - ligand binding residue prediction from sequence using Class Activation Maps

A Implementation of Class Activation maps to deduce residue-level binding prediction from a CNN.

## Setup
For Python 3 but should also work for Python2 :)
To run this project, install it locally using:
```
git clone git@github.com:lenamariahackl/protein-binding-site-prediction.git
cd protein-binding-site-prediction
pip3 install -r requirements.txt
python3 cam_seq.py 
```
The logging files as well as the trained model are available in the log/ folder.

## Project description
A CNN is trained on protein sequence data to predict the binding probability of a protein with classes metal / small / nuclear / peptide. Network weights are used to calculate class activation maps (CAMs) and thereby deduce residue-level information. Binding residues are predicted using a combination of protein-level prediction and the extracted CAMs. 

`![network architecture](pic_net.png)`
(from [tensorflow-class-activation-mapping](https://github.com/philipperemy/tensorflow-class-activation-mapping))
