# Protein - ligand binding residue prediction from sequence using Class Activation Maps

A Implementation of Class Activation maps to deduce residue-level binding prediction from a CNN.

## Setup
For Python 3 but should also work for Python 2 :)
To run this project, install it locally using:
```
git clone git@github.com:lenamariahackl/protein-binding-site-prediction.git
cd protein-binding-site-prediction
pip3 install -r requirements.txt
```

## Example execution
You can train the network from scratch on the example data (parsed from [Biolib](https://zhanglab.ccmb.med.umich.edu/BioLiP/)) with 
```
python3 cam_seq.py
```
Instead train the network from scratch on custom data with 
```
python3 cam_seq.py -i your_input_data
```
Be careful that the folder contains all input files like in the example_input_data folder.

Instead read in a pre-trained model to only make predictions on the test set with
```
python3 cam_seq.py -m pretrained.model
```
Adjust the batch size with
```
python3 cam_seq.py -b 215
```
Log protein-level and residue-level predictions for the test set as well as the trained model with
```
python3 cam_seq.py -l log
```
The files as well as the trained model are available in the log/ folder.

## Project description
A CNN is trained on protein sequence data to predict the binding probability of a protein with classes metal / small / nuclear / peptide. Network weights are used to calculate class activation maps (CAMs) and thereby deduce residue-level information. Binding residues are predicted using a combination of protein-level prediction and the extracted CAMs. 

`![network architecture](pic_net.png)`
(from [tensorflow-class-activation-mapping](https://github.com/philipperemy/tensorflow-class-activation-mapping))

The implementation of the class activation maps is based on https://github.com/zhoubolei/CAM (the implementation of the [paper](http://cnnlocalization.csail.mit.edu/)).
