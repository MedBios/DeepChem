#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:14:01 2018

@author: mpcr
"""

#seq2seq? LSTM#
##Pubchem smiles predict name##
# https://pubchempy.readthedocs.io/en/latest/api.html

#get utilities
import pubchempy as pcp
from pubchemprops.pubchemprops import *
import tensorflow as tf
import tflearn  as tl
import numpy as np
import matplotlib.pyplot as plt
import os
from progressbar import ProgressBar


#Gather data#
#get chemicals via pubchem id
chems = []
for i in range (1000):
    chem = pcp.Compound.from_cid(i+1)
    chem = chem.iupac_name
    chems.append(chem)
print(chems)

#get properties
for chem in chems:
    props = get_second_layer_props(chem, ['IUPAC Name', 'Canonical SMILES'])
    chemprops = []
    chemprops.append(props)
    print(chemprops)

#Convert SMILES to integers --> create intsmiles dataset 
for props in chemprops: 
    input = props
    input = input.lower()
    intprop = []
    for character in input:
        number = ord(character)
        intprop.append(number)
        
#organize the data
dataset = chemprops

print(chemprops)

#build the network#





