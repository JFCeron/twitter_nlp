# -*- coding: utf-8 -*-
"""
The unique downloaded content was split up to facilitate the classification 
process. Here we consolidate it back into a single .csv file

@author: Bananin
"""
import os
import pandas as pd

def consolidate_split (split_root):
    # paste the content files back together
    content = pd.DataFrame()
    for xlsx in os.listdir(split_root):
        df = pd.read_excel(split_root+xlsx)
        content = content.append(df,ignore_index=True)
        
    # store correctly-classified contents
    train = [clase in range(1,7) for clase in content.Clase]
    classified = content.iloc[train]
    # store the rest of the contents for classification
    unclassified = content.iloc[[not b for b in train]]
    
    return classified.reset_index(drop=True), unclassified.reset_index(drop=True)