# Learned Index with FMI
## 1 Introduction
This repository is partial experiment code for DIFFERENTIALLY PRIVATE LEARNED INDEXES paper.
> Training code has been excluded in this repository.

## 2 Set up
- The version of PyTorch is 2.4.0. For other needed package, install them by conda.
- All trained model weight files are under weights named `best-{CASE}.pth`.
- All datasets are under `datasets/`

## 3 Script explanation
### 3.1 xx_index_generation.py
There are two such files. One is for Cryptε and the other is for SPECIAL. 
Run these two files will generate specific indexes.

### 3.2 overhead_evaluation.py
This file is to evaluate FMI and SPECIAL. After executing, four overhead evaluation csv will be generated. 
They represent absolute and relative of point and range query results.

### 3.3 trade_off_view.py and charts_view.py
Run these files to draw charts for overhead comparison and max error trend with epsilon.

## 4 Execution sequence
two index generation files (3.1) -> overhead evalution (3.2) -> draw charts (3.3)