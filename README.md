# Perceptions of polarization 

This repository contains the code required to reproduce the results and figures in the manuscript: **"How opinion variation among in-groups can skew perceptions of ideological polarization"** (submitted on 2025-01-06).

See prepint for a detailed description.



## Requirements
The code was developed and tested using the following dependencies:
- Python 3.11
- NumPy
- Pandas
- Matplotlib and Seaborn
- Scipy
- Jupyter Notebook

(see requirements.yml)

## Installation
Download this repository and install the required dependencies:

```bash
$ conda env create --name percpol --file=requirements.yml
```

## Dataset
Download the dataset from the [European Social Survey (ESS) website](https://ess.sikt.no/en/) and place it in the `inputdata/` directory. Here, we used waves 8, 10SC, and 11 of the German subset.



## Usage
To reproduce the analysis in the manuscript, run the code in the notebooks

1. notebook `Sketch_FabricatedData.ipynb` reproduces the sketch of the model procedure in Figure 1.
2. notebook `ExploratoryDataAnalysisClean.ipynb` reproduces Figure 2 and 3 in the manuscript.
3. notebook `AnalysisClean.ipynb` reproduces our analysis of perceived disagreement and polarization in the German climate debate in Figure 4 and 5 in the manuscript.

The files `functions.py` contains helper functions used in the notebooks. 
