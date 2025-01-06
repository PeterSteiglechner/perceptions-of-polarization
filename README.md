# Reproducing Manuscript Results

This repository contains the code and data required to reproduce the results presented in our manuscript: **"How opinion variation among in-groups can skew perceptions of ideological polarization"** (submitted).

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Reproducing Results](#reproducing-results)
6. [Citing](#citing)
7. [License](#license)

---

## Requirements
The code was developed and tested using the following dependencies:
- Python 3.11
- NumPy
- Pandas
- Matplotlib and Seaborn
- Scipy
- Jupyter Notebook
- conda

(see requirements.yml)

## Installation
Clone this repository and install the required dependencies:

```bash
$ git clone https://github.com/PeterSteiglechner/perceptions-of-polarization.git
$ cd perceptions-of-polarization
$ conda env create --name percpol --file=requirements.yml
```

## Dataset
Download the dataset from the [European Social Survey (ESS) website](https://ess.sikt.no/en/) and place it in the `inputdata/` directory.


## Usage
To reproduce our analysis in the manuscript, run the code in the notebooks

1. notebook `Sketch_FabricatedData.ipynb` reproduces the sketch of the model procedure in Figure 1.
2. notebook `ExploratoryDataAnalysisClean.ipynb` reproduces Figure 2 and 3 in the manuscript.
3. notebook `AnalysisClean.ipynb` reproduces our analysis of perceived disagreement and polarization in the German climate debate in Figure 3 and 4 in the manuscript.

The files `functions.py` contains helper functions used in the notebooks. 
