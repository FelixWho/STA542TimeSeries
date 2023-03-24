# STA542TimeSeries

## Setup

Note, Python 3.10 was used to set up this repository. You can create the proper environment through `conda` as follows

```
conda create -n sta542 python=3.10
conda activate sta542
```

Install the necessary Python packages

```
pip install -r requirements.txt
```

## Data

F4M1 and Pleth signals are in the repository in their respective CSV files. Not sure why HT included `dataset.zip`, it contains a lot more Pleth and F4M1 signals. It's a big file.

## Tasks

- Implement Hua et al. kernel EDMD
- Implement Sinha et al. online robust EDMD