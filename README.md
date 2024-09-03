# OnSSET for Mini-Grids (OMG)
Quick techno-economic analysis of potential mini-grid sites. The script is based on [OnSSET](http://www.onsset.org/) model with modifications to accommodate specifications related to the overall viability of mini-grids.

[![Documentation Status](https://readthedocs.org/projects/omg-userguide/badge/?version=latest)](https://omg-userguide.readthedocs.io/en/main/?badge=main)

## Content
This repository contains:
* An environment file (onssetmg_env.yml) needed for setting up a fully functioning python 3.7 environment necessary for the model.
* The OMG code in ipynb format. The file contains necessary steps in order to reproduce results.
* Input/Output sample data for testing.

## Installing and running the clustering notebook

**Requirements**

The OMG module (as well as all supporting scripts in this repo) has been developed in Python 3. We recommend installing [Anaconda's free distribution](https://www.anaconda.com/distribution/) as suited for your operating system. 

**Install the repository from GitHub**

After installing Anaconda you can download the repository directly or clone it to your designated local directory using:

```
> conda install git
> git clone https://github.com/SEforALL-IEAP/OMG.git
```
Once installed, open anaconda prompt and move to your local "OMG" directory using:
```
> cd ..\OMG
```

To be able to run the come (.ipynb) you have to install all necessary packages. "onssetmg_env.yml" contains all of these and can be easily set up by creating a new virtual environment using:

```
conda env create --name onssetmg_env --file onssetmg_env.yml
```

This might take some time. When complete, activate the virtual environment using:

```
conda activate onssetmg_env
```

With the environment activated, you can now move to the main directory and start a "jupyter notebook" session by simply typing:

```
..\OMG> jupyter notebook 
```

## Changelog
**28-February-2024**: Original code base developed <br />
**27-August-2024**: Updated code base published

## Resources
Documentation is available [here](https://omg-userguide.readthedocs.io/en/latest/?badge=latest)

## Credits

**Funding** [SEforALL](https://www.seforall.org/)<br />
**Conceptualization:** [Alexandros Korkovelos](https://github.com/akorkovelos), [Andreas Sahlberg](https://github.com/AndreasSahlberg)<br />
**Methodology:** [Andreas Sahlberg](https://github.com/AndreasSahlberg), [Alexandros Korkovelos](https://github.com/akorkovelos),  [Davide Mazzoni](https://github.com/davidemazzoni2) <br />
**Software:** [Andreas Sahlberg](https://github.com/AndreasSahlberg), [Alexandros Korkovelos](https://github.com/akorkovelos), [Julian Cantor](https://github.com/julcan7)<br />
**Validation:** [Alexandros Korkovelos](https://github.com/akorkovelos), [Andreas Sahlberg](https://github.com/AndreasSahlberg), [Julian Cantor](https://github.com/julcan7), [Cristina Dominguez](https://github.com/cristinadomher) <br />
**Supervision and Advisory support:** Nishant Narayan, Irene Calvé Saborit <br />
