# MiniGridOpt
Quick techno-economic analysis of potential mini-grid sites. The script is based on [OnSSET](http://www.onsset.org/) model with modifications to accommodate specifications related to the overall viability of mini-grids.

## Content
This repository contains:
* An environment .yml file needed for generating a fully functioning python 3.7 environment necessary for the algorithms.
* The mini-grid optimization code in ipynb format. The file contains necessary steps in order to reproduce results.
* Input/Output sample data for the case of Zambia.

## Installing and running the clustering notebook

**Requirements**

The mini-grid module (as well as all supporting scripts in this repo) have been developed in Python 3. We recommend installing [Anaconda's free distribution](https://www.anaconda.com/distribution/) as suited for your operating system. 

**Install the repository from GitHub**

After installing Anaconda you can download the repository directly or clone it to your designated local directory using:

```
> conda install git
> git clone https://github.com/SEforALL-IEAP/MiniGridOpt.git
```
Once installed, open anaconda prompt and move to your local "clustering" directory using:
```
> cd ..\MiniGridOpt
```

In order to be able to run the come (.ipynb) you have to install all necessary packages. "geospatial_env.yml" contains all of these and can be easily set up by creating a new virtual environment using:

```
conda env create --name geospatial_env --file geospatial_env.yml
```

This might take some time. When complete, activate the virtual environment using:

```
conda activate geospatial_env
```

With the environment activated, you can now move to the main directory and start a "jupyter notebook" session by simply typing:

```
..\MiniGridOpt> jupyter notebook 
```

## Changelog
**28-February-2024**: Original code base published

## Resources
To be added

## Credits

**Conceptualization:** [Alexandros Konrkovelos](https://github.com/akorkovelos)<br />
**Methodology:** [Alexandros Konrkovelos](https://github.com/akorkovelos)<br />
**Software:** [Alexandros Konrkovelos](https://github.com/akorkovelos)<br />
**Validation:** [Alexandros Konrkovelos](https://github.com/akorkovelos) <br />
**Supervision and Advisory support:** TBD <br />