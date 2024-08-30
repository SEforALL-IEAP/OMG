Overview
=================================

This document serves as a data/process descriptor for the code available at the `project github repo <https://github.com/SEforALL-IEAP/OMG?tab=readme-ov-file>`_. Note that what is included here is far from exhaustive. The aim, instead, it to highlight the main methodological steps as well as to provide a better understanding of the open source code developed to support the modelling exercise. 

.. note::
	Detailed documentation supporting this project is available `here <Add link here when ready>`_.

General info
****************

The methodological approach of this exercise is visually presented below.

.. figure::  images/omg_diagram.jpg
   :align:   center

   Methodological flow and key modelling elements

The core part of the methodology is the **mini-grid optimization model**, which was developed to provide an estimate of the optimal configurations of hybrid solar-diesel-storage systems in order to meet the power requirements in each site or location. The model has been constructed as such to allow for its full customization by the user based on available data, information and other modelling constraints.

A **network layout generator** that has been developed in order to provide an automated way of generating the distribution layout - including trunk line, laterals and service drops - for a given site or location. The methods works without the need of road network data and it is - to the best of our knowledge - the first of its kind type of algorithm in the field. 

With the combination of the mini-grid optimization model and the network layout generator, the user can get an estiation of the bill of quantities within seconds. The model can be configured to work for multiple locations at a time, which makes it a great resource to support electrification planning at scale.

OMG comes as an open source code (available in the form of `jupyter notebooks <https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html>`_) that provides a clear step-by-step description of how to run embeded processes. Sources are linked where needed (e.g. equations, specific values, assumptions etc.) for transparency and open review.

.. note::
	This is a spatial analysis, therefore some processes require either the installation of spatial libraries in python or the use of a GIS (check `QGIS <https://qgis.org/en/site/>`_) environment. It is recommended that the user uses Python >= 3.5 through `anaconda <https://www.anaconda.com/distribution/>`_ distribution; all required python packages are included in the `full_project_environment <https://github.com/SEforALL-IEAP/OMG/blob/main/onssetmg_env.yml>`_ file.


Recommended navigation flow
**************************************
First you may want to follow the instructions for cloning (or downloading) the repository to your local machine. This may include setting up a fresh python (conda) environment using the yml file provided. See `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for more info.

Then, you may want to open the **OMG.ipynb** notebook, which contains the script. Follow the steps in the OMG notebook and you should be able to run the code. Note that OMG is a "user interface"; all functions needed are included in two separate files (hybrids.py & dist_funcs.py).

The **"Input_Data"** folder includes sample files to test the model. Make sure you follow the same format in case you want to run the model for other locations.

Thw **"Output_Data"** include sample results of the model. You should expect similar results when you run the model with own data.