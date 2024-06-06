Software installation
=====================

Requirements
************

**QGIS**

OnSSET-MG is a spatial electrification tool and as such highly relies on the usage of
Geographic Information Systems (GIS).
A GIS environment is therefore necessary for two main reasons:

* Extract trivial characteristics for the electrification analysis from GIS layers
  and combine them all together in a format easy to read in python
* Visualize the final results in maps.

OnSSET-MG is not dependent on any desktop GIS, however we prefer the open-source QGIS to complement our work!

Download QGIS for free from the official `QGIS website <http://www.qgis.org/en/site/>`_.

**Python - Anaconda package**

OnSSET-MG is written in python, an open source programming language used widely in many applications.
Python is a necessary requirement for the OnSSET-MG tool to work.
Programming in python usually relies on the usage of pre-defined functions
that can be found in the so called modules.
In order to work with OnSSET-MG, certain modules need to be installed/updated.
The easiest way to do so is by installing Anaconda, a package that contains a wide range of
Python packages in one bundle.
Anaconda includes all the Python packages required to run OnSSET-MG successfully.
It can be downloaded and installed for free from the official
`Anaconda website <https://docs.anaconda.com/free/anaconda/install/>`_.


**Python Interfaces - Integrated Development Environment (IDEs)**

**Jupyter notebook (via Anaconda)**

Jupyter notebook is a console-based, interactive computing approach providing a web-based application suitable for capturing the whole computation process: developing, documenting, and executing code, as well as communicating the results. Jupyter notebook is used for the online OnSSET interface, recommended for small analyses and exploring code and results.

**GitHub**

GitHub is a web-based Git repository hosting service. It provides access control and several collaboration features such as bug tracking, feature requests, task management, and wikis for every project. OnSSET-MG is an open source tool therefore the code behind it is open and freely accessible to any user. The code behind the OnSSET-MG tool is is available at SEforALL's Github space. A GitHub account will allow you to propose changes, modifications and upgrades to the existing code. Access the repository on `Github <https://github.com/SEforALL-IEAP>`_.

Software installation and setup
*******************************

1. Download `**Anaconda** here <https://www.continuum.io/downloads>`_ and install.

* Please make sure that you download the version that is compatible with your operating system
  (Windows/MacOS/Linux - In case you run Windows open the *Windows Control Panel*,
  go to *System and Security  System* and check e.g. Windows 32-bit or 64-bit).
* Following the installation process make sure that you click on the option “Add Python X.X to PATH”.
  Also by choosing to customize the installation, you can specify the directory of your
  preference (suggest something convenient e.g. C:/Python35/..).

* After the installation you can use the Anaconda command line (search for “Anaconda Prompt”)
  to run python. It should work by simply writing “python” and pressing enter,
  since the path has already been included in the system variables.
  In case this doesn’t work, you can either navigate to the specified directory and write “python” there,
  or add the directory to the PATH by editing the
  `environment variables <https://www.computerhope.com/issues/ch000549.htm>`_.

2. Download the code from **GitHub** `here <https://github.com/SEforALL-IEAP>`_.


Additional Info
***************

* Basic `navigating commands for DOS (cmd) <https://community.sophos.com/kb/en-us/13195>`_.
* `Modules <https://docs.python.org/3/installing/index.html>`_
  and `packages <https://packaging.python.org/tutorials/installing-packages/>`_
  installation documentation from python.org.