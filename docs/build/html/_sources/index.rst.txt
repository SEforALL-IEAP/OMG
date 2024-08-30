Welcome to OnSSET for Mini-Grids (OMG) user's guide!
=======================================================

Achieving universal electricity access by 2030 will require a rapid increase in the rate of new connections and in levels of investment, particularly in countries with low level of access to electricity. The challenge is significant. Designing and selecting the optimal electrification approach requires access to reliable data and information regarding electricity resource availability, demand levels, economic activity and functional infrastructure to name a few. 

The paucity of such information may hamper electrification progress. However, this situation is gradually being improved with the increasing availability of new data and analytical tools, especially in the field of geospatial analysis. Geographic Information Systems (GIS) and remote sensing techniques are becoming openly available and can now provide a range of location-specific information that has not been previously accessible. Take for example the field of agriculture, where data is rapidly approaching the scale of ‘big data’. They can provide farm stakeholders with spatial and temporal information about climate and local weather, soil conditions, crop quality, field biodiversity, and crop yields. 

Mini-grids have emerged as a crucial solution, especially in areas where traditional grid expansion is impractical or unreliable. Recognizing their potential, the World Bank estimates that mini-grids could bring electricity to 0.5 billion people within the next decade across thousands of potential sites in the subcontinent. However, assessing the viability of these sites involves considering multiple factors. 

Here we introduce **"OnSSET for Mini-Grids (OMG),"** an open-source Python-based tool that builds upon previous open-source electrification modelling framework efforts for techno-economic optimization, namely the Open Source Spatial Electrification Tool (OnSSET). The real added value of OMG lies in its automated methods to generate distribution network components for multiple sites simultaneously. Leveraging spatial analytics, the tool offers insights into optimal configurations of hybrid solar-diesel-storage systems and provides detailed bill-of-quantities for distribution design. OnSSET-MG offers full customizability of parameters and is designed to adjust to data availability per site. It can be integrated into a national least cost modelling framework or serve as a stand-alone tool for site-specific mini-grid analysis. 

.. figure::  images/omg_sample_results.jpg
   :align:   center

   Example of mini-grid distribution design and key components BoQ. Generation and distribution components required to connect and supply all buildings were automatically generated in a matter of seconds, taking into account the road network layout and local energy resource availability


This document presents an analytical and spatially explicit approach for estimating aspects of sizing mini-grids in data-deprived areas. The underlying work has been supported by `SEforALL <https://www.seforall.org/programmes/universal-integrated-energy-plans>`_ and is part of a multi-year `GEAPP <https://energyalliance.org/>`_ funded project aiming to facilitate the execution of geospatial electrification planning activities in various sub-Saharan African countries. 

The following sections provide a brief overview of this approach with indicative examples for the main processes included and links to the code developed to implement those processes in an open and reproducible way.


Contents
+++++++++++++++++++++

.. toctree::
   :maxdepth: 3

   Overview
   Environmental resources module
   Electricity Demand module
   Mini-Grid optimization module
   Network layout generator module
   Results
   Credit and Contact


