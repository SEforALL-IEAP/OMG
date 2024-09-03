Mini-Grid optimization module
=================================

The generation system algorithm simulates hybrid mini-grid systems using combinations of PV panels, diesel generator and batteries using an hourly resolution over one year. In each hour, the algorithm seeks to meet the electricity demand using PV, diesel or batteries,  based on the availability of each component at that time and a set of dispatch rules as described `here <https://www.diva-portal.org/smash/get/diva2:1197546/FULLTEXT01.pdf>`_.


.. figure::  images/dispatch.png
   :align:   center


If the simulated configuration meets the user-defined minimum criteria for reliability levels and renewable share of electricity generation over the year, the LCOE of electricity generation is calculated using costs and expected technology life on a component levels. The battery life is calculated using a battery throughput model, based on the maximum number of full cycles possible for the battery type. Inverters and charge controllers are sized based on the other component. To find the optimal sizing of the generations system, a number of different configurations are simulated and the one that can meet the electricity demand and fulfill the criteria at the lowest LCOE  is selected

In order to find the optimal configuration within the solution space, `SciPy <https://scipy.org/>`_'s `Differential Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_ package is used. This is a stochastic optimization algorithm, which is not guaranteed to find the global optimum, but has been found to rapidly find a near-optimal solution.