==============================================================================
Optga: multi-objective optimization for ML predictions
==============================================================================

.. image:: https://travis-ci.org/horoiwa/optga.svg?branch=master
    :target: https://travis-ci.org/horoiwa/optga
.. image:: https://img.shields.io/badge/python-3.7-blue
    :target: https://img.shields.io/badge/python-3.7-blue
.. image:: https://img.shields.io/badge/license-MIT-blue
    :target: https://spdx.org/licenses/MIT

**Finding out pareto-optimum inputs of your machine learning models in order to balance multipule target variables in trade-off relationship**

Overview
========

Real world machine learning projects in **scientific problems** often have two or more target variables in trade-off relationship.

For example:

* Speed and fuel efficiency in automobiles.

* Lower voltage and higher capacity in Li-ion batteries.

* Wing weight and lift in aircrafts

* Permeability and selectivity in polymer membranes


After you have successfully built predictive models for each target variable,
Optga finds out pareto-optimum inputs of your machine learning models in order to help with dicision making and achieve desired outcomes.

Key features
============

* Support constraints specific to machine learning inputs.
  (e.g. Onehot constraint)

* Rapid multi-objective optimization by genetic algorithm (NSGA2)

* Easy user interface

Install
=======

.. code-block:: bash

    pip install optga


Getting started
===============

Exploring the trade-off between house prices and age in Boston dataset.


.. code-block:: python

    import optga
    from optga.optimizer import Optimizer
    optimizer = Optimizer(samples=X)

    #: Add predicive model
    optimizer.add_objective("Price", model_price, direction="minimize")
    optimizer.add_objective("Age", model_age, direction="minimize")

    #: Add constraints on explanatory variables
    optimizer.add_discrete_constraint("CHAS", [0, 1])
    optimizer.add_discrete_constraint("ZN", [0, 100])
    optimizer.add_discrete_constraint("RAD", list(range(1, 9)) + [24])

    #: Run optimization
    optimizer.run(n_gen=300, population_size=500)


See full examples on example/1_BostonHousingPrice.ipynb.
