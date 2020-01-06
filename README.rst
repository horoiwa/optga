==============================================================================
Optga: multi-objective optimizer for ML-model outputs
==============================================================================

.. image:: https://travis-ci.org/horoiwa/optga.svg?branch=master
    :target: https://travis-ci.org/horoiwa/optga
.. image:: https://img.shields.io/badge/python-3.7-blue
    :target: https://img.shields.io/badge/python-3.7-blue
.. image:: https://img.shields.io/badge/license-MIT-blue
    :target: https://spdx.org/licenses/MIT

**Finding out pareto-optimum inputs of your machine learning models in order to balance multipule target variables which in trade-off relationship**

Overview
========

In real world, **machine learning projects dealing with scientific issues** often have two or more target variables which in trade-off relationship.

For example:

* Speed and fuel efficiency in automobiles.

* Lower voltage and higher capacity in Li-ion batteries.

* Wing weight and lift in aircrafts

* Permeability and selectivity in polymer membranes


After you have successfully built predictive models for each target variable,
Optga finds out pareto-optimum inputs of your machine learning models in order to help making a dicision and achieve desired outcomes.

Key features
============

* Rapid multi-objective optimization by genetic algorithm (NSGA2) with numba

* Support various constraints specific to machine learning inputs.
  (e.g. Onehot constraint, descrete constraint)

* Easy user interface

Install
=======

If you use conda, ``conda update numba`` may be required before pip install.

.. code-block:: bash

    pip install optga


Getting started
===============

Exploring the trade-off between house prices and age in Boston dataset.


.. code-block:: python

    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    import optga
    from optga.optimizer import Optimizer

    #: Load boston housing price datasets
    df =  pd.DataFrame(load_boston().data,
                       columns=load_boston().feature_names)
    X = df.drop(["AGE"], 1)
    y_price = pd.DataFrame(boston.target, columns=["Price"])
    y_age = pd.DataFrame(df["AGE"])

    #: create predictive models
    model_price = RandomForestRegressor().fit(X, y_price)
    model_age = RandomForestRegressor().fit(X, y_age)

    optimizer = Optimizer(sample_data=X)

    #: Add scoring function
    optimizer.add_objective("Price", model_price.predict, direction="minimize")
    optimizer.add_objective("Age", model_age.predict, direction="minimize")

    #: Add constraints on explanatory variables
    optimizer.add_discrete_constraint("CHAS", [0, 1])
    optimizer.add_discrete_constraint("ZN", [0, 100])
    optimizer.add_discrete_constraint("RAD", list(range(1, 9)) + [24])

    #: Run optimization and export results
    optimizer.run(n_gen=300, population_size=300)
    optimizer.export_result("boston_result")


See full examples on example/1_BostonHousingPrice.ipynb.

ToDo
====

* Add island strategy as multi-processing GA optimization.
