======
Optga
======

.. image:: https://travis-ci.org/horoiwa/optga.svg?branch=master
    :target: https://travis-ci.org/horoiwa/optga
.. image:: https://img.shields.io/badge/python-3.7-blue
    :target: https://img.shields.io/badge/python-3.7-blue
.. image:: https://img.shields.io/badge/license-MIT-blue
    :target: https://spdx.org/licenses/MIT

**Finding out optimal inputs of your machine learning model in order to balance multipule objective variables in trade-off relationship**

Overview
========

Optga is framework to find out optimal input of your machine learning model in order to achieve desired outcomes.

Especially useful when one input two or more objective variables.

Such situation is often the case in **scientific problems**, including novel materials development.

    *Our dataset has three obejctive variables; Y1, Y2, and Y3.
    Fortunately, they are well predicted by machine learing models.*

    *Then, how can we optimize these objective variables simultaneously?*


Key features
============

* Rapid multi-objective optimization by genetic algorithm (NSGA2)

* Support constraints specific to machine learning inputs.
  (e.g. Onehot constraint)

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
