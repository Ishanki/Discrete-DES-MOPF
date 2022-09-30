# Discrete-DES-MOPF
 Discrete designs for Distributed Energy Systems with Multiphase Optimal Power Flow.
 
## Technical Information
Use main.py under "Code" to run the model with the parameters provided. 
\
It is advised to run the models with the MILP solver CPLEX and NLP solver CONOPT (these are the defaults). 
\
Note that the results from the OPF/MOPF classes (voltage magnitudes, angles)
are all returned in p.u. Please use the bases of these to convert them to SI units. 

#### Dependencies and versions used during testing:
Pyomo 5.7.3  \
Pandas 0.25.1 \
Numpy 1.17.0 \
xlrd 1.2.0  

#### Case study:
The original IEEE EU LV Test Case can be found here: 
\
https://cmte.ieee.org/pes-testfeeders/resources/
\
All the input files for the modified test case, which is used to test the models, are provided in each folder. 

## Preprint:
I. De Mel, O. V. Klymenko, and M. Short, “Discrete Optimal Designs for Distributed Energy Systems with Nonconvex Multiphase Optimal Power Flow,” Apr. 2022, 
\
Available: https://arxiv.org/abs/2209.14354.

## License:
Copyright (c) 2022, Ishanki De Mel. GPL-3.

