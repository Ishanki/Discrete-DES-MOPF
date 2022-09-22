## Guide to understanding results file names ##

MILP: DES design MILP with DC approximation (used for initialisations)
NLP_comp_slow: PA algorithm (with Algorithm CR)
NLP_comp_fast: PA-H algorithm (with Algorithm CR-H)
MINLP: MINLP solved with SBB

"Smallmod" (Network 1) and "Medmod" (Network 2) are the names of the networks used. 
The files with network parameters are available in the "Code" folder.

The following codes are used to indicate what technologies have been switched on or off (using 0 or 1):
B - Boiler
PV - Solar panels
BR - Boiler
HP - Heat pumps and tanks (combined)

The number at the end of the filename indicates the season.

For example:
MILP_smallmod_wtfDY_B1_PV1_BR1_HP0_results1.xlsx
MILP model for Network 1 with batteries, PVs, and boilers considered (no ASHPs), Season 1 results.


Files with "bounds" in their name will have the upper and lower bounds from all the iterations.
