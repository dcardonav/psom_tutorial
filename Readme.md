# Aggregated Power System Optimization Models


## System Requirements:
- Python 3 (tested on 3.9 but should work on 3.10 too)
- Libraries: look at the environment configuration file
- A LP solver, even an **open source** one like GLPK (https://www.gnu.org/software/glpk/)

## Installation Instructions

These instructions are meant to be hassle-free and do not constitute the best practices; those who know Python can look at the environment.xml file and manually install the requirements. In case you don't have a solver installed in your machine, please follow the instructions to install and configure GLPK.

### Installing a Solver on Windows
1. Go to https://sourceforge.net/projects/winglpk/
2. Download the latest version of the GLPK binaries
3. Extract the contents in the folder of your choice
4. Add the w64 folder from the glpk installation to the *Path* system variable, as indicated in the picture

### Installing Python
1. Download and install the Anaconda distribution (https://www.anaconda.com/)
2. Open an Anaconda command line: Start -> Anaconda Command
3. Navigate to the folder where environment.xml file is located
4. Create the environment: conda create --name winter_school --file environment.xml
5. Open Jupyter: Start -> Jupyter (winter_school)

### Installing the Environment 
For those who are familiar with Python and do not want to make a complete installation, an environment.xml file is provided which contains the required packages.

### Checking the Installation
1. Open jupyter and select the 00_Python_Basics.ipynb
2. Press Ctrl+Enter to run each cell, if the code runs without any errors then you're ready for the tutorial session.


This is a work-in-progress, so feedback is greatly appreciated. Have fun!


**Author**: David Cardona-Vasquez \
**Copyright**: Copyright 2022, Graz University of Technology \
**Status**: Development 