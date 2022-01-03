# Nephron_Packing_Voronoi
Nephron packing problem analysis and simulation


METHOD FOR INSTALLING:


1) in a parent directory on your local machine:
	$ git clone https://github.com/trevor-chan/Nephron_Packing_Voronoi.git

2) create a python environment:
	see references for conda environemnt or python virtualenv:
	https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
	https://docs.python.org/3/library/venv.html

3) activate python environment

4) install required libraries:
	$ cd Nephron_Packing_Voronoi
	$ pip install requirements.txt

5) run jupyter to access notebooks and edit code
	$ jupyter-lab


*)
simulation can be run using $ python sim.py

if running image analysis scripts to analyze new data, be sure to edit (i think)
model_predict_func.py to run on cpu, not gpu (if cpu and not gpu)

