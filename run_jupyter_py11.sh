pyenv local 3.11.12
poetry env use 3.11.12

#pip install jupyter notebook ipykernel
#pip install matplotlib yfinance
#pip install timesfm[torch]

# jupyter python 11
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token="abc123" --port 8888 --NotebookApp.notebook_dir="./"


