pyenv local 3.10.17
poetry env use 3.10.17

#pip install jupyter notebook ipykernel
#pip install matplotlib yfinance
#pip install timesfm[pax]

#jupyter notebook
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token="abc123" --port 8888 --NotebookApp.notebook_dir="./"
