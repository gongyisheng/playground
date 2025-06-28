# Install
`https://docs.anaconda.com/anaconda/install/linux/`
# Activate
`eval "$(/media/hdddisk/anaconda3/bin/conda shell.bash hook)"`
# List venv
`conda env list`
# Create venv
`conda create --name <VENV_NAME> python=<PYTHON_VERSION>`
# Remove venv
`conda remove --name <VENV_NAME> --all`
# Rename venv
`conda create --name newenv --clone oldenv`
`conda remove --name oldenv --all`
# Set default venv
go to ~/.bashrc or ~/.zshrc and add `conda activate myenv`
# Install dependency on venv
`conda install <DEPENDENCY_NAME>`
# Connect to jupyter notebook
`conda install -c anaconda ipykernel`
`python -m ipykernel install --user --name=<VENV_NAME>`