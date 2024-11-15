# Install
`https://docs.anaconda.com/anaconda/install/linux/`
# Activate
`eval "$(/media/hdddisk/anaconda3/bin/conda shell.bash hook)"`
# Create venv
`conda create --name <VENV_NAME>`
# Install dependency on venv
`conda install <DEPENDENCY_NAME>`
# Connect to jupyter notebook
`conda install -c anaconda ipykernel`
`python -m ipykernel install --user --name=<VENV_NAME>`