STEP 1
Connect to the host via ssh
ctrl+shift+p > remote ssh - connect to host

STEP 2
# Create a workspace folder on the remote (run from local terminal)
$ssh nanda_mats 'mkdir -p ~/workspace/nanda_mats'

STEP 3
# To update remote directory
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude 'node_modules/' \
  ./nanda_mats/  nanda_mats:~/workspace/nanda_mats/

STEP 4
# Create remote env if it doesn't exist yet by cloning the base (main) env (with preinstalled torch and other fundamental libraries)
$conda create -y -n nanda_mats --clone main

STEP 5
# activate the (new) env
$conda activate nanda_mats

-----------------------------
# in ANY env (base is fine)
pip install conda-pack   # or: conda install -c conda-forge conda-pack

# pack the source env
conda pack -n main -o /tmp/main.tar.gz

# create the new env directory and unpack
mkdir -p /venv/nanda_mats
tar -xzf /tmp/main.tar.gz -C /venv/nanda_mats

# fix absolute paths inside the clone
/venv/nanda_mats/bin/conda-unpack

# register/use it
conda activate /venv/nanda_mats
-----------------------------


STEP 6
# install in the env only the packages listed in requirements that where not present in the base (main) env
$pip install -r requirements.txt --upgrade-strategy only-if-needed




LAST STEP (AFTER THE CODE HAS BEEN MODIFIED REMOTELY OR RESULTS ARE BEING UPDATE IN THE REMOTE VM)
# To update local diretory
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude 'node_modules/' \
  nanda_mats:~/workspace/nanda_mats/  ./nanda_mats/
