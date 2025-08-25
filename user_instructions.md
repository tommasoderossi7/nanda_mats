STEP 1
Connect to the host via ssh
ctrl+shift+p > remote ssh - connect to host

STEP 2
# Create a workspace folder on the remote (run from local terminal)
$ssh nanda_mats 'mkdir -p ~/workspace/nanda_mats'

STEP 3
# To update remote directory (with data not synced in github repo)
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude 'node_modules/' \
  ./nanda_mats/  nanda_mats:~/workspace/nanda_mats/

STEP 4
# install conda-pack (in ANY env, base is fine)
$pip install conda-pack   # or: conda install -c conda-forge conda-pack

STEP 5
# pack the source env, to clone conda and pip packages from source env to the project env
$conda pack -n main -o /tmp/main.tar.gz

STEP 6
# create the new env directory and unpack
$mkdir -p /venv/nanda_mats
$tar -xzf /tmp/main.tar.gz -C /venv/nanda_mats

STEP 7
# fix absolute paths inside the clone
$/venv/nanda_mats/bin/conda-unpack

STEP 8
# activate the env
$conda activate /venv/nanda_mats

STEP 9
# install in the env only the packages listed in requirements that where not present in the base (main) env
$pip install -r requirements.txt --upgrade-strategy only-if-needed




LAST STEP (AFTER THE CODE HAS BEEN MODIFIED REMOTELY OR RESULTS ARE BEING UPDATE IN THE REMOTE VM)
# To update local diretory (with data not synced in github repo)
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' --exclude 'node_modules/' \
  nanda_mats:~/workspace/nanda_mats/  ./nanda_mats/
