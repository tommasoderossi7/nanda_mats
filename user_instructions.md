SETUP VM & SYNC CODE-DATA

STEP 1
Connect to the host via ssh
ctrl+shift+p > remote ssh - connect to host

STEP 2
# Create a workspace folder on the remote (run from local terminal)
$ssh nanda_mats 'mkdir -p /workspace/nanda_mats'
OR
# Clone the github directory inside /workspace/ with User interface clicks

STEP 3
# To update remote directory (with data not synced in github repo)
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete  \
 --exclude '.git/' --exclude '__pycache__/' --exclude 'venv/' --exclude 'node_modules/' --exclude '.env' --exclude 'models/' \
  ./nanda_mats/  nanda_mats:/workspace/nanda_mats/

STEP 4
# disable auto-tmux for the cursor agent to be able to do IO operations with the terminal
$touch ~/.no_auto_tmux

STEP 5
# install conda-pack (in ANY env, base is fine)
$pip install conda-pack   # or: conda install -c conda-forge conda-pack

STEP 6
# pack the source env, to clone conda and pip packages from source env to the project env
$conda pack -n main -o /tmp/main.tar.gz

STEP 7
# create the new env directory and unpack
$mkdir -p /workspace/nanda_mats/venv/nanda_mats
$tar -xzf /tmp/main.tar.gz -C /workspace/nanda_mats/venv/nanda_mats

STEP 8
# fix absolute paths inside the clone
$/workspace/nanda_mats/venv/nanda_mats/bin/conda-unpack

STEP 9
# activate the env
$conda activate /workspace/nanda_mats/venv/nanda_mats

STEP 10
# save in a file the dependencies to avoid touching with pip install requirements.txt
$pip freeze > /tmp/pinned.txt

STEP 11
# install in the env only the packages listed in requirements that where not present in the base (main) env (to not break dependencies)
$pip install -r nanda_mats/requirements.txt -c /tmp/pinned.txt --upgrade-strategy only-if-needed

STEP 12
# verify dependency consistency
$pip check




LAST STEP (AFTER THE CODE HAS BEEN MODIFIED REMOTELY OR RESULTS ARE BEING UPDATE IN THE REMOTE VM)
# To update local diretory (with data not synced in github repo)
# Run from the local terminal within the directory that contains the local nanda_mats/ folder
$rsync -avz --progress --delete   --exclude '.git/' --exclude '__pycache__/' --exclude 'venv/' --exclude 'node_modules/' --exclude '.env' --exclude 'models/'   nanda_mats:/workspace/nanda_mats/  ./nanda_mats/



DOWNLOAD HF MODEL

STEP 1
# put hf read token inside a .env file, from within the project directory execute
$printf "HF_TOKEN=<your_hf_token>" > .env

STEP 2 (ONLY FIRST TIME)
# add it to .gitignore
$echo ".env" >> .gitignore

STEP 3
# Load all KEY=VALUE pairs from .env into the environment
$set -a; source .env; set +a

STEP 4
# example command to download a model from HF
$huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --include "config.json" "generation_config.json" "tokenizer*" "model*" "special*"\
  --local-dir ./nanda_mats/models/llama-3.1-8b-instruct --local-dir-use-symlinks False