# Templete
<p> This repository is for AI project templete. It is recommanded to use PyTorch 2.0 and PyTorch-Lightning. It uses WandB web logger so please make <a href="https://wandb.ai/home">WandB account.</a> All configs are managed by hydra-core library.</p>
<br>

# Dependencies installation
0. Python version is 3.9
1. Goto ./docs
2. Open terminal and activate conda virtual environment.
3. Use below command for conda enviroment.
```bash
conda env update --file environment.yaml
```
4. Use below command for pip
```bash
pip install -r requirements.txt
```
<br>

## How to login to wandb
1. After you create wandb account, you can see that API token in your setting page. Copy it.
2. Type below command
```bash
wand login
```
3. Past it to terminal. Then, you don't need to login again.

# Execution
## configs
There are several configs you can modify.
* optimizer
1. radam
   ```bash
   ex)
   The default optimizer is RAdam, so you do not need to specifiy it
   python main.py

   You can specify radam. (Default lr and weight_decay are adopted)
   python main.py optimizer=radam
   
   You can specify radam and corresponding lr and weight_decay.
   python main.py optimizer=radam optimizer.lr=0.1 optimizer.weight_decay=0.01
   ```
2. adam
   ```bash
   ex)
   The default optimizer is RAdam, so you have to specifiy it
   python main.py optimizer=adam
   
   You can specify radam and corresponding lr and weight_decay.
   python main.py optimizer=adam optimizer.lr=0.1 optimizer.weight_decay=0.01
   ```
* batch_size
```bash
python main.py dataset.batch_size=128
```
* random seed
```bash
python main.py seed=777777
```
* max epochs
```bash
python main.py max_epochs=200
```
* gpu
1. Single GPU
```bash
python main.py devices=0
```
2. Multi GPUs
```bash
python main.py devices=0,1
```
* logger
```bash
python main.py logger.offline=False # Online mode
python main.py logger.offline=True # Offline mode
```
* model compile
This is only for PyTorch 2.0
```bash
python main.py compile_model=True
```
# File structure
Remember that all of unsued code version should not be removed. Instead, these codes should be at **legacy folder** which you should create for needs.
## config
* config.yaml: All of configs are specified here.

## docs
<p> All of documents should be here </p>

## utils
* utils: Useful utilities are here.
* optimizer: Optimizer which is determined by config file.
* model: Model file
* bcolors: All of color codes are here
* dataset: All of dataset related things will be here.

# How to create enviroment.yaml and requirements.txt
1. Activate virtual environment
2. Execute gen_enviroment.sh(or bat)
3. Execute gen_requirements.sh(or bat)