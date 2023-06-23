# Pytorch Lightning + Hydra + DVC + ViT

## Installations Required:
- VS Code
- DVC extension

## Steps to run the code
1. `pip install -e .`.
2. Run `pip install -r requirements.txt`.
3. Check default parameters : `copper_train --help`.
3. To train ViT model : `copper_train experiment=cat_dog data.num_workers=16`.
4. Infer model on cat/dog images: 
    - from projects' default test data folder : `copper_infer experiment=cat_dog`.
    - from any random image from web : `copper_infer experiment=cat_dog imagepath=<image_path>`.

## DVC Setup
- Initialized DVC using `dvc init`. (there should be a .dvc folder created in your project)

- Before start adding the data, make sure that data tracking has been removed from git. Run below two commands to ensure this:
    -   `git rm -r --cached 'data'`
    -   `git commit -m "stop tracking data"`

- Add data to DVC : `dvc add data`.

- To track the changes with git, run: git add data.dvc .gitignore
- To enable auto staging, run: dvc config core.autostage true

You will find a data.dvc file created.

## Integrating local storage with DVC
- Add data to local folder dvc-data: `dvc remote add -d local /workspace/dvc-data`

- Check if a local remote storage has been added by using this command: `dvc remote list` (it will give you list of all remote storage for your project)

- Push the data using : `dvc push -r local`

- Pull data from local storage : `dvc pull -r local`

## Group Members:
- Aman Jaipuria
- Anurag Mittal