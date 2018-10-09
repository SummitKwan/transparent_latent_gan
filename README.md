# TL-GAN: transparent latent-space GAN

This is the repository of my three-week project "**Describe as you can tell**: controlled image synthesis and edit using TL-GAN"

This project provides a novel method to ...

- Slides explaining the core ideas of this project are available at 
[this Google Drive link](https://docs.google.com/presentation/d/1OpcYLBVpUF1L-wwPHu_CyKjXqXD0oRwBoGP2peSCrSA/edit#slide=id.p1)
- Video showing the interactive demo will be available soon
- An interactive demo can be found in this Kaggle notebook: [https://www.kaggle.com/summitkwan/tl-gan-demo](https://www.kaggle.com/summitkwan/tl-gan-demo)

## Motivation for this project format:
- **src** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installatio
- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Setup
Clone repository and update python path
```
repo_name=Insight_Project_Framework # URL of your new repository
username=mrubash1 # Username for your personal github account
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180905 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
git push origin $branch_name
```

## Requisites
- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.
```
# Example
- A
- B
- C
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Get Data

### Get IAM handwritting top50 dataset

The dataset used here is the selected subset of the IAM handwritting dataset
that can be downloaded from Kaggle throught the link

https://www.kaggle.com/tejasreddy/iam-handwriting-top50

Note that you have to log in your Kaggle account and download the zip file
 manually and put it under the project folder 
`/data/raw/`

and run `python ./scr/ingestion/process_IAM_top50.py` to extract data

### Get the Transient attribute scenes dataset:

The dataset is described in  
http://transattr.cs.brown.edu

in terminal run `python ./src/ingestion/process_transient_attribute_scenes.py` to download and extract

### Get CelebA, cifar or minist

in terminal run `python ./src/ingestion/process_celeba.py celebA` to download and extract celebA

in terminal run `python ./src/ingestion/process_celeba.py cifar` to download and extract cifar

in terminal run `python ./src/ingestion/process_celeba.py mnist` to download and extract mnist


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint 
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
