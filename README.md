<div align="center">    
 
# AI Security Term Project  
## 'Informative One-Class Classifier'   

<!--  
Conference   
-->   
</div>
 
## Description   
AI Security Term Project for 'Informative One-Class Classifier'       

## How to run   
First, install dependencies   
```bash
# Clone project   
git clone https://github.com/ByeongGil-Jung/2021-AI-Security.git

# Install project   
cd 2021-AI-Security
pip install -e .   
pip install -r requirements.txt
 ```    
If you want to modify configurations, move below directory and modify it. 
 ```bash
# Move configuration directory        
cd config/hyperparameters
```    
Next, navigate to any file and run it.   
 ```bash
# Run module (example: mnist as your main contribution)   
python main.py --model "conv_cae" --data "mnist" --stage "fit" --tqdm_env "script"    
```

## Example code
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from config.factory import HyperparameterFactory
from dataset.factory import DataModuleFactory
from domain.metadata import ModelMetadata
from model.factory import ModelFactory
from trainer.base import TrainerBase

# Arguments
model_name = "conv_cae"
data_name = "mnist"
stage = "fit"
tqdm_env = "script"

model_metadata = ModelMetadata(model_name=model_name, information=None)

# Arguments controller
hyperparameter_factory = HyperparameterFactory.create(data_name=data_name, model_name=model_name)
datamodule_params = hyperparameter_factory.datamodule_params
trainer_params = hyperparameter_factory.trainer_params
model_params = hyperparameter_factory.model_params

# DataModule controller
datamodule = DataModuleFactory.create(data_name=data_name)
datamodule = datamodule(**datamodule_params)

datamodule.prepare_data()
datamodule.setup(stage=stage)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

# Model controller
model = ModelFactory.create(model_name=model_name, model_metadata=model_metadata, model_params=model_params)

# Trainer controller
trainer = TrainerBase(model_metadata=model_metadata, model=model, **trainer_params)

# Training & Validation
if stage == "fit" or stage == "whole":
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

# Testing
if stage == "test" or stage == "whole":
    trainer.test(model=model, test_dataloaders=test_loader)    
```

### Citation   
```
@article{Byeonggil Jung,
  title={2021-AI-Security},
  author={AIR_Lab},
  year={2021}
}
```   
