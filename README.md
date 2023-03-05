# Plant Disease Classification
Simple Deep Learning Project for CNN Backbone architecture implementation. This is also can be a template repository for an image classification project using deep learning with PyTorch. The repository contains a basic project structure and code for training and testing classification models.

### Project Structure
The project structure is organized as follows:
```bash
image_classification/
    ├── dataset/
    ├── experiments/
    │   ├── experiment_01/
    |   |   ├── config.yaml
    |   |   ├── checkpoint.pth
    │   ├── experiment_02/
    |   |   ├── config.yaml
    |   |   ├── checkpoint.pth
    ├── models/
    ├── utils/
    ├── train.py
    └── test.py
```
In this structure, there are three main directories:

__dataset/__: This directory is images dataset, current format used is :

    ```bash
    |-- dataset/
    |   |-- train/
    |   |   |-- class1/
    |   |   |-- class2/
    |   |   |-- class3/
    |   |   |-- ...
    |   |
    |   |-- validation/
    |       |-- class1/
    |       |-- class2/
    |       |-- class3/
    |       |-- ...
    ```

__models/__: This directory contains the model files. Each model is defined in a separate Python file. ( __COMING SOON ...__ )

__experiments/__: Each experiment has a configuration file (__config.yaml__), a training script (__train.py__), and a testing script (__test.py__). The configuration file specifies the hyperparameters and other settings for the experiment.

The __utils/__ directory contains utility scripts and functions for working with the data and models.

__TO DO__ :

- Implement test.py
- Implement resume from checkpoint
- Implement scheduler
- Implement early stopping
- Monitoring using Tensorboard
- Custom Dataset for different dataset format
    - version 1
        ```bash
        |-- dataset/
        |   |-- class1/
        |   |-- class2/
        |   |-- class3/
        |   |-- class4/
        ```
    - version 2    
        ```bash
        |-- dataset/
        |   |-- images/
        |   |-- labels.txt or labels.csv
        ```        

        