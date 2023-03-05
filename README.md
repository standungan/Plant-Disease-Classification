# Plant-Disease-Classification
Simple Deep Learning Project for CNN Backbone architecture implementation... feel free to clone them


In this structure, there are three main directories:

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

        