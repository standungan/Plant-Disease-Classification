import torch
import yaml
import torchvision
from tqdm import tqdm
from torchvision import transforms
from utils.data import imageFolderDataloader


if __name__ == "__main__":

    # Load the configuration file
    with open("experiments/config_exp01.yaml") as f:
        config = yaml.safe_load(f)

    # Define the transformations to be applied to the images
    trainTransform = transforms.Compose([
        transforms.Resize(config['DATASET']['INPUT_SIZE']),
        transforms.CenterCrop(config['DATASET']['INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['DATASET']['MEAN'], 
                             std=config['DATASET']['STD'])
    ])
    testTransform = transforms.Compose([
        transforms.Resize(config['DATASET']['INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['DATASET']['MEAN'], 
                             std=config['DATASET']['STD'])
    ])


    # Loading The Dataset
    trainDataloader = imageFolderDataloader(
        data_dir=config["DATASET"]["DATASET_DIR"]+"Train",
        transform=trainTransform,
        batch_size=config['DATASET']['BATCH_SIZE'],
        num_workers=config['DATASET']['NUM_WORKERS']
    )
    validDataloader = imageFolderDataloader(
        data_dir=config["DATASET"]["DATASET_DIR"]+"Validation",
        transform=testTransform,
        batch_size=config['DATASET']['BATCH_SIZE'],
        num_workers=config['DATASET']['NUM_WORKERS']
    )

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model, loss function, and optimizer
    if config["MODEL"]["BACKBONE"] == "resnet18":
        model = torchvision.models.resnet18(pretrained=False);
        model.fc = torch.nn.Linear(in_features=512, 
                                   out_features=config["DATASET"]["NUM_CLASSES"])
        model = model.to(device)                                   
    if config["TRAIN"]["OPTIMIZER"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["LEARNING_RATE"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["TRAIN"]["LEARNING_RATE"])

    if config["TRAIN"]["LOSS_FUNCTION"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    epochs = config["TRAIN"]["EPOCHS"]
    for epoch in range(epochs):
        model.train()
        print(f'Epoch [{epoch+1}/{epochs}]')
        for images, labels in tqdm(trainDataloader, desc="Train", ):
            # Forward pass
            images = images.to(device)
            labels = labels.to(device) 
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Loss: {loss.item():.4f}')

        if (epoch+1) == 5:
            # torch.save(model.state_dict(), f'experiment/checkpoint_exp01_{epoch+1}.pth') # Save the model after every epoch
            model.eval() # set model to evaluation mode
            # Define metrics
            test_loss = 0.0
            test_acc = 0.0
            # Loop through test set
            for inputs, targets in tqdm(validDataloader, desc="Validation"):
                # Move inputs and targets to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.no_grad(): # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                # Update metrics
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == targets.data)

            # Compute final metrics
            test_loss /= len(validDataloader.dataset)
            test_acc = test_acc.double() / len(validDataloader.dataset)
            print("Test Loss: {:.4f}, Test Accuracy: {:.4f} %".format(test_loss, test_acc*100))

        print("-"*100)

        

        
        