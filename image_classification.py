import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# Define data transformations for data augmentation and normalization
data_transforms = {
    # training data
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # randomly resize and crop all the input images to 224 x 2245
        transforms.RandomHorizontalFlip(), # random horizontal flip
        transforms.ToTensor(), # transform all the data into a tensor, pytorch accept data in the form of tensor (mandatory)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the data (R, G, B), standard deviation value
    ]),
    # validation data
    'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "dataset"

# Create data loaders
image_datasets = {}
for x in ['train', 'validate']:
    dataset_path = os.path.join(data_dir, x) # generate the full path to the dataset: dataset/train || dataset/validate
    
    # Create an ImageFolder dataset using torchvision's datasets.ImageFolder
    # This loads images from the dataset_path directory and applies the specified transformation
    image_datasets[x] = datasets.ImageFolder(dataset_path, data_transforms[x])
print (image_datasets)

dataloaders = {}
for x in ['train', 'validate']:
    # torch.utils.data.DataLoader is used to create data loaders for the image datasets created earlier (image_datasets['train'] and image_datasets['val']).
    # The batch_size parameter is set to 4, which means each iteration of the data loader will provide a batch containing 4 samples.
    # shuffle=True specifies that the data will be shuffled at the beginning of each epoch (for the 'train' dataset) or before iterating through the data (for the 'val' dataset).
    # num_workers=4 indicates the number of worker processes to use for data loading, which can speed up data retrieval in parallel.

    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
# print (dataloaders)
dataset_sizes = {}
for x in ['train', 'validate']:
    # Calculate the length (number of samples) of the image dataset (image_datasets[x])
    dataset_sizes[x] = len(image_datasets[x])
print (dataset_sizes)

class_names = image_datasets['train'].classes
print (class_names)

# Pre traineed ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    # fc == fully connected
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False
        
# calculate the loss of classification problem
# This loss function is commonly used for classification tasks where the model is required to classify input data into multiple classes.
# Cross-Entropy: It calculates the similarity or dissimilarity between the predicted class probabilities and the actual target class. It is particularly useful when dealing with multi-class classification problems.
# During the training process, you can use this criterion object within the training loop to calculate the loss between the model's predictions and the actual ground truth labels. This loss is then utilized in the optimization process 
criterion = nn.CrossEntropyLoss()
# This is a optimizer. SGD == stochastic gradient descent
# lr == learning rate, 0.001 is the stp size at which the optimizer update the weights duringi training
# small learning rate leads to slower convergence but more precise weight update
# momentum =  It adds a fraction of the previous update to the current update, helping to accelerate gradients in the right direction and dampen oscillations.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Use all parameters

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training the model
# Training loop
num_epochs = 10 # 10 box each time
for epoch in range(num_epochs):
    for phase in ['train', 'validate']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device) # the image
            labels = labels.to(device) # the class name

            optimizer.zero_grad() # Clear the gradient from the previous iteration 

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) 
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")