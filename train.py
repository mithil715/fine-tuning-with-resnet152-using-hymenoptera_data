

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

datatransforms={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
data_dir='hymenoptera_data'
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x ),datatransforms[x])
for x in ['train','val']
}
dataloader={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,
shuffle=True,num_workers=4)
for x in ['train','val']
}
image_sizes={x:len(image_datasets[x]) for x in ['train','val']}
class_name=image_datasets["train"].classes
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if you want to see images of your dataset along with its labels
def imshow(inp, title=None):
    inp=inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    standard=np.array( [0.229, 0.224, 0.225])
    inp=inp*standard+mean
    inp=np.clip(inp,0,1)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / image_sizes[phase]
            epoch_acc = running_corrects.double() / image_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)    
#now in the below the steps there would be a doubt that where are the labels coming from but mostly according to me they are using next in it so that the can take the labels as given the names of the corresponding classes
inputs, labels=next(iter(dataloader["train"]))   
output =torchvision.utils.make_grid(inputs)
imshow(output,title=[class_name[x] for x in labels])

new_model=models.resnet152(pretrained= True)
num_filters=new_model.fc.in_features
new_model.fc=nn.Linear(num_filters, 2)
new_model=new_model.to(device)
criterion =nn.CrossEntropyLoss()
new_optimiser =optim.SGD(new_model.parameters(),lr =0.0001, momentum=0.9)
new_lr_scheduler=lr_scheduler.StepLR(new_optimiser, step_size=7,gamma=0.1)
new_model=train_model(new_model,criterion,new_optimiser,new_lr_scheduler,num_epochs=25  )
