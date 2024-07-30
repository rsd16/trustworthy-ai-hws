#i add noise to whole data, i don't do dataset enlargening. is it ok?

######################################################################################################
# our imports
######################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import umap.plot
import copy
from umap import UMAP
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from skimage.util import random_noise
from pytorch_metric_learning import losses, miners
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# taken from https://github.com/khornlund/pytorch-balanced-sampler
from pytorch_balanced_sampler.sampler import SamplerFactory


######################################################################################################
# our imports
######################################################################################################


######################################################################################################
# our constants
######################################################################################################


BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001 # learning rate

# the noises which we will utilize:
noise_types = ['simple noise', 'gaussian noise', 'speckle', 'color jitter', 'poisson', 'salt and pepper', 'fgm']

# the transformations we will implement on our dataset at the beginning of loading them:
transform = transforms.Compose([transforms.ToTensor()])

# the device on which our codes will run:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################################################################################################
# our constants
######################################################################################################


######################################################################################################
# helper functions and class (write once, use forever)
######################################################################################################


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def create_model():
    # load the model without the weights from imagenet dataset:
    # i could've implemented the model from scratcg. the approach below is faster and better.
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    print(model)

    # total parameters and trainable parameters:
    # taken from https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'Total parameters: {total_parameters:,}')

    total_trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f'Total trainable parameters: {total_trainable_parameters:,} ')

    return model

def convert_and_make_dataset(x, y, convert=True, dataset=True):
    if convert and not dataset:
        # convert our numpy arrays to torch tensor and create dataset and data loader from them:
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y
    elif dataset and not convert:
        # convert torch tensor to torch dataset and then torch data loader:
        set_ = TensorDataset(x, y)
        loader = DataLoader(set_, batch_size=64, shuffle=True)
        return loader
    elif convert and dataset:
        # do both of the above at the same time:
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        set_ = TensorDataset(x, y)
        loader = DataLoader(set_, batch_size=64, shuffle=True)
        return x, y, loader

# training function:
def train(model, loader, optimizer, criterion, device, adv_train=False, noise_types=None):
    print('Training')

    # set the model on training mode, so the gradients can be updated:
    model.train()

    # a counter to efficiently calculate loss:
    counter = 0

    # to calculate loss and accuracy for whole epoch:
    train_running_loss = 0.0
    train_running_correct = 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        # a counter to calculate loss and accuracy efficiently:
        counter += 1

        # read data, images and labels, and move them to gpu:
        images, labels = data
        labels = labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        # if we watn to have adversarial examples during training:
        if adv_train:
            # the procedure to do adversarial training is inspired by https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
            # perform noise on only half of the datapoints in the batch...
            # ...we chhoose noise type randomly:
            random_indices = np.random.randint(images.shape[0], size=int(images.shape[0]/2))
            for index in random_indices:
                noise_type = random.choice(noise_types)
                if noise_type == 'fgm':
                    images = fast_gradient_method(model, images, 0.3, np.inf) # eps=0.3
                else:
                    images[index,:] = add_noises(images[index,:], noise_types)

        # preparing for training and validation:
        optimizer.zero_grad()

        # forward-pass:
        outputs = model(images)

        # calculating the loss:
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # calculating the accuracy:
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation:
        loss.backward()

        # updating the weights:
        optimizer.step()

    # loss and accuracy for the whole epoch:
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(loader.dataset))

    return epoch_loss, epoch_acc

# training function, specific for angular loss:
def train_angular(model, loader, optimizer, criterion, device, mining_func):
    print('Training')

    # set the model on training mode, so the gradients can be updated:
    model.train()

    # loss for whole epoch:
    train_running_loss = 0.0

    # a counter to efficiently calculate loss:
    counter = 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        # a counter to calculate loss and accuracy efficiently:
        counter += 1

        # read data, images and labels:
        images, labels = data

        # to prove that in each batch we have equal distribution of datapoints for each of the labels:
        #unique, counts = np.unique(labels, return_counts=True)
        #print(dict(zip(unique, counts)))

        # move the images and labels in appropriate format to gpu:
        labels = labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        # preparing for training and validation:
        optimizer.zero_grad()

        # forward-pass:
        outputs = model(images)

        # normalize embedding maps using l2 normalization:
        # inspired by https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209
        outputs = F.normalize(outputs, p=2, dim=1)

        # calculating the loss:
        # inspired by https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb
        indices_tuple = mining_func(outputs, labels)
        loss = criterion(outputs, labels, indices_tuple)
        train_running_loss += loss.item()

        # backpropagation:
        loss.backward()

        # updating the weights:
        optimizer.step()

    # loss and accuracy for the whole epoch:
    epoch_loss = train_running_loss / counter

    return epoch_loss

# validation function:
def validate(model, loader, criterion, device, noise_type=None):
    print('Validation')

    # set the model to evaluation mode:
    model.eval()
    
    # loss and accuracy for whole validation or test set:
    valid_running_loss = 0.0
    valid_running_correct = 0

    # a counter to efficiently calculate loss:
    counter = 0

    # we don't want gradients being updated:
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            counter += 1

            # read data, images and labels, and move them on gpu:
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images = images.to(device)
            labels = labels.to(device)

            # this is specific to question 1-3, we add noises to the whole data:
            if noise_type is not None:
                images = add_noises(images, noise_type)

            # forward-pass:
            outputs = model(images)

            # calculating the loss:
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # calculating the accuracy:
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the whole epoch:
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(loader.dataset))

    return epoch_loss, epoch_acc

# validation function, specific for angular loss:
def validate_angular_without_accuracy(model, loader, criterion, device, mining_func):
    print('Validation')

    # set the model on training mode, so the gradients can be updated:
    model.eval()
    
    # loss for whole validation or test set:
    valid_running_loss = 0.0

    # a counter to efficiently calculate loss:
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            counter += 1

            # read data, images and labels, and move them to gpu:
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images = images.to(device)
            labels = labels.to(device)

            # forward-pass:
            outputs = model(images)

            # normalize embedding maps using l2 normalization:
            # inspired by https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209
            outputs = F.normalize(outputs, p=2, dim=1)

            # calculating the loss:
            # inspired by https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb
            indices_tuple = mining_func(outputs, labels)
            loss = criterion(outputs, labels, indices_tuple)
            valid_running_loss += loss.item()

    # loss and accuracy for the whole epoch:
    epoch_loss = valid_running_loss / counter

    return epoch_loss

# whole learning process:
def main(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, adv_train=False, noise_types=None):
    # noise_types is a list of noises which we will utilize in training. we will mix and choose...
    # ...randomly among them during training (adversarial training).

    # efficient codes for main process of learning (training and validation, also testing) are inspired from https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/

    # lists to keep track of losses and accuracies during training:
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    # start of the training process:
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch} of {EPOCHS}')

        # train the model:
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device, adv_train, noise_types)

        # validate the model:
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)

        # save results in corresponding lists:
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)

        # show results for each epoch:
        print(f'Training Loss: {train_epoch_loss:.3f}, Training Accuracy: {train_epoch_acc:.3f}')
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}')
        print('-' * 50)

    # plot accuracy progress for both training and validation:
    plt.plot(train_acc, label='Training Accuracy', linewidth=3)
    plt.plot(valid_acc, label='Validation Accuracy', linewidth=3)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best', fontsize=14)
    plt.show()

    # plot loss progress for both training and validation:
    plt.plot(train_loss, label='Training Loss', linewidth=3)
    plt.plot(valid_loss, label='Validation Loss', linewidth=3)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best', fontsize=14)
    plt.show()

    # evaluate the model on train-set:
    train_epoch_loss, train_epoch_acc = validate(model, train_loader, criterion, device)
    print(f'Training Loss: {train_epoch_loss:.3f}, Training Accuracy: {train_epoch_acc:.3f}\n')

    # evaluate the model on validation-set:
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
    print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')

    # evaluate the model on test-set and show the results:
    test_epoch_loss, test_epoch_acc = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_epoch_loss:.3f}, Test Accuracy: {test_epoch_acc:.3f}\n')

# whole learning process, specific for angular loss:
def main_angular(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, mining_func=None):
    # lists to keep track of losses and accuracies during training:
    train_loss = []
    valid_loss = []

    # start of the training process:
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch} of {EPOCHS}')

        # train the model:
        train_epoch_loss = train_angular(model, train_loader, optimizer, criterion, device, mining_func)

        # validate the model:
        valid_epoch_loss = validate_angular_without_accuracy(model, valid_loader, criterion, device, mining_func)

        # save results in corresponding lists:
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        print(f'Training Loss: {train_epoch_loss:.3f}')
        print(f'Validation Loss: {valid_epoch_loss:.3f}')
        print('-' * 50)

    # plot loss progress for both training and validation:
    plt.plot(train_loss, label='Training Loss', linewidth=3)
    plt.plot(valid_loss, label='Validation Loss', linewidth=3)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best', fontsize=14)
    plt.show()

    # evaluate the model on train-set:
    train_epoch_loss = validate_angular_without_accuracy(model, train_loader, criterion, device, mining_func)
    print(f'Training Loss: {train_epoch_loss:.3f}\n')

    # evaluate the model on validation-set:
    valid_epoch_loss = validate_angular_without_accuracy(model, valid_loader, criterion, device, mining_func)
    print(f'Validation Loss: {valid_epoch_loss:.3f}\n')

    # evaluate the model on test-set:
    test_epoch_loss = validate_angular_without_accuracy(model, test_loader, criterion, device, mining_func)
    print(f'Test Loss: {test_epoch_loss:.3f}\n')

def get_feature_extractor(model_temp, type_='remove linear and avgpool'):
    '''we mostly have two approaches to remove fully-connected layers from our model if we use object of torchvision.'''

    if type_ == 'identity':
        # set linear and fully-connected layers of the model as a proxy which passes the data unchanged:
        # averagepooling layers will not be deleted:
        # taken from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
        #model.avgpool = Identity()
        model_temp.fc = Identity()
    elif type_ == 'remove linear and avgpool':
        # outright remove the whole fully-connected and averagepooling layers (preferred way):
        # inspired from https://stackoverflow.com/a/52548419
        model_temp = torch.nn.Sequential(*(list(model_temp.children())[:-2]))

    return model_temp

def umap_validation(model, x, y, model_type='remove linear and avgpool', phase='Test', noise_type=None):
    # we first construct an array to hold all the feature maps, later utilized in umap:
    if model_type == 'remove linear and avgpool':
        feature_maps = np.empty((y.shape[0], 512, 1, 1)) # if we don't use Identity class.
    elif model_type == 'identity':
        feature_maps = np.empty((y.shape[0], 512)) # if we use Identity class (averagepooling layers will not be deleted).

    # set the model to evaluation mode since, we don't want to deal with any gradient updating processes:
    model.eval()

    for i in range(x.shape[0]):
        image = x[i].to(device)

        # to convert the single image format from (3, 32, 32) to (1, 3, 32, 32), a batch of 1 image:
        image = image.unsqueeze(0)

        # if we want to deal with noisy data, we add noise to whole data:
        if noise_type == 'fgm':
            image = fast_gradient_method(model, image, 0.3, np.inf) # eps=0.3
        elif noise_type is not None:
            image = add_noises(image, noise_type)

        feature_map = model(image)

        # append the feature maps corresponging each datapoint in a numpy array:
        feature_maps[i, :] = feature_map[0].cpu().detach().numpy() # save resulting feature maps.

    print(feature_maps.shape)

    # the umap package accepts =< 2 dimensions. so, we have to reshape the feature maps in this format...
    # ...the format is: (#datapoints, channel * width * height)...
    # ...we use these lines of codes in case we don't use Identity class (which is done on only fully-connected layers, not averagepooling layer):
    if model_type == 'remove linear and avgpool':
        feature_maps = feature_maps.reshape((feature_maps.shape[0], feature_maps.shape[1] * feature_maps.shape[2] * feature_maps.shape[3]))
    
    print(feature_maps.shape)

    # umap object, n_components is 2, because we want to map it into a 2-d image.
    reducer = UMAP(n_components=2)

    # we fit and train the umap on feature maps to reduce the dimensions:
    components = reducer.fit(feature_maps)
    print(components)

    # plot them in most presentable way possible:
    umap.plot.points(components, labels=y, theme='fire')
    plt.title(f'Feature Map for {phase} after UMAP')
    plt.show()

def add_noises(x, noise_type):
    if noise_type == 'simple noise':
        # taken from https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/15.Pytorch%20Robust%20Deep%20learning%20Neural%20Network%20with%20Adding%20Noise.ipynb#scrollTo=TZuumDV85HYm
        noise = torch.randn(x.shape).to(device)
        new_images = x + noise
        x = new_images
    elif noise_type == 'gaussian noise':
        # taken from https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=bvS4INxv9yBf
        new_images = random_noise(x.cpu().detach().numpy(), mode='gaussian', mean=0, var=0.05, clip=True)
        x = torch.from_numpy(new_images).float().to(device)
    elif noise_type == 'speckle':
        # taken from https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=bvS4INxv9yBf
        new_images = random_noise(x.cpu().detach().numpy(), mode='speckle', mean=0, var=0.05, clip=True)
        x = torch.from_numpy(new_images).float().to(device)
    elif noise_type == 'color jitter':
        # taken from https://www.tutorialspoint.com/pytorch-randomly-change-the-brightness-contrast-saturation-and-hue-of-an-image
        color_jitter = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))
        new_images = color_jitter(x)
        x = new_images.to(device)
    elif noise_type == 'poisson':
        # https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        values = len(torch.unique(x))
        values = 2 ** np.ceil(np.log2(values))
        new_images = np.random.poisson(x.cpu().detach().numpy() * values) / float(values)
        x = torch.from_numpy(new_images).float().to(device)
    elif noise_type == 'salt and pepper':
        # taken from https://colab.research.google.com/github/ashishpatel26/Awesome-Pytorch-Tutorials/blob/main/13.%20Pytorch%20Image%20Data%20for%20Deep%20learning%20Data%20Augmentation.ipynb#scrollTo=bvS4INxv9yBf
        new_images = random_noise(x.cpu().detach().numpy(), mode='salt', amount=0.05)
        x = torch.from_numpy(new_images).float().to(device)

    return x

def validate_fgm(model, loader, criterion, device):
    # this function is inspired by https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py

    print('Validation')

    # set the model to evaluation mode:
    model.eval()

    # loss and accuracy for whole validation or test set:
    valid_running_loss = 0.0
    valid_running_correct = 0

    # a counter to efficiently calculate loss:
    counter = 0

    # we don't disable gradients movements, si
    for i, data in tqdm(enumerate(loader), total=len(loader)):
        counter += 1

        # read data, images and labels, and move them on gpu:
        images, labels = data
        labels = labels.type(torch.LongTensor)
        images = images.to(device)
        labels = labels.to(device)

        # we want our attack on data to be of "fast gradient method" type:
        images = fast_gradient_method(model, images, 0.3, np.inf) # eps=0.3

        # forward-pass:
        outputs = model(images)

        # calculating the loss:
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()

        # calculating the accuracy:
        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the whole epoch:
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(loader.dataset))

    return epoch_loss, epoch_acc

def batch_sampler_dataset(x, y, mode):
    # a list of 10 (number of labels) sub-lists. each sub-list corresonds to indices that belong to that label:
    class_idxs = [list(np.where(y == i)[0]) for i in range(10)]
    #print(class_idxs)

    # we have 12000 datapoints in training phase. we want the batch size to be 60 (6 datapoints of each labels)...
    # ...so the n_batches would be 12000 / 60 = 200
    # for vlidation set: n_batches parameter: 7200 / 60 = 120.
    # for test-set: n_batches parameter: 40800 / 60 = 680.
    # we therefore use mode parameter to appropriately assign n_batches parameter for batchsampler:
    if mode == 'train':
        n_batches = 200
    elif mode == 'valid':
        n_batches = 120
    elif mode == 'test':
        n_batches = 680

    # a custom batch-sampler taken from https://github.com/khornlund/pytorch-balanced-sampler
    # by setting alpha to 1 we want the batch sampler to be exact in...
    # ...selecting exact and same number of datapoints from each label for each batch.
    batch_sampler = SamplerFactory().get(class_idxs=class_idxs, batch_size=60, n_batches=n_batches, alpha=1, kind='fixed')

    # preparing our training data for this very section using batchsampler:
    set_ = TensorDataset(x, y)
    loader = DataLoader(set_, batch_sampler=batch_sampler)

    return set_, loader

def validate_angular_with_accuracy(model_temp, feature_maps_train, yy_train, xx_valid, yy_valid, xx_test, yy_test, noise_type=None):
    # we first construct arrays for all three learning phases to hold all the feature maps, later utilized in knn classfication:
    # the second shape is 64, since the last fully-connected layer in our model has 64 units.
    feature_maps_valid = np.empty((yy_valid.shape[0], 64))
    feature_maps_test = np.empty((yy_test.shape[0], 64))

    for i in range(xx_valid.shape[0]):
        image = xx_valid[i].to(device)

        # to convert the single image format from (3, 32, 32) to (1, 3, 32, 32), a batch of 1 image:
        image = image.unsqueeze(0)

        # if we want to deal with noisy data, we add noise to whole data:
        if noise_type == 'fgm':
            image = fast_gradient_method(model_temp, image, 0.3, np.inf) # eps=0.3
        elif noise_type is not None:
            image = add_noises(image, noise_type)

        feature_map = model_temp(image)
        #print(feature_map)

        # append the feature maps corresponging each datapoint in a numpy array:
        feature_maps_valid[i, :] = feature_map[0].cpu().detach().numpy() # save resulting feature maps.

    for i in range(xx_test.shape[0]):
        image = xx_test[i].to(device)

        # to convert the single image format from (3, 32, 32) to (1, 3, 32, 32), a batch of 1 image:
        image = image.unsqueeze(0)

        # if we want to deal with noisy data, we add noise to whole data:
        if noise_type == 'fgm':
            image = fast_gradient_method(model_temp, image, 0.3, np.inf) # eps=0.3
        elif noise_type is not None:
            image = add_noises(image, noise_type)

        feature_map = model_temp(image)
        #print(feature_map)

        # append the feature maps corresponging each datapoint in a numpy array:
        feature_maps_test[i, :] = feature_map[0].cpu().detach().numpy() # save resulting feature maps.

    # we first check the shape of our resulting feature maps:
    print(f'{feature_maps_train.shape = }')
    print(f'{feature_maps_valid.shape = }')
    print(f'{feature_maps_test.shape = }')

    # we construct an object for our KNN classifier:
    # number of neighbours is inspired by https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb
    knn_model = KNeighborsClassifier(n_neighbors=1)

    # we train the model on resulting feature maps for passing the training data from the model...
    # ...the labels are the same ground-truth values:
    knn_model.fit(feature_maps_train, yy_train)

    # the model first predicts and is being evaluated on validation set:
    y_pred = knn_model.predict(feature_maps_valid)

    # show the accuracy score:
    print(f'Accuracy of the model on Validation-set is {accuracy_score(yy_valid, y_pred)}')

    # the model then predicts and is being evaluated on test set:
    y_pred = knn_model.predict(feature_maps_test)

    # show the accuracy score:
    print(f'Accuracy of the model on Test-set is {accuracy_score(yy_test, y_pred)}')

######################################################################################################
# helper functions (write once, use forever)
######################################################################################################


######################################################################################################
# q1-1
######################################################################################################


# we produce our original dataset. we don't use dataloader, because we like to be able to...
# ...modify the data in any way we please.

# download CIFAR10 dataset and assign it for training:
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# save all datapoints in numpy arrays for later uses:
x_train = np.array([np.array(datapoint[0]) for datapoint in train_set])
y_train = np.array([np.array(datapoint[1]) for datapoint in train_set])

print(f'{x_train.shape = }')
print(f'{y_train.shape = }')

# use existing and downloaded CIFAR10 dataset and assign it for test:
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

x_test = np.array([np.array(datapoint[0]) for datapoint in test_set])
y_test = np.array([np.array(datapoint[1]) for datapoint in test_set])

print(f'{x_test.shape = }')
print(f'{y_test.shape = }')

# gather all of the datapoints in corresponding numpy arrays for better train/validation/test splits:
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

print(f'\n{x.shape = }')
print(f'{y.shape = }')

# to shuffle the arrays a bit, to add randomness:
p = np.random.permutation(len(x))

x = x[p]
y = y[p]

# split the datapoints into train (20%) and test (80%):
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, stratify=y, shuffle=True)

# split the datapoints into validation (15%) and test (85%):
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.85, stratify=y_test, shuffle=True)

# train: 20% of all data
# validation: 12% of all data
# test: 68% of all data

print(f'\n{x_train.shape = }')
print(f'{y_train.shape = }')

print(f'{x_valid.shape = }')
print(f'{y_valid.shape = }')

print(f'{x_test.shape = }')
print(f'{y_test.shape = }')

# to prove that the datapoints are evenly distributed, class-wise:
print('\nFirst for training phase:')
uniques, counts = np.unique(y_train, return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(y_train)))
print(f'{percentages = }')

print('\nThen, for validation phase:')
uniques, counts = np.unique(y_valid, return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(y_valid)))
print(f'{percentages = }')

print('\nLastly, for test phase:')
uniques, counts = np.unique(y_test, return_counts=True)
percentages = dict(zip(uniques, counts * 100 / len(y_test)))
print(f'{percentages = }\n')

# preparing our data for coming codes:
x_train, y_train, train_loader = convert_and_make_dataset(x_train, y_train, convert=True, dataset=True)
x_valid, y_valid, valid_loader = convert_and_make_dataset(x_valid, y_valid, convert=True, dataset=True)
x_test, y_test, test_loader = convert_and_make_dataset(x_test, y_test, convert=True, dataset=True)


######################################################################################################
# q1-1:
######################################################################################################


######################################################################################################
# q1-2:
######################################################################################################


# our model:
model = create_model()

# our optimizer and loss:
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# doing the whol learning process, training, validation and testing:
main(model, train_loader, valid_loader, test_loader, optimizer, criterion, device)

# to get feature extractor (convolutional layers):
reduced_model = get_feature_extractor(copy.deepcopy(model), type_='identity')
print(reduced_model)

# now, we are going to evaluate the reduced model by visualizing the backbone feature maps. we will get help from umap.

# first for training set:
umap_validation(reduced_model, x_train, y_train, model_type='identity', phase='Training')

# second for validation set:
umap_validation(reduced_model, x_valid, y_valid, model_type='identity', phase='Validation')

# then, for test set:
umap_validation(reduced_model, x_test, y_test, model_type='identity', phase='Test')


######################################################################################################
# q1-3:
######################################################################################################


# we will iterate through our noise types and show the performance of our model on each:
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    if noise_type == 'fgm':
        # evaluate the model on validation-set with "fast gradient method" attack and show the results:
        valid_epoch_loss, valid_epoch_acc = validate_fgm(model, valid_loader, criterion, device)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')

        # evaluate the model on test-set with "fast gradient method" attack and show the results:
        valid_epoch_loss, valid_epoch_acc = validate_fgm(model, test_loader, criterion, device)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')
    else:
        # evaluate the model on validation-set with noise and show the results:
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device, noise_type)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')

        # evaluate the model on test-set with noise and show the results:
        test_epoch_loss, test_epoch_acc = validate(model, test_loader, criterion, device, noise_type)
        print(f'Test Loss: {test_epoch_loss:.3f}, Test Accuracy: {test_epoch_acc:.3f}\n')

# now, we are going to evaluate the model by visualizing the backbone feature maps. we will get help from umap.
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    # first for noisy validation set:
    umap_validation(reduced_model, x_valid, y_valid, model_type='identity', phase='Noisy Validation', noise_type=noise_type)

    # then, for noisy test set:
    umap_validation(reduced_model, x_test, y_test, model_type='identity', phase='Noisy Test', noise_type=noise_type)


######################################################################################################
# q1-3:
######################################################################################################


######################################################################################################
# q1-4:
######################################################################################################


# our model:
model = create_model()

# our optimizer and loss:
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# doing the whol learning process, training, validation and testing...
# ...first, we will test on clean validation and test dataset:
main(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, adv_train=True, noise_types=noise_types)

# we will iterate through our noise types and show the performance of our model on each:
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    if noise_type == 'fgm':
        # evaluate the model on validation-set with "fast gradient method" attack and show the results:
        valid_epoch_loss, valid_epoch_acc = validate_fgm(model, valid_loader, criterion, device)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')

        # evaluate the model on test-set with "fast gradient method" attack and show the results:
        valid_epoch_loss, valid_epoch_acc = validate_fgm(model, test_loader, criterion, device)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')
    else:
        # evaluate the model on validation-set with noise and show the results:
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device, noise_type)
        print(f'Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {valid_epoch_acc:.3f}\n')

        # evaluate the model on test-set with noise and show the results:
        test_epoch_loss, test_epoch_acc = validate(model, test_loader, criterion, device, noise_type)
        print(f'Test Loss: {test_epoch_loss:.3f}, Test Accuracy: {test_epoch_acc:.3f}\n')

# to get feature extractor (convolutional layers):
reduced_model = get_feature_extractor(model, type_='identity')
print(reduced_model)

# now, we are going to evaluate the model by visualizing the backbone feature maps. we will get help from umap.
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    # first for noisy validation set:
    umap_validation(reduced_model, x_valid, y_valid, model_type='identity', phase='Noisy Validation', noise_type=noise_type)

    # then, for noisy test set:
    umap_validation(reduced_model, x_test, y_test, model_type='identity', phase='Noisy Test', noise_type=noise_type)


######################################################################################################
# q1-4:
######################################################################################################


######################################################################################################
# q1-6:
######################################################################################################


# preparing our training data for this very section using batchsampler:
train_set, train_loader = batch_sampler_dataset(x_train, y_train, 'train')
valid_set, valid_loader = batch_sampler_dataset(x_valid, y_valid, 'valid')
test_set, test_loader = batch_sampler_dataset(x_test, y_test, 'test')

# our model:
model = create_model()

# since we're dealing with metric learning, the last layer of the model should be a fully-connected layer...
# ...with the output size of 128 or 64 as for embedding map size.
model.fc = nn.Linear(512, 64)
model = model.to(device)
print(model)

# our optimizer and loss:
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# these two lines of codes are inspired by the official documentation of pytorch metric learning...
# https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#angularminer
# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss
# the parameters are set to their respective default values.
criterion = losses.AngularLoss(alpha=40)
miner_func = miners.AngularMiner(angle=20)

# doing the whole learning process, training, validation and testing, all the data from these phases are clean:
main_angular(model, train_loader, valid_loader, test_loader, optimizer, criterion, device, mining_func=miner_func)
print(model)

# to get feature extractor (convolutional layers):
reduced_model = get_feature_extractor(copy.deepcopy(model), type_='identity')
print(reduced_model)

# we are going to evaluate the model on both validation and test data accuracy-wise:
# this process is inspired by the posts and videos we read and watched for understanding metric learning...
# ...better, since we haven't had experience in this field. also, we got help from Abbas Nosrat and...
# ...he guided us on overall approach.

# first we pass the train data through our model once and for all, we will use the resulting feature-maps...
# ...for all sorts of evaluation later down the line:

# set the model to evaluation mode since, we don't want to deal with any gradient updating processes:
model.eval()

# we first construct array for train phase to hold all the feature maps, later utilized in knn classfication:
# the second shape is 64, since the last fully-connected layer in our model has 64 units.
# we compute feature maps for training once and for all...
feature_maps_train = np.empty((y_train.shape[0], 64))

for i in range(x_train.shape[0]):
    image = x_train[i].to(device)

    # to convert the single image format from (3, 32, 32) to (1, 3, 32, 32), a batch of 1 image:
    image = image.unsqueeze(0)

    feature_map = model(image)
    #print(feature_map)

    # append the feature maps corresponging each datapoint in a numpy array:
    feature_maps_train[i, :] = feature_map[0].cpu().detach().numpy() # save resulting feature maps.

# now we evaluate the model on clean validation and test data using the feature maps:
validate_angular_with_accuracy(copy.deepcopy(model), feature_maps_train, y_train, x_valid, y_valid, x_test, y_test, noise_type=None)

# now, we are going to evaluate the reduced model by visualizing the backbone feature maps. we will get help from umap.
# all the data in this section are clean and not noisy.

# first for training set:
umap_validation(reduced_model, x_train, y_train, model_type='identity', phase='Training')

# second for validation set:
umap_validation(reduced_model, x_valid, y_valid, model_type='identity', phase='Validation')

# then, for test set:
umap_validation(reduced_model, x_test, y_test, model_type='identity', phase='Test')

# after we evaluated the model on clean data (much like q1-2), we will evaluate the model on noisy data...
# ...(much like q1-3). for this we will have:

# we will iterate through our noise types and show the performance of our model on each:
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    # evaluate the model on validation-set and test-set with noise and show the results:
    validate_angular_with_accuracy(copy.deepcopy(model), feature_maps_train, y_train, x_valid, y_valid, x_test, y_test, noise_type)

# now, we are going to evaluate the model by visualizing the backbone feature maps. we will get help from umap.
for noise_type in noise_types:
    print(f'\n{noise_type = }\n')

    # first for noisy validation set:
    umap_validation(reduced_model, x_valid, y_valid, model_type='identity', phase='Noisy Validation', noise_type=noise_type)

    # then, for noisy test set:
    umap_validation(reduced_model, x_test, y_test, model_type='identity', phase='Noisy Test', noise_type=noise_type)


######################################################################################################
# q1-6:
######################################################################################################