import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from CIFAR10_dataloader import mean_val, std_val, CIFAR10_class
import torch.optim as optim
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Define custom transformation to invert normalization
class InvertNormalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for i in range(3):  # Assuming RGB images
            tensor[i] = (tensor[i] * self.std[i]) + self.mean[i]
        return tensor


def get_invert_transformations():
    # Normalize image
    inv_transforms = A.Normalize([-0.48215841/0.24348513, -0.44653091/0.26158784, -0.49139968/0.24703223],
                                              [1/0.24348513, 1/0.26158784, 1/0.24703223], max_pixel_value=1.0)
    return inv_transforms

def show_sample_images(dataSet, title):
    """
        Function to show some sample images
        Args:
            dataSet (datasets.CIFAR10): The dataset to be shown
            title (str): The title for the images
    """
    fig = plt.figure()
    plt.suptitle(title)

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        img, label = dataSet[i]
        plt.imshow(img, cmap='gray')
        plt.title(dataSet.classes[label])
        plt.xticks([])
        plt.yticks([])


def analyse_dataset():
    """
        Function to analyse a dataset and return the mean and standard deviation
    """
    # Index of the sample image
    idx = 77

    # Download the dataset
    exp_CIFAR = datasets.CIFAR10('./data', train=True, download=True)

    print('------ CIFAR10 Dataset Type and Classes ------')
    print('exp_CIFAR type => ',type(exp_CIFAR.data))

    print('------ CIFAR10 Sample Image - InTensor  ------')
    tns_CIFAR = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    img_t,label = tns_CIFAR[idx]
    plt.imshow(img_t.permute(1,2,0))    # we have to use permute to change the order of the axes from C × H × W to H × W × C to match what Matplotlib expects.
    plt.title(exp_CIFAR.classes[label])
    plt.show()

    print('------ CIFAR10 Dataset MEAN & STD_DEV ------')
    imgs = torch.stack([img_t for img_t ,_ in tns_CIFAR],dim=3)
    imgs.shape
    print('Shape of CIFAR10 =>', img_t.shape, img_t.dtype)
    print('Mean of CIFAR10 => ',imgs.view(3,-1).mean(dim=1))
    print('Std Dev of CIFAR10 => ',imgs.view(3, -1).std(dim=1))

    print('------ CIFAR10 Sample Image - Normalized ------')
    trs_CIFAR = datasets.CIFAR10('./data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616))
    ]))
    img_t,label = trs_CIFAR[idx]
    plt.imshow(img_t.permute(1,2,0))    # we have to use permute to change the order of the axes from C × H × W to H × W × C to match what Matplotlib expects.
    plt.title(exp_CIFAR.classes[label])
    plt.show()

def cuda_availabilty() -> bool:
    """ Returns True if cuda is available, else False """
    return torch.cuda.is_available()

def set_manualSeed(seed):
    """ Function to set manual seed for reproducible results """
    # Sets the seed for PyTorch's Random Number Generator
    torch.manual_seed(seed)
    if cuda_availabilty():
        torch.cuda.manual_seed(seed)

def selectDevice():
    """ Function to select Device """
    using_cuda = cuda_availabilty()
    print("Using CUDA!" if using_cuda else "Not using CUDA.")
    # if so select "cuda" as device for processing else "cpu"
    device = torch.device("cuda" if using_cuda else "cpu")
    return device


def show_augmented_images(train_loader):
    inv_transform = get_invert_transformations()

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    images, labels = next(iter(train_loader))

    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(CIFAR10_class[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

def viewAnalysis(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


""" Function to Get Correct Prediction Count """
def GetCorrectPredCount(pPrediction, pLabels ):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def show_misclassified_images(device, model, test_loader):
    model.eval()
    inv_transform = get_invert_transformations()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CIFAR10_class[target[i]])
                    pred_list.append(CIFAR10_class[pred[i]])

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)


def show_gradcam_images(device, model, test_loader, target_layers):
    model.eval()
    inv_transform = get_invert_transformations()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CIFAR10_class[target[i]])
                    pred_list.append(CIFAR10_class[pred[i]])

    cam = GradCAM(model=model, target_layers=target_layers)

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        input_tensor = missclassified_image_list[index].cpu()
        targets = [ClassifierOutputTarget(CIFAR10_class.index(pred_list[index]))]

        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)

        grayscale_cam = grayscale_cam[0, :]
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        image = np.clip(image, 0, 1)
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)