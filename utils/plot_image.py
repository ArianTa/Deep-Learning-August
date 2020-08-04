import matplotlib.pyplot as plt


def show_img(dataset, n_images = 25): 
    images, labels = zip(*[(image, label) for image, label in 
                               [dataset[i] for i in range(n_images)]])


    dataset.classes = [format_label(c) for c in dataset.classes]
    classes = dataset.classes
    plot_images(images, labels, classes)

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image
    
def plot_images(images, labels, classes, normalize = True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')
    plt.show()
