import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from matplotlib import pyplot as plt

from src.constants import labels

# Function to visualize n (rows * cols) numbers of images from training or test set
def images_viz(rows, cols, set_length, X, y, set = 'train'):

    fig, axes = plt.subplots(rows, cols, figsize = (20,20))
    fig.suptitle(f"X {set} images", y = 0.95, fontsize = 25)
    for i in np.arange(0, rows * cols):
        axes = axes.flatten()
        img = np.random.randint(0, set_length)
        axes[i].imshow(X[img,])
        label = int(y[img])
        axes[i].set_title(labels[label], fontsize = 15)
        axes[i].axis('off')

    plt.show()

# Function to visualize class distribution in training and test
def class_distribution(y_train, y_test):

    train_label_counts = np.bincount(y_train)
    test_label_counts = np.bincount(y_test)

    label_names = [labels[i] for i in np.arange(0,10)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x = train_label_counts,
            y = label_names,
            name = 'Training set',
            marker_color = 'coral',
            orientation='h'
        ))

    fig.add_trace(
        go.Bar(
            x = test_label_counts,
            y = label_names,
            name = 'Test set',
            marker_color = 'blue',
            orientation = 'h'
        ))

    fig.update_layout(
        title = "Labels distribution for Training and Test set",
        xaxis = dict(
            title = 'Count'
        ),
        yaxis = dict(
            title = 'Labels'    
        ),
        barmode = 'group'
    )

    fig.show()


# Function to visualize histogram distributions (all pixels and single channels) for a single image
def hisogram_distributions(instance, X):

    img = X[instance]

    red_channel = img[:, :, 0].flatten()
    green_channel = img[:, :, 1].flatten()
    blue_channel = img[:, :, 2].flatten()
    all_pixels = img.flatten()


    fig1 = ff.create_distplot(
        [all_pixels],
        group_labels=['All Pixels'],
        show_hist=True,
        show_rug=True,
        colors=['grey']
    )

    fig2 = ff.create_distplot(
        [red_channel, green_channel, blue_channel], 
        group_labels=['Red', 'Green', 'Blue'], 
        show_hist=True, 
        show_rug=True,
        colors=['red', 'green', 'blue']
    )

    fig1.update_layout(title='All combined Pixels Distribution', xaxis_title='Pixel Intensity', yaxis_title='Probability Density')
    fig2.update_layout(title='Channels (R, G, B) Distributions', xaxis_title='Pixel Intensity', yaxis_title='Probability Density')

    fig1.show()
    fig2.show()