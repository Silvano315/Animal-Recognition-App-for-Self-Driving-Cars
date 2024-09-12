import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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