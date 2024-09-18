import pickle
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Plot a choosen metric from history file
def scatter_plot_metrics(file_name):

    with open("History_models/"+file_name, 'rb') as history_file:
        history = pickle.load(history_file)

    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    fig = go.Figure()

    dropdown_buttons = []
    metric_pairs = {}
    for key in history.keys():
        if key.startswith('val_'):
            metric_name = key[4:] 
            if metric_name in history:
                metric_pairs[metric_name] = (history[metric_name], history[key])
    
    for i, (metric, (train_data, val_data)) in enumerate(metric_pairs.items()):
        train_mean, train_std = calculate_mean_std(train_data)
        val_mean, val_std = calculate_mean_std(val_data)
        fig.add_trace(go.Scatter(
            x=epochs, y=train_data, mode='lines+markers', visible=(i == 0),
            name=f'Train {metric.capitalize()} (Mean: {train_mean:.2f}, Std: {train_std:.2f})'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_data, mode='lines+markers', visible=(i == 0),
            name=f'Val {metric.capitalize()} (Mean: {val_mean:.2f}, Std: {val_std:.2f})'
        ))

        dropdown_buttons.append({
            'label': metric.capitalize(),
            'method': 'update',
            'args': [
                {'visible': [False] * len(metric_pairs) * 2},  
                {'title': f'{metric.capitalize()}'}
            ]
        })
        dropdown_buttons[-1]['args'][0]['visible'][i*2] = True  
        dropdown_buttons[-1]['args'][0]['visible'][i*2 + 1] = True  

    fig.update_layout(
        title=f'Model Performance per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Metric Value',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
        }],
        legend_title="Legend"
    )

    fig.show()


def plot_model_history(history):

    fig = plt.figure(figsize=(12, 16))

    mean_loss, std_loss = calculate_mean_std(history['loss'])
    mean_val_loss, std_val_loss = calculate_mean_std(history['val_loss'])
    plt.subplot(4, 2, 1)
    plt.plot(history['loss'], label=f'Loss (mean: {mean_loss:.3f} ± {std_loss:.3f})')
    plt.plot(history['val_loss'], label=f'Val Loss (mean: {mean_val_loss:.3f} ± {std_val_loss:.3f})')
    plt.title('Loss Metric per epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    mean_acc, std_acc = calculate_mean_std(history['accuracy'])
    mean_val_acc, std_val_acc = calculate_mean_std(history['val_accuracy'])
    plt.subplot(4, 2, 2)
    plt.plot(history['accuracy'], label=f'Accuracy (mean: {mean_acc:.3f} ± {std_acc:.3f})')
    plt.plot(history['val_accuracy'], label=f'Val Accuracy (mean: {mean_val_acc:.3f} ± {std_val_acc:.3f})')
    plt.title('Accuracy Metric per epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision
    mean_prec, std_prec = calculate_mean_std(history['precision'])
    mean_val_prec, std_val_prec = calculate_mean_std(history['val_precision'])
    plt.subplot(4, 2, 3)
    plt.plot(history['precision'], label=f'Precision (mean: {mean_prec:.3f} ± {std_prec:.3f})')
    plt.plot(history['val_precision'], label=f'Val Precision (mean: {mean_val_prec:.3f} ± {std_val_prec:.3f})')
    plt.title('Precision Metric per epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()
    return fig


# function to add a model name, description, performance to report.csv
def add_to_report(model_name, metrics, description, report, file_path):
    metrics_calculated = {}
    for key, values in metrics.items():
        mean, std = calculate_mean_std(values)
        metrics_calculated[f'{key} Mean'] = mean
        metrics_calculated[f'{key} Std'] = std

    new_row = {
        'Model': model_name,
        'Description': description,
        **metrics_calculated
    }
    new_row_df = pd.DataFrame([new_row])
    report = pd.concat([report, new_row_df], ignore_index=True)
    report.to_csv(file_path, index=False)


def bar_plot_metric_perfomances(df_metric):

    metrics = ['accuracy', 'loss', 'precision']
    dropdown_buttons = []


    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
        x=df_metric['Model'],
        y=df_metric[f'{metric} Mean'].round(3),
        error_y=dict(type='data', array=df_metric[f'{metric} Std'].round(3), visible=True),
        name=f'Train {metric.capitalize()}',
        marker_color='green',
        visible=(i == 0) 
        ))
        fig.add_trace(go.Bar(
        x=df_metric['Model'],
        y=df_metric[f'val_{metric} Mean'].round(3),
        error_y=dict(type='data', array=df_metric[f'val_{metric} Std'].round(3), visible=True),
        name=f'Validation {metric.capitalize()}',
        marker_color='coral',
        visible=(i == 0) 
        ))

        dropdown_buttons.append({
            'label': metric.capitalize(),
            'method': 'update',
            'args': [
                {'visible': [False] * len(metrics) * 2},  
                {'title': f'{metric.capitalize()} Comparison'}
            ]
        })
        dropdown_buttons[-1]['args'][0]['visible'][i * 2] = True   
        dropdown_buttons[-1]['args'][0]['visible'][i * 2 + 1] = True  

    fig.update_layout(
    title='Model Comparison: Accuracy, Loss, Precision',
    xaxis_title='Model',
    yaxis_title='Metric Value',
    barmode='group', 
    updatemenus=[{
        'buttons': dropdown_buttons,
        'direction': 'down',
        'showactive': True,
    }])

    return fig


def evaluate_model_and_save_results(model, model_name, X_test, y_test_binary, BATCH_SIZE, SEED, results_file='Results/test_metrics.csv'):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        X_test,
        y_test_binary,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False  # Keep shuffle False for evaluation
    )
    
    evaluation = model.evaluate(test_generator)
    test_loss = evaluation[0]
    test_accuracy = evaluation[1] * 100
    test_precision = evaluation[2] * 100
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test Accuracy: {test_accuracy:.3f}%')
    print(f'Test Precision: {test_precision:.3f}%')
    
    results_dir = os.path.dirname(results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
    else:
        df = pd.DataFrame(columns=['Model', 'Test Loss', 'Test Accuracy', 'Test Precision'])
    new_row = {
        'Model': model_name,
        'Test Loss': round(test_loss, 3),
        'Test Accuracy': round(test_accuracy, 3),
        'Test Precision': round(test_precision, 3)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_file, index=False)
    
    y_pred_prob = model.predict(test_generator)
    y_pred = tf.where(y_pred_prob <= 0.5, 0, 1).numpy()  # Convert probabilities to binary labels (sigmoid activation fx for last Dense layer)
    
    cm = confusion_matrix(test_generator.y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(xticks_rotation='horizontal', ax=ax, cmap=plt.cm.Blues)
    plt.show()
    
    print(classification_report(test_generator.y, y_pred))


def calculate_mean_std(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std