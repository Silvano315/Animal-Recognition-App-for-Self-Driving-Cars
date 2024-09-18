import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Plot a choosen metric from history file
def scatter_plot_metrics(file_name):

    with open("History_models/"+file_name, 'rb') as history_file:
        history = pickle.load(history_file)

    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    def calculate_mean_std(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

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