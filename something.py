# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
import pandas as pd

import time
import numpy as np
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import nn

from datetime import datetime

# Replace this to use Noisy QLSTM
# from QLSTM_Noisy import SequenceDataset
from QLSTMv1 import SequenceDataset  # <-- Your existing dataset class

import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

# %%
df = pd.read_csv('infosys stock full combined.csv')

# %%
columns = [
    'Open Price', 
    'High Price', 
    'Low Price', 
    'Close Price',
    'feature1','feature2','feature3','feature4','feature5'
]
# If you want to incorporate the multi-modal features, e.g., "Encoded Text" or "Sentiment Score",
# you could add them to 'columns' or handle them separately. 
# columns += ['Sentiment Score']  # Example only if you want to treat them as numeric features.

# %%
data = df.filter(columns)
dataset = data.values

# %%
# Splitting the data into train and test
size = int(len(df) * 0.7)
df_train = dataset[:size].copy()
df_test = dataset[size:].copy()

# %%
# Select the features
df_train = pd.DataFrame(df_train, columns=columns)
df_test = pd.DataFrame(df_test, columns=columns)

features = df_train.columns
target = 'Close Price'

# %%
def normalize(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

# %%
# Normalizing the data
df_train, min_train, max_train = normalize(df_train)
df_test, _, _ = normalize(df_test, min_train, max_train)

# %%
torch.manual_seed(101)

batch_size = 1
sequence_length = 3

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

X, y = next(iter(train_loader))
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# %%
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output.cpu()

# %% [ORIGINAL QShallowRegressionLSTM IMPORT & USAGE]
from QLSTMv1 import QShallowRegressionLSTM

learning_rate = 0.01
num_hidden_units = 16

Qmodel = QShallowRegressionLSTM(
    num_sensors=len(features),
    hidden_units=num_hidden_units,
    n_qubits=7,
    n_qlayers=1
).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(Qmodel.parameters(), lr=learning_rate)

# %%
# Count number of parameters
num_params = sum(p.numel() for p in Qmodel.parameters() if p.requires_grad)
print(f"Number of parameters (Original QLSTM): {num_params}")

# %%
quantum_loss_train = []
quantum_loss_test = []
num_epochs = 50

for ix_epoch in range(num_epochs):
    print(f"Epoch {ix_epoch}\n---------")
    start = time.time()
    train_loss = train_model(train_loader, Qmodel, loss_function, optimizer=optimizer)
    test_loss = test_model(test_loader, Qmodel, loss_function)
    end = time.time()
    print("Execution time", end - start)
    quantum_loss_train.append(train_loss)
    quantum_loss_test.append(test_loss)

# %%
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

ystar_col_Q = "Model Forecast"
df_train[ystar_col_Q] = predict(train_eval_loader, Qmodel).cpu().numpy()
df_test[ystar_col_Q] = predict(test_eval_loader, Qmodel).cpu().numpy()

# %%
plt.figure(figsize=(12, 7))
plt.plot(range(len(df_train)), df_train["Close Price"], label = "Real Data")
plt.plot(range(len(df_train)), df_train["Model Forecast"], label = "QLSTM Train Prediction")
plt.ylabel('Stock Price')
plt.xlabel('Days')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(12, 7))
plt.plot(range(len(df_test)), df_test["Close Price"], label = "Real Data")
plt.plot(range(len(df_test)), df_test["Model Forecast"], label = "QLSTM Test Prediction")
plt.ylabel('Stock Price')
plt.xlabel('Days')
plt.legend()
plt.show()

# %%
# Calculate the RMSE for the train and test data
from sklearn.metrics import mean_squared_error

train_rmse = math.sqrt(mean_squared_error(df_train["Close Price"], df_train["Model Forecast"]))
test_rmse = math.sqrt(mean_squared_error(df_test["Close Price"], df_test["Model Forecast"]))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# %%
# Calculate the accuracy of the model
def accuracy(y, y_star):
    return np.mean(np.abs(y - y_star) < 0.1)

train_accuracy = accuracy(df_train["Close Price"], df_train["Model Forecast"])
test_accuracy = accuracy(df_test["Close Price"], df_test["Model Forecast"])
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# %%
# Save the trained model
torch.save(Qmodel.state_dict(), "QLSTM_Stock_Price_Model.pt")

# %% 
################################################################################
#                        NEW ENHANCEMENTS / MODIFICATIONS                      #
################################################################################

"""
Below we define a new model class: `EnhancedQLSTMModel`.

Key changes:
1) A classical LSTM layer is stacked in front of the QLSTM (hybrid approach).
2) We add a simple self-attention layer after the QLSTM outputs.
3) We include a skip connection around the QLSTM to help gradient flow.
4) Demonstrate how you could (optionally) handle multi-modal inputs 
   (like 'Encoded Text' or 'Sentiment Score') by splitting them into 
   a separate branch. This is just an example; adapt to your actual data usage.
"""

from QLSTMv1 import QLSTM  # Import the QLSTM class directly if needed

class EnhancedQLSTMModel(nn.Module):
    def __init__(
        self,
        num_sensors,
        hidden_units,
        n_qubits=4,
        n_qlayers=1,
        multi_modal=False,
        text_embedding_dim=4,
        sentiment_dim=3
    ):
        super(EnhancedQLSTMModel, self).__init__()
        
        self.multi_modal = multi_modal
        self.hidden_units = hidden_units
        
        # If multi_modal is True, we define separate heads for text and sentiment
        if self.multi_modal:
            # Example: you might have 'Encoded Text' as a 4D vector
            self.text_branch = nn.Sequential(
                nn.Linear(text_embedding_dim, 16),
                nn.ReLU(),
                nn.Linear(16, hidden_units),
            )
            # Example: you might have 'Sentiment Score' as a 3D vector
            self.sentiment_branch = nn.Sequential(
                nn.Linear(sentiment_dim, 16),
                nn.ReLU(),
                nn.Linear(16, hidden_units),
            )
        
        # A classical LSTM to extract initial temporal features from the main numeric features
        self.classical_lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=1,
            batch_first=True
        )
        
        # QLSTM for the second stage
        self.qlstm = QLSTM(
            input_size=hidden_units,
            hidden_size=hidden_units,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            batch_first=True
        )
        
        # A simple self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=2,
            batch_first=True
        )
        
        # Final linear layer for regression
        self.output_layer = nn.Linear(hidden_units, 1)
        
    def forward(self, x, text_input=None, sentiment_input=None):
        """
        Args:
          x: shape [batch_size, seq_length, num_sensors]
          text_input: optional, shape [batch_size, text_embedding_dim]
          sentiment_input: optional, shape [batch_size, sentiment_dim]
        """
        # If multi_modal, process the additional inputs
        # and fuse them as needed.
        if self.multi_modal and text_input is not None and sentiment_input is not None:
            text_feat = self.text_branch(text_input)  # [batch_size, hidden_units]
            sent_feat = self.sentiment_branch(sentiment_input)  # [batch_size, hidden_units]
            # Expand dims to match time-series shape or incorporate them differently
            # E.g., you could broadcast or simply add them to each timestep.
            # For simplicity, let's just add them as a bias to x's first time step:
            # x[:, 0, :hidden_units] += text_feat + sent_feat
            # Alternatively, you might replicate them across all timesteps.
        
        # 1) Pass through classical LSTM
        lstm_out, _ = self.classical_lstm(x)  # [batch_size, seq_length, hidden_units]
        
        # 2) Skip connection: we'll store the output of classical LSTM to add after QLSTM
        skip_connection = lstm_out.clone()
        
        # 3) Pass through QLSTM
        qlstm_out, _ = self.qlstm(lstm_out)   # [batch_size, seq_length, hidden_units]
        
        # 4) Simple skip connection: Add classical LSTM output to QLSTM output
        combined_out = qlstm_out + skip_connection
        
        # 5) Apply attention. 
        #    Note: For multihead attention, the shape is (batch, seq, embed_dim).
        attn_out, _ = self.attention(combined_out, combined_out, combined_out)
        
        # 6) We use the last time step for regression
        last_step = attn_out[:, -1, :]  # shape [batch_size, hidden_units]
        
        # 7) Final regression output
        out = self.output_layer(last_step).flatten()  # [batch_size]
        return out

# %% 
"""
Below is an example usage of the new EnhancedQLSTMModel.
We will create an instance and train it similarly to your existing QShallowRegressionLSTM.
Comment out the original Qmodel training if you want to avoid double runs.
"""

enhanced_model = EnhancedQLSTMModel(
    num_sensors=len(features),
    hidden_units=num_hidden_units,
    n_qubits=7,
    n_qlayers=1,
    multi_modal=False  # set to True if you plan to use 'Encoded Text' or 'Sentiment Score' 
).to(device)

loss_function_enh = nn.MSELoss()
optimizer_enh = torch.optim.Adam(enhanced_model.parameters(), lr=learning_rate)

# Count parameters in the enhanced model
num_params_enh = sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad)
print(f"Number of parameters (Enhanced QLSTM): {num_params_enh}")

enh_train_loss = []
enh_test_loss = []
num_epochs_enh = 50

for ix_epoch in range(num_epochs_enh):
    print(f"[Enhanced] Epoch {ix_epoch}\n---------")
    start = time.time()
    t_loss = train_model(train_loader, enhanced_model, loss_function_enh, optimizer_enh)
    v_loss = test_model(test_loader, enhanced_model, loss_function_enh)
    end = time.time()
    print("Execution time", end - start)
    enh_train_loss.append(t_loss)
    enh_test_loss.append(v_loss)

# Evaluate predictions
df_train["Enhanced Forecast"] = predict(train_eval_loader, enhanced_model).cpu().numpy()
df_test["Enhanced Forecast"] = predict(test_eval_loader, enhanced_model).cpu().numpy()

plt.figure(figsize=(12, 7))
plt.plot(range(len(df_train)), df_train["Close Price"], label="Real Data")
plt.plot(range(len(df_train)), df_train["Enhanced Forecast"], label="Enhanced Train Prediction")
plt.ylabel("Stock Price")
plt.xlabel("Days")
plt.legend()
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(range(len(df_test)), df_test["Close Price"], label="Real Data")
plt.plot(range(len(df_test)), df_test["Enhanced Forecast"], label="Enhanced Test Prediction")
plt.ylabel("Stock Price")
plt.xlabel("Days")
plt.legend()
plt.show()

enh_train_rmse = math.sqrt(mean_squared_error(df_train["Close Price"], df_train["Enhanced Forecast"]))
enh_test_rmse = math.sqrt(mean_squared_error(df_test["Close Price"], df_test["Enhanced Forecast"]))
print(f"Enhanced Model Train RMSE: {enh_train_rmse}")
print(f"Enhanced Model Test RMSE: {enh_test_rmse}")

enh_train_accuracy = accuracy(df_train["Close Price"], df_train["Enhanced Forecast"])
enh_test_accuracy = accuracy(df_test["Close Price"], df_test["Enhanced Forecast"])
print(f"Enhanced Model Train accuracy: {enh_train_accuracy}")
print(f"Enhanced Model Test accuracy: {enh_test_accuracy}")

torch.save(enhanced_model.state_dict(), "Enhanced_QLSTM_Stock_Price_Model.pt")

# %%
