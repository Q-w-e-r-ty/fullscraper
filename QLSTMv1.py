import torch
from torch import nn
from torch.utils.data import Dataset
import pennylane as qml


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


# Classical LSTM
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device  # Ensure hidden states are on the same device as input x
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        # Use the output from the first (or only) layer
        out = self.linear(hn[0]).flatten()
        return out


# Quantum LSTM
class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        n_vrotations=3,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        backend="default.qubit",
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # Define the ansatz used for the quantum circuit.
        def ansatz(params, wires_type):
            # Entangling layer.
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(
                            wires=[wires_type[j], wires_type[j + i - self.n_qubits]]
                        )

            # Variational layer.
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features, weights, wires_type):
            # Encode input features into the quantum circuit.
            qml.templates.AngleEmbedding(features, wires=wires_type)
            # Prepare additional rotation parameters based on input features.
            ry_params = [torch.arctan(feature) for feature in features][0]
            rz_params = [torch.arctan(feature**2) for feature in features][0]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(ry_params[i], wires=wires_type[i])

            # Apply variational ansatz.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def _circuit_forget(inputs, weights):
            VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        self.qlayer_forget = qml.QNode(
            _circuit_forget, self.dev_forget, interface="torch"
        )

        def _circuit_input(inputs, weights):
            VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        self.qlayer_update = qml.QNode(
            _circuit_update, self.dev_update, interface="torch"
        )

        def _circuit_output(inputs, weights):
            VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_output = qml.QNode(
            _circuit_output, self.dev_output, interface="torch"
        )

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(
            f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})"
        )

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = {
            "forget": qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            "input": qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            "update": qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            "output": qml.qnn.TorchLayer(self.qlayer_output, weight_shapes),
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        Uses sigmoid for recurrent_activation and tanh for activation.
        """
        if self.batch_first:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        device = x.device  # Make sure to create all new tensors on the same device
        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_states
            # Assuming init_states are provided in a similar shape but ensure they're on the proper device.
            h_t = h_t[0].to(device)
            c_t = c_t[0].to(device)

        for t in range(seq_length):
            # Get features from the t-th element in sequence for all entries in the batch.
            x_t = x[:, t, :]

            # Concatenate previous hidden state and current input.
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)

            # Quantum layers acting as gates.
            f_t = torch.sigmoid(self.clayer_out(self.VQC["forget"](y_t)))  # Forget gate
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t)))   # Input gate
            g_t = torch.tanh(self.clayer_out(self.VQC["update"](y_t)))       # Update gate
            o_t = torch.sigmoid(self.clayer_out(self.VQC["output"](y_t)))    # Output gate

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class QShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=0, n_qlayers=1):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = QLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device  # Create hidden states on the same device as input
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # Use the first layer's hidden state
        return out
