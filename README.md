# Privacy-Preserving Federated Learning System for Text Prediction

A simulated Federated Learning (FL) system that trains an LSTM-based next-word prediction model across multiple clients without centralizing raw text data — built with the [Flower](https://flower.ai/) framework and the FedAvg algorithm.

## Overview

This project demonstrates how deep learning models can be trained collaboratively across distributed clients while keeping each client's raw data local. Instead of sending training data to a central server, each client trains locally and only shares model weight updates, which are aggregated using Federated Averaging (FedAvg).

On top of the core FL pipeline, the system adds two practical layers often needed in real-world federated deployments:
- **Device-eligibility checks** — simulating resource-constrained environments (e.g., mobile devices)
- **Encrypted parameter transmission** — adding a security layer over model updates

## Features

- **Federated Learning with Flower**: Implements a `NumPyClient` and `FedAvg` server strategy to coordinate training across multiple clients.
- **LSTM-based Next-Word Prediction**: A 2-layer LSTM (Embedding → LSTM → Dropout → LSTM → Dropout → Dense/Softmax) trained on each client's local text corpus.
- **Shared Global Tokenizer**: A custom `DataHandler` pipeline uses a globally-consistent tokenizer and sequence length so that model weights remain compatible for aggregation across heterogeneous (non-IID) client datasets.
- **Device-Eligibility Simulation**: A `DeviceResourceMonitor` module checks CPU, RAM, battery, and network conditions before a client participates in a training round — modeling real-world constraints of on-device federated training.
- **Encrypted Communication**: Model parameters and metrics are encrypted using Fernet symmetric encryption with a PBKDF2-derived key before transmission.
- **Modular Architecture**: The system is organized into four independent modules — resource monitoring, encryption, ML model/data handling, and Flower client-server orchestration.

## Architecture

```
┌─────────────────────┐
│   Flower Server      │
│   (FedAvg Strategy)  │
└──────────┬───────────┘
           │  aggregates encrypted weight updates
   ┌───────┴────────┬────────────────┐
   │                │                │
┌──▼───┐        ┌───▼──┐        ┌────▼─┐
│Client │        │Client │        │Client │
│  0    │        │  1    │        │  2    │
└───────┘        └───────┘        └───────┘

Each client runs:
1. DeviceResourceMonitor  → checks CPU/RAM/battery/network
2. DataHandler            → preprocesses local text (shared tokenizer)
3. LSTMModel              → local training
4. EncryptionHandler      → encrypts weights/metrics before sending
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Federated Learning | Flower (`flwr`) |
| Security | `cryptography` (Fernet, PBKDF2) |
| System Monitoring | `psutil` |

## Project Structure

```
├── simulation.py         # Main FL system: resource monitor, encryption, Flower client/server
├── text_prediction.py    # DataHandler and LSTMModel classes
├── CommEff.txt           # Sample text corpus (client 0 & 1)
├── Abstract.txt          # Sample text corpus (client 2)
├── requirements.txt      # Project dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Ensure `simulation.py`, `text_prediction.py`, `CommEff.txt`, and `Abstract.txt` are all in the same directory.

## Usage

This simulation requires at least **2 clients** and **1 server**, each run in a separate terminal.

**Terminal 1 — Start the server:**
```bash
python simulation.py --mode server --rounds 3
```

**Terminal 2 — Start client 0:**
```bash
python simulation.py --mode client --id 0
```

**Terminal 3 — Start client 1:**
```bash
python simulation.py --mode client --id 1
```

> Start the server first, then launch the clients within a few seconds — the server waits for `min_available_clients=2` before beginning training rounds.

Optional: add a third client (uses `Abstract.txt` as its corpus):
```bash
python simulation.py --mode client --id 2
```

Running with no arguments defaults to `--mode demo`, which only prints these instructions.

### CLI Arguments

| Flag | Description | Default |
|---|---|---|
| `--mode` | `server`, `client`, or `demo` | `demo` |
| `--id` | Client ID (client mode only) | `0` |
| `--rounds` | Number of FL training rounds (server mode only) | `5` |

## How It Works

1. **Global tokenizer initialization**: All client corpora are combined once at startup to build a shared vocabulary and max sequence length, ensuring every client's model has matching input/output dimensions for weight aggregation.
2. **Local preprocessing**: Each client converts its local text into n-gram sequences, pads them, and one-hot encodes the target word — without ever sharing the raw text itself.
3. **Resource check**: Before training, each client verifies it has sufficient CPU, RAM, battery, and network resources; if not, it skips the round.
4. **Local training**: Each client trains its LSTM locally on its own data for a configurable number of epochs.
5. **Encrypted weight sharing**: Updated weights and metrics are encrypted before being sent to the server.
6. **Federated averaging**: The server aggregates weight updates from all participating clients using FedAvg and redistributes the updated global model.

## Design Notes & Limitations

- The global tokenizer is built by combining all client corpora upfront for simplicity — in a stricter FL setting, this centralization step would itself be avoided or handled via a pre-agreed vocabulary.
- This is a simulation intended for learning and demonstration purposes (single machine, `localhost` communication) rather than a production-grade deployment.

## References

- McMahan, H. B., et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg paper)
- Survey paper on Federated Learning systems and applications in healthcare (used for background/motivation)

## Author

- **Niranjan Krishnakumar** - [@niranjan-crypt](https://github.com/niranjan-crypt)
