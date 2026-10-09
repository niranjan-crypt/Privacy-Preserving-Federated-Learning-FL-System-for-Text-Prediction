# ============================================================================
# IMPORTS: All modules grouped here
# ============================================================================

# --- Module 1 Imports ---
import psutil
import time
from typing import Dict, Tuple
import logging

# --- Module 2 Imports ---
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import numpy as np
import pickle
import base64

# --- Module 3 Imports (Keras & Local) ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import sys

# --- Module 4 Imports (Flower) ---
import flwr as fl
from collections import OrderedDict

# --- Main Execution Imports ---
import argparse


# ============================================================================
# MODULE 1: Device Resource Monitor
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow logging to ERROR to reduce spam
tf.get_logger().setLevel('ERROR')


class DeviceResourceMonitor:
    """Monitors device resources to ensure client can participate in training"""
    
    def __init__(self, 
                 min_cpu_percent: float = 30.0,
                 min_ram_percent: float = 20.0,
                 min_battery_percent: float = 30.0,
                 min_network_speed_mbps: float = 1.0):
        self.min_cpu = min_cpu_percent
        self.min_ram = min_ram_percent
        self.min_battery = min_battery_percent
        self.min_network = min_network_speed_mbps
    
    def check_cpu(self) -> Tuple[bool, float]:
        """Check if CPU is available"""
        cpu_available = 100 - psutil.cpu_percent(interval=1)
        is_ok = cpu_available >= self.min_cpu
        logger.info(f"CPU available: {cpu_available:.2f}% (Required: {self.min_cpu}%)")
        return is_ok, cpu_available
    
    def check_ram(self) -> Tuple[bool, float]:
        """Check if RAM is available"""
        ram = psutil.virtual_memory()
        ram_available = ram.available / ram.total * 100
        is_ok = ram_available >= self.min_ram
        logger.info(f"RAM available: {ram_available:.2f}% (Required: {self.min_ram}%)")
        return is_ok, ram_available
    
    def check_battery(self) -> Tuple[bool, float]:
        """Check battery level"""
        battery = psutil.sensors_battery()
        if battery is None:
            logger.info("No battery detected (desktop), skipping check")
            return True, 100.0
        
        battery_percent = battery.percent
        is_plugged = battery.power_plugged
        is_ok = battery_percent >= self.min_battery or is_plugged
        logger.info(f"Battery: {battery_percent}% (Plugged: {is_plugged})")
        return is_ok, battery_percent
    
    def check_network(self) -> Tuple[bool, float]:
        """Estimate network speed"""
        net_before = psutil.net_io_counters()
        time.sleep(1)
        net_after = psutil.net_io_counters()
        
        bytes_sent = net_after.bytes_sent - net_before.bytes_sent
        bytes_recv = net_after.bytes_recv - net_before.bytes_recv
        
        speed_mbps = (bytes_sent + bytes_recv) * 8 / (1024 * 1024)
        is_ok = speed_mbps >= self.min_network
        logger.info(f"Network speed: {speed_mbps:.2f} Mbps (Required: {self.min_network} Mbps)")
        return is_ok, speed_mbps
    
    def can_participate(self) -> Tuple[bool, Dict[str, float]]:
        """Check all resources and determine if device can participate"""
        cpu_ok, cpu_val = self.check_cpu()
        ram_ok, ram_val = self.check_ram()
        battery_ok, battery_val = self.check_battery()
        network_ok, network_val = self.check_network()
        
        resources = {
            'cpu': cpu_val,
            'ram': ram_val,
            'battery': battery_val,
            'network': network_val
        }
        
        can_train = all([cpu_ok, ram_ok, battery_ok, network_ok])
        
        if can_train:
            logger.info("✓ Device READY for training")
        else:
            logger.warning("✗ Device NOT READY for training")
        
        return can_train, resources


# ============================================================================
# MODULE 2: Encryption Handler
# ============================================================================

class EncryptionHandler:
    """Handles encryption and decryption of model parameters"""
    
    def __init__(self, password: str = "federated_learning_key"):
        """Initialize encryption with password-based key"""
        self.key = self._generate_key(password)
        self.cipher = Fernet(self.key)
    
    def _generate_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        salt = b'federated_learning_salt_2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_parameters(self, parameters: list) -> bytes:
        """Encrypt model parameters (list of NumPy arrays for Keras)"""
        params_bytes = pickle.dumps(parameters)
        encrypted = self.cipher.encrypt(params_bytes)
        logger.info(f"Encrypted {len(parameters)} parameter arrays")
        return encrypted
    
    def decrypt_parameters(self, encrypted_params: bytes) -> list:
        """Decrypt model parameters"""
        decrypted_bytes = self.cipher.decrypt(encrypted_params)
        parameters = pickle.loads(decrypted_bytes)
        logger.info(f"Decrypted {len(parameters)} parameter arrays")
        return parameters
    
    def encrypt_metrics(self, metrics: dict) -> bytes:
        """Encrypt metrics dictionary"""
        metrics_bytes = pickle.dumps(metrics)
        encrypted = self.cipher.encrypt(metrics_bytes)
        return encrypted
    
    def decrypt_metrics(self, encrypted_metrics: bytes) -> dict:
        """Decrypt metrics dictionary"""
        decrypted_bytes = self.cipher.decrypt(encrypted_metrics)
        metrics = pickle.loads(decrypted_bytes)
        return metrics


# ============================================================================
# MODULE 3: LSTM Model and Data Handler (from text_prediction.py)
# (MODIFIED to support a shared, pre-fitted tokenizer for FL)
# ============================================================================
from text_prediction import DataHandler
from text_prediction import LSTMModel

# ============================================================================
# MODULE 4: Flower Client and Server
# ============================================================================

class FederatedClient(fl.client.NumPyClient):
    """Flower client for Keras"""
    
    def __init__(self, model_wrapper, X_train, y_train, X_test, y_test, 
                 device_monitor, encryption_handler, client_id):
        
        self.model_wrapper = model_wrapper 
        self.model = model_wrapper.model
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device_monitor = device_monitor
        self.encryption = encryption_handler
        self.client_id = client_id
    
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        logger.info(f"Client {self.client_id}: Sending parameters")
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Train the Keras model if resources are available"""
        can_train, resources = self.device_monitor.can_participate()
        
        if not can_train:
            logger.warning(f"Client {self.client_id}: Insufficient resources, skipping training")
            flat_metrics = {
                "skipped": 1.0,
                "cpu": resources['cpu'], "ram": resources['ram'],
                "battery": resources['battery'], "network": resources['network']
            }
            return self.get_parameters(config), 0, flat_metrics
        
        # Check if client has data (from preprocessing)
        if len(self.X_train) == 0:
            logger.warning(f"Client {self.client_id}: No data to train on, skipping")
            return self.get_parameters(config), 0, {"skipped": 1.0}

        self.set_parameters(parameters)
        
        logger.info(f"Client {self.client_id}: Starting training")
        epochs = config.get("local_epochs", 50)
        
        history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=32, 
            verbose=0
        )
        
        loss = history.history["loss"][-1]
        accuracy = history.history["accuracy"][-1]
        logger.info(f"Client {self.client_id}: Training complete. Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "cpu": resources['cpu'],
            "ram": resources['ram'],
            "battery": resources['battery'],
            "network": resources['network']
        }
        
        return self.get_parameters(config), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate the Keras model"""
        self.set_parameters(parameters)
        
        # Check if client has data
        if len(self.X_test) == 0:
            logger.warning(f"Client {self.client_id}: No data to evaluate")
            return 0.0, 0, {"loss": 0.0, "accuracy": 0.0}

        loss, accuracy = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0
        )
        
        logger.info(f"Client {self.client_id}: Evaluation Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        return float(loss), len(self.X_test), {"loss": float(loss), "accuracy": float(accuracy)}


# ============================================================================
# GLOBAL SHARED PARAMETERS FOR FEDERATED LEARNING
# ============================================================================

import os

def load_text_file(filename: str) -> str:
    """Load text from a file in the current folder."""
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return ""

# Load client corpus texts from files
CLIENT_CORPUS = {
    0: load_text_file('CommEff.txt'),
    1: load_text_file('CommEff.txt'),
    2: load_text_file('Abstract.txt')  # Or another file if you want different text
}

# Create global tokenizer by combining all client corpora
def create_global_tokenizer():
    """Create a shared tokenizer from all client corpora"""
    logger.info("Creating global shared tokenizer...")
    
    # Combine all corpora
    all_text = " ".join(CLIENT_CORPUS.values())
    
    # Create and fit tokenizer
    global_tokenizer = Tokenizer()
    global_tokenizer.fit_on_texts([all_text])
    
    # Calculate vocabulary size
    global_vocab_size = len(global_tokenizer.word_index) + 1
    
    # Calculate max sequence length from all corpora
    max_len = 0
    for corpus in CLIENT_CORPUS.values():
        for line in corpus.lower().split('\n'):
            token_list = global_tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                max_len = max(max_len, len(n_gram_sequence))
    
    if max_len == 0:
        raise ValueError("Could not determine max_sequence_len. Corpus may be empty.")

    logger.info(f"Global vocabulary size: {global_vocab_size}")
    logger.info(f"Global max sequence length: {max_len}")
    
    return global_tokenizer, global_vocab_size, max_len

# Initialize global parameters
GLOBAL_TOKENIZER, GLOBAL_VOCAB_SIZE, GLOBAL_MAX_SEQ_LEN = create_global_tokenizer()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def start_client(client_id: int = 0, connect_to_server: bool = False):
    """Start a federated learning client with Keras"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Federated Learning Client {client_id}")
    logger.info(f"{'='*60}\n")
    
    # Initialize components
    device_monitor = DeviceResourceMonitor(
        min_cpu_percent=5.0,
        min_ram_percent=5.0,
        min_battery_percent=0.0,
        min_network_speed_mbps=0.0
    )
    
    encryption_handler = EncryptionHandler()
    
    # --- CRITICAL: Use shared tokenizer and parameters ---
    corpus_text = CLIENT_CORPUS.get(client_id, CLIENT_CORPUS[0])
    
    # Use the new DataHandler from Module 3
    data_handler = DataHandler(
        corpus_text,
        shared_tokenizer=GLOBAL_TOKENIZER,
        shared_total_words=GLOBAL_VOCAB_SIZE,
        shared_max_seq_len=GLOBAL_MAX_SEQ_LEN
    )
    data_handler.preprocess()
    X, y = data_handler.get_data()
    
    # Split data into train and test
    if len(X) < 10:
        logger.warning(f"Client {client_id}: Not enough data to create a train/test split. Using all for training.")
        X_train, X_test = X, np.array([])
        y_train, y_test = y, np.array([])
    else:
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        logger.info(f"Client {client_id}: Train/Test split: {len(X_train)} / {len(X_test)} samples")

    
    # --- Use global parameters for model (from Module 3) ---
    model_builder = LSTMModel(embedding_dim=50, lstm_units_1=100)
    model_builder.build_model(GLOBAL_VOCAB_SIZE, GLOBAL_MAX_SEQ_LEN)
    
    if model_builder.model is None:
        logger.error(f"Client {client_id}: Model failed to build. Exiting.")
        sys.exit()

    # --- Flower Client ---
    client = FederatedClient(
        model_wrapper=model_builder,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_monitor=device_monitor,
        encryption_handler=encryption_handler,
        client_id=client_id
    )
    
    logger.info(f"Client {client_id} (Keras) initialized and ready")
    
    if connect_to_server:
        logger.info("Connecting to server at localhost:8080...")
        fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    
    return client


def start_server(num_rounds: int = 5, start_server: bool = False):
    """Start federated learning server"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Federated Learning Server")
    logger.info(f"{'='*60}\n")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )
    
    logger.info(f"Server configured with FedAvg strategy")
    logger.info(f"Waiting for 2 clients, training for {num_rounds} rounds")
    
    if start_server:
        logger.info("Starting Flower server on localhost:8080...")
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Federated Learning with Keras/LSTM")
    parser.add_argument("--mode", type=str, choices=["server", "client", "demo"], 
                        default="demo", help="Run mode: demo, server, or client")
    parser.add_argument("--id", type=int, default=0, help="Client ID (for client mode)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    args = parser.parse_args()
    
    if args.mode == "server":
        start_server(num_rounds=args.rounds, start_server=True)
    
    elif args.mode == "client":
        start_client(client_id=args.id, connect_to_server=True)
        
    elif args.mode == "demo":
        print("--- DEMO MODE ---")
        print("This script is designed to run in 'server' or 'client' mode.")
        print("To run the demo, open three terminals:")
        print("\nTerminal 1: Run the server")
        print(f"  python {__file__} --mode server --rounds 3")
        print("\nTerminal 2: Run Client 0")
        print(f"  python {__file__} --mode client --id 0")
        print("\nTerminal 3: Run Client 1")
        print(f"  python {__file__} --mode client --id 1")