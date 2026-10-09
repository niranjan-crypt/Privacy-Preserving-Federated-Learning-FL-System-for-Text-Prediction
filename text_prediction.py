import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import logging
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles loading and preprocessing text corpus using a SHARED, pre-fitted
    tokenizer to ensure consistency across all federated clients.
    """
    def __init__(self, text_corpus: str, 
                 shared_tokenizer: Tokenizer, 
                 shared_total_words: int, 
                 shared_max_seq_len: int):
        """
        Initializes the handler with the raw text and SHARED parameters.
        
        Args:
            text_corpus (str): This client's local text corpus.
            shared_tokenizer (Tokenizer): The pre-fitted global tokenizer.
            shared_total_words (int): The global vocabulary size.
            shared_max_seq_len (int): The global max sequence length for padding.
        """
        self.text_corpus = text_corpus
        self.tokenizer = shared_tokenizer      # Use shared tokenizer
        self.total_words = shared_total_words  # Use shared vocab size
        self.max_sequence_len = shared_max_seq_len # Use shared max length
        self.input_sequences = []
        self.X = None
        self.y = None

    def preprocess(self):
        """
        Runs the preprocessing pipeline using the shared tokenizer:
        1. Creates n-gram sequences from local text.
        2. Pads sequences to the shared max length.
        3. Splits into features (X) and labels (y).
        4. One-hot encodes labels to the shared vocabulary size.
        """
        logger.info("Starting data preprocessing (using shared tokenizer)...")
        
        # --- DO NOT FIT TOKENIZER ---
        # The tokenizer is already fitted globally.
        
        # Create n-gram sequences using the shared tokenizer
        for line in self.text_corpus.lower().split('\n'):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                self.input_sequences.append(n_gram_sequence)
        
        if not self.input_sequences:
            logger.warning("No n-gram sequences created. Corpus may be too small.")
            # Create empty arrays to avoid errors during training
            self.X = np.array([])
            self.y = np.array([])
            return
            
        # Pad sequences using the shared max length
        padded_sequences = np.array(pad_sequences(self.input_sequences, 
                                                  maxlen=self.max_sequence_len, 
                                                  padding='pre'))
        
        # Separate input (X) and target (y)
        # X is all but the last word, y is the last word
        self.X, labels = padded_sequences[:, :-1], padded_sequences[:, -1]
        
        # One-hot encode the target labels using the shared vocab size
        self.y = to_categorical(labels, num_classes=self.total_words)
        
        logger.info("Data preprocessing complete.")
        logger.info(f"  Using shared vocabulary size: {self.total_words}")
        logger.info(f"  Using shared max sequence length: {self.max_sequence_len}")
        logger.info(f"  Shape of X: {self.X.shape}")
        logger.info(f"  Shape of y: {self.y.shape}")

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the processed features (X) and labels (y)."""
        return self.X, self.y

    def get_params(self) -> tuple[int, int, Tokenizer]:
        """Returns the shared parameters."""
        # These are now just returning the params that were passed in
        return self.total_words, self.max_sequence_len, self.tokenizer


class LSTMModel:
    """
    Handles the creation and compilation of the LSTM model.
    (Keras model state is managed by the Flower client)
    """
    def __init__(self, embedding_dim=100, lstm_units_1=150, lstm_units_2=100, dropout_rate=0.3):
        """
        Initializes the model hyperparameters.
        """
        self.embedding_dim = embedding_dim
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.model = None
        self.total_words = 0
        self.max_sequence_len_input = 0 # (max_sequence_len - 1)

    def build_model(self, total_words, max_sequence_len):
        """
        Creates and compiles the Sequential Keras model.
        
        Args:
            total_words (int): The size of the vocabulary.
            max_sequence_len (int): The (padded) length of full n-gram sequences.
        """
        logger.info("Building LSTM model...")
        self.total_words = total_words
        # Input length for the model is max_sequence_len - 1 
        # (since X is all but the last word)
        self.max_sequence_len_input = max_sequence_len - 1
        
        if self.max_sequence_len_input <= 0:
            logger.error(f"Cannot build model: max_sequence_len ({max_sequence_len}) is too small.")
            return

        self.model = Sequential()
        
        # Embedding Layer
        self.model.add(Embedding(self.total_words, 
                                 self.embedding_dim, 
                                 input_length=self.max_sequence_len_input))
        
        # LSTM Layer 1
        self.model.add(LSTM(self.lstm_units_1, return_sequences=True))
        self.model.add(Dropout(self.dropout_rate))
        
        # LSTM Layer 2
        self.model.add(LSTM(self.lstm_units_2))
        self.model.add(Dropout(self.dropout_rate))
        
        # Output Layer
        self.model.add(Dense(self.total_words, activation='softmax'))
        
        # Compile
        self.model.compile(loss='categorical_crossentropy', 
                           optimizer='adam', 
                           metrics=['accuracy'])
        
        logger.info("Model built and compiled successfully.")
        # self.model.summary(print_fn=logger.info) # Can be noisy

