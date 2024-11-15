# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# Model configuration
MAX_LENGTH = 10  # Maximum sentence length
MIN_COUNT = 3    # Minimum word count threshold for trimming

# Model parameters
model_config = {
    'model_name': 'cb_model',
    'attn_model': 'dot',
    'hidden_size': 500,
    'encoder_n_layers': 2,
    'decoder_n_layers': 2,
    'dropout': 0.1,
    'batch_size': 64
}

# Training parameters
training_config = {
    'clip': 50.0,
    'teacher_forcing_ratio': 1.0,
    'learning_rate': 0.0001,
    'decoder_learning_ratio': 5.0,
    'n_iteration': 4500,
    'print_every': 1,
    'save_every': 500
}

# Paths
save_dir = 'save'
corpus_name = 'movie-corpus'



