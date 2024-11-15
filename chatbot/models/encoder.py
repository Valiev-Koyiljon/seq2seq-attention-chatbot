import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        """
        Initialize the EncoderRNN.

        Args:
            hidden_size (int): Size of the RNN's hidden states.
            embedding (nn.Embedding): Embedding layer for input tokens.
            n_layers (int, optional): Number of GRU layers (default: 1).
            dropout (float, optional): Dropout probability (default: 0 for 1 layer).
        """
        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Bidirectional GRU, with dropout applied if multiple layers are used
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        Forward pass for EncoderRNN.

        Args:
            input_seq (torch.Tensor): Input sequence of shape (max_length, batch_size).
            input_lengths (torch.Tensor): Lengths of sequences in the batch.
            hidden (torch.Tensor, optional): Initial hidden state of the GRU.

        Returns:
            torch.Tensor: Encoder outputs (max_length, batch_size, hidden_size).
            torch.Tensor: Final hidden state of the GRU (n_layers*2, batch_size, hidden_size).
        """
        # Embed input sequence
        embedded = self.embedding(input_seq)

        # Pack the sequence for efficient processing
        packed = pack_padded_sequence(embedded, input_lengths)

        # Pass through the GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack the padded sequence
        outputs, _ = pad_packed_sequence(outputs)

        # Combine bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden