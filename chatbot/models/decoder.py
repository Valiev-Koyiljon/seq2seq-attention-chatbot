import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attn

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        """
        Initialize LuongAttnDecoderRNN.

        Args:
            attn_model (str): Type of attention ('dot', 'general', 'concat')
            embedding (nn.Embedding): Shared embedding layer
            hidden_size (int): Size of hidden state
            output_size (int): Size of output vocabulary
            n_layers (int): Number of GRU layers
            dropout (float): Dropout probability
        """
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        Forward step of the decoder.

        Args:
            input_step (torch.LongTensor): One time step input
            last_hidden (torch.Tensor): Last hidden state of the decoder
            encoder_outputs (torch.Tensor): Outputs from the encoder

        Returns:
            torch.Tensor: Output probabilities for each word
            torch.Tensor: Hidden state for this time step
        """
        # Get embedding of current input word
        embedded = self.embedding_dropout(self.embedding(input_step))

        # Forward through unidirectional GRU
        gru_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights
        attn_weights = self.attn(gru_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output
        gru_output = gru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden