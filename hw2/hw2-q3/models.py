import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """

        batch_size, max_src_len, hidden_size = encoder_outputs.size()

        # Transform query and encoder outputs
        query_transformed = self.Ws(query)  # (batch_size, max_tgt_len, hidden_size)
        encoder_transformed = self.Wh(encoder_outputs)  # (batch_size, max_src_len, hidden_size)

        # Expand and repeat query for alignment
        query_expanded = query_transformed.unsqueeze(2).repeat(1, 1, max_src_len, 1)  # (batch_size, max_tgt_len, max_src_len, hidden_size)
        encoder_expanded = encoder_transformed.unsqueeze(1)  # (batch_size, 1, max_src_len, hidden_size)

        # Compute scores
        scores = self.v(torch.tanh(query_expanded + encoder_expanded)).squeeze(-1)  # (batch_size, max_tgt_len, max_src_len)

        # Mask padding positions in the source sequence
        mask = (torch.arange(max_src_len).unsqueeze(0).to(src_lengths.device) < src_lengths.unsqueeze(1))
        mask = mask.unsqueeze(1)  # (batch_size, 1, max_src_len)
        scores = scores.masked_fill(~mask, float('-inf'))  # Apply mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, max_tgt_len, max_src_len)

        # Compute context vector as a weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, max_tgt_len, hidden_size)

        # Concatenate context and query
        concat = torch.cat((context, query), dim=2)  # (batch_size, max_tgt_len, hidden_size * 2)

        # Pass through a linear layer to get the attention-enhanced state
        attn_out = torch.tanh(self.out(concat))  # (batch_size, max_tgt_len, hidden_size)

        return attn_out, attn_weights

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        
        # Embedding the source sequence
        embedded = self.dropout(self.embedding(src))

        # Packing the padded sequences
        packed_input = pack(embedded, lengths, batch_first=True, enforce_sorted=False)

        # Passing through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpacking the sequences
        encoder_outputs, _ = unpack(packed_output, batch_first=True)

        return encoder_outputs, (hidden, cell)
        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################

        # a)
        # embedded = self.dropout(self.embedding(tgt))
        # outputs = []
        # dec_hidden = dec_state
        # # Decoder LSTM step-wise

        # for t in range(tgt.size(1)):
            
        #     input_t = embedded[:, t, :].unsqueeze(1)  # Current timestep input
        #     output, dec_hidden = self.lstm(input_t, dec_hidden)
        #     output = self.dropout(output)

        #     if self.attn is not None:
        #         output = self.attn(output, encoder_outputs, src_lengths)
        #     outputs.append(output)
            
        # outputs = torch.cat(outputs, dim=1)
        # return outputs, dec_hidden

        # b)
        embedded = self.dropout(self.embedding(tgt))  # (batch_size, max_tgt_len, hidden_size)
        outputs = []
        dec_hidden = dec_state

        for t in range(tgt.size(1)):  # Loop over target sequence length
            input_t = embedded[:, t, :].unsqueeze(1)  # (batch_size, 1, hidden_size)
            output, dec_hidden = self.lstm(input_t, dec_hidden)  # LSTM step
            # output = output.squeeze(1)  # (batch_size, hidden_size)
            if self.attn is not None:
                output, _ = self.attn(output, encoder_outputs, src_lengths)  # Attention mechanism
            output = self.dropout(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_tgt_len, hidden_size)

        return outputs, dec_hidden

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
