import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


def match_seq_len(q_seqs, r_seqs, o_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []
    proc_o_seqs = []

    for q_seq, r_seq, o_seq in zip(q_seqs, r_seqs, o_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_o_seqs.append(o_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 -
                             len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 -
                             len(q_seq)))
                ]
            )
        )
        proc_o_seqs.append(
            np.concatenate(
                [
                    o_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 -
                             len(q_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs, proc_o_seqs


def collate_fn(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    o_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    oshft_seqs = []

    for q_seq, r_seq, o_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        o_seqs.append(FloatTensor(o_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        oshft_seqs.append(FloatTensor(o_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    o_seqs = pad_sequence(
        o_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )
    oshft_seqs = pad_sequence(
        oshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, o_seqs, qshft_seqs, rshft_seqs, oshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, o_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, oshft_seqs * mask_seqs

    '''
    mask_seqs example
    (1)
    q_seqs = [1,2,3,4]
    qshft_seqs = [2,3,4,5]
    -->
    q_seqs = [1,2,3,4]
    qshft_seqs = [2,3,4,5]

    (2)
    q_seqs = [1,2,3,4]
    qshft_seqs = [2,3,4,-1]
    -->
    q_seqs = [1,2,3,0]
    qshft_seqs = [2,3,4,0]

    '''

    return q_seqs, r_seqs, o_seqs, qshft_seqs, rshft_seqs, oshft_seqs, mask_seqs
