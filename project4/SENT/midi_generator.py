import os
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_transformer import build_transformer_model
from train_classifier import preprocess_sentence

GENERATED_DIR = './generated'
TRAIN_DIR = "./trained"


def sample_next(predictions, k, temp=2):
    predictions = predictions / temp
    # Sample using a categorical distribution over the top k midi chars
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    p_choices = tf.math.softmax(top_k_values).numpy()
    predicted_id = np.random.choice(top_k_choices, 1, p=p_choices)[0]
    return predicted_id

def process_init_text(model, init_text, char2idx):

    tokens = init_text.split()
    input_eval = [char2idx[c] for c in tokens]
    input_tensor = tf.expand_dims(input_eval, 0)
    predictions = model(input_tensor, training=False)
    last_prediction = predictions[:, -1, :]
    return input_eval, last_prediction

def generate_midi(
    model,
    char2idx,
    idx2char,
    init_text="",
    seq_len=256,
    gen_len=255,
    k=3,
):

    input_sequence_ids, last_prediction = process_init_text(
        model = model, 
        char2idx=char2idx, 
        init_text=init_text
    )

    midi_generated_ids = input_sequence_ids.copy()
    num_generate = gen_len - len(input_sequence_ids)
    
    for i in range(num_generate):

        # Truncate the input sequence - sliding window
        current_input = midi_generated_ids[-seq_len:] 
        input_tensor = tf.expand_dims(current_input, 0)
        predictions = model(input_tensor, training=False)

        # Get the logits for the *last* token in the current_input
        last_prediction = predictions[:, -1, :] 
        predicted_id = sample_next(last_prediction, int(k))
        midi_generated_ids.append(predicted_id)

        if idx2char[predicted_id] == "\n":
            break

    generated_tokens = [idx2char[i] for i in midi_generated_ids]
    return " ".join(generated_tokens)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--blocks', type=int, default=4, help="Number of decoder blocks.")
    parser.add_argument('--ffdim', type=int, default=1024, help="Feed Forward layer dimension.")
    parser.add_argument('--seqinit', type=str, default="<START>", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.") 
    parser.add_argument('--topk', type=int, default=10, help="Top k to sample from during generation.")
    parser.add_argument('--gen_len', type=int, default=500, help="The desired total number of tokens to generate.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char,idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    transformer_model = build_transformer_model(
        vocab_size=vocab_size,
        seq_length=opt.seqlen,
        embed_dim=opt.embed,
        num_heads=opt.heads,
        ff_dim=opt.ffdim,
        num_blocks=opt.blocks,
        dropout=0.1,
    )
    weights_path = os.path.join(TRAIN_DIR, "transformer_ckpt.weights.h5")
    transformer_model.load_weights(weights_path)
    midi_txt = generate_midi(
        transformer_model, 
        char2idx, 
        idx2char, 
        opt.seqinit, 
        opt.seqlen, 
        opt.gen_len,
        opt.topk
    )
    print(midi_txt)

    me.write(midi_txt, os.path.join(GENERATED_DIR, "generated.mid"))
