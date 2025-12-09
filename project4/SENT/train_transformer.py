import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

import midi_encoder as me

# Directory where the checkpoints will be saved
TRAIN_DIR = "./trained"

class PositionalEncoding(layers.Layer):
    """
    Adds positional information to the token embeddings.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)

    def call(self, inputs):
        # position indices: 0, 1, 2, ..., sequence_length-1
        sequence_length = tf.shape(inputs)[-1]
        max_sequence_length = self.position_embeddings.input_dim
        positions = tf.range(start=0, limit=max_sequence_length, delta=1)
        current_positions = positions[:sequence_length]

        # Get embeddings
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(current_positions)
        
        # Add them together
        return embedded_tokens + embedded_positions

class TransformerDecoderBlock(layers.Layer):
    """
    Transformer Decoder.
    """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):

        # 1. Masked Multi-Head Attention (Causal Self-Attention)
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        
        # Create a look-ahead mask (upper triangular matrix)
        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((sequence_length, sequence_length)),
            -1,
            0,
        )
        padding_mask = tf.cast(
            tf.equal(inputs, 0),
            dtype=tf.float32,
        )[:, tf.newaxis, :]

        combined_mask = tf.maximum(look_ahead_mask, padding_mask)

        # Apply attention
        attn_output = self.att(
            inputs,
            inputs,
            attention_mask=combined_mask,
        )
        attn_output = self.dropout1(attn_output, training=training)
        
        # Add & Norm (Residual connection)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 2. Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Add & Norm
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(
        vocab_size,
        seq_length,
        embed_dim,
        num_heads,
        ff_dim,
        num_blocks,
        batch_size,
        dropout=0.1,
):
    """
    Builds a Decoder-only Transformer model for generative MIDI sequence prediction.
    """
    # The input layer must *not* be batch_shape-specified for flexible inference later.
    inputs = layers.Input(shape=(seq_length,), batch_size=batch_size, dtype=tf.int32)
    
    # 1. Positional Encoding
    x = PositionalEncoding(seq_length, vocab_size, embed_dim)(inputs)
    x = layers.Dropout(dropout)(x)

    # 2. Stacked Transformer Decoder Blocks
    for _ in range(num_blocks):
        x = TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout_rate=dropout)(x)

    # 3. Final Prediction
    # We could use transpose of embedding weights for efficiency
    outputs = layers.Dense(vocab_size)(x)
    
    # Use Model class for custom layers
    model = Model(inputs=inputs, outputs=outputs)
    return model

def generative_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_generative_model(model, train_dataset, test_dataset, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=generative_loss)
    checkpoint_prefix = os.path.join(TRAIN_DIR, f"transformer_ckpt.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    return model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[checkpoint_callback],
    )

def build_dataset(text, char2idx, seq_length, batch_size, buffer_size=10000):

    tokens = text.split(" ")
    filtered_tokens = [c for c in tokens if c]
    text_as_int = np.array([char2idx[c] for c in filtered_tokens])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(__split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset

def build_char2idx(train_vocab, test_vocab):
    # Merge train and test vocabulary
    vocab = list(train_vocab | test_vocab)
    vocab.sort()

    # Calculate vocab size
    vocab_size = len(vocab)

    # Create dict to support char to index conversion
    char2idx = { char:i for i,char in enumerate(vocab) }

    # Save char2idx encoding as a json file for generate midi later
    with open(os.path.join(TRAIN_DIR, "char2idx.json"), "w") as f:
        json.dump(char2idx, f)

    return char2idx, vocab_size

def __split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

if __name__ == "__main__":
    # Ensure the train directory exists
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test', type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--blocks', type=int, default=4, help="Number of decoder blocks.")
    parser.add_argument('--ffdim', type=int, default=1024, help="Feed Forward layer dimension.")
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs.")
    parser.add_argument('--seqlen', type=int, default=100, help="Sequence lenght.")
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--drop', type=float, default=0.1, help="Dropout.")
    opt = parser.parse_args()

    # 1. Encode midi files as text with vocab
    train_text, train_vocab = me.load(opt.train)
    test_text, test_vocab = me.load(opt.test)

    # 2. Build dictionary to map from char to integers
    char2idx, vocab_size = build_char2idx(train_vocab, test_vocab)

    # 3. Build dataset from encoded unlabelled midis
    train_dataset = build_dataset(train_text, char2idx, opt.seqlen, opt.batch)
    test_dataset = build_dataset(test_text, char2idx, opt.seqlen, opt.batch)

    # 4. Build Transformer model (Replaces build_generative_model)
    # The 'units' and 'layers' LSTM arguments are replaced by 'embed', 'heads', 'ffdim', and 'blocks'
    transformer_model = build_transformer_model(
        vocab_size=vocab_size,
        seq_length=opt.seqlen,
        embed_dim=opt.embed,
        num_heads=opt.heads,
        ff_dim=opt.ffdim,
        num_blocks=opt.blocks,
        batch_size=opt.batch,
        dropout=opt.drop
    )

    if opt.model:
        # If pre-trained model was given as argument, load weights from disk
        print("Loading weights...")
        # Use tf.train.latest_checkpoint for TensorFlow format checkpoints
        transformer_model.load_weights(tf.train.latest_checkpoint(opt.model))
    
    print(transformer_model.summary())

    # 5. Train model
    history = train_generative_model(transformer_model, train_dataset, test_dataset, opt.epochs, opt.lrate)