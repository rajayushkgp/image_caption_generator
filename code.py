import os
import string
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu
# Define file paths
TOKEN_PATH = 'Flickr8k_text/Flickr8k.token.txt'
IMG_FOLDER = 'Flicker8k_Dataset/'

# Load and parse the captions
def load_captions(filename):
    with open(filename, 'r') as file:
        text = file.read()
    captions = {}
    for line in text.split('\n'):
        if len(line) < 2:
            continue
        parts = line.split()
        img_id, img_caption = parts[0], parts[1:]
        img_id = img_id.split('.')[0]
        img_caption = ' '.join(img_caption)
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(img_caption)
    return captions

# Clean the captions
def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in captions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)
    return captions

# Add <start> and <end> tokens to each caption
def add_start_end_tokens(captions):
    processed_captions = {}
    for img_id, cap_list in captions.items():
        processed_captions[img_id] = []
        for cap in cap_list:
            processed_captions[img_id].append(f'startseq {cap} endseq')
    return processed_captions

# --- Running the preprocessing ---
captions = load_captions(TOKEN_PATH)
cleaned_captions = clean_captions(captions)
final_captions = add_start_end_tokens(cleaned_captions)

# Create a flat list of all captions for tokenizer
all_captions_list = []
for key in final_captions:
    for cap in final_captions[key]:
        all_captions_list.append(cap)

# Tokenize the vocabulary (your project mentioned ~7500 words)
# You can adjust num_words to fit your exact vocabulary size.
tokenizer = Tokenizer(num_words=7500, oov_token="<unk>")
tokenizer.fit_on_texts(all_captions_list)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(cap.split()) for cap in all_captions_list)

print(f"Vocabulary Size: {vocab_size}")
print(f"Max Caption Length: {max_length}")
def extract_features(directory):
    # Load ResNet-101 model, excluding the final classification layer
    base_model = ResNet101()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    
    features = {}
    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features

# NOTE: This step takes a while. Run it once and save the results.
# It's recommended to save the features to a file using pickle.

# image_features = extract_features(IMG_FOLDER)
# with open('resnet101_features.pkl', 'wb') as f:
#     pickle.dump(image_features, f)

# To load them back later:
with open('resnet101_features.pkl', 'rb') as f:
    image_features = pickle.load(f)
def define_model(vocab_size, max_length):
    # Feature extractor input
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (merging both inputs)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

model = define_model(vocab_size, max_length)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# Create a data generator
def data_generator(captions, photos, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in captions.items():
            n += 1
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = [], [], []
                n = 0

# --- Training ---
# You would need to split your data into train/validation sets first.
# For simplicity, this example trains on the full dataset.
# The project mentioned two phases of 8 & 10 epochs.
# You can run model.fit() twice to achieve this.

EPOCHS_PHASE_1 = 8
EPOCHS_PHASE_2 = 10
BATCH_SIZE = 64
steps = len(final_captions) // BATCH_SIZE

# Create the generator
generator = data_generator(final_captions, image_features, tokenizer, max_length, vocab_size, BATCH_SIZE)

# Phase 1 Training
print("\n--- Starting Training Phase 1 ---")
model.fit(generator, epochs=EPOCHS_PHASE_1, steps_per_epoch=steps, verbose=1)
model.save('model_phase_1.h5')

# Phase 2 Training (could involve unfreezing encoder layers or just continuing)
print("\n--- Starting Training Phase 2 ---")
model.fit(generator, epochs=EPOCHS_PHASE_2, steps_per_epoch=steps, verbose=1)
model.save('model_final.h5')
