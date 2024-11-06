import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load the CSV file
data = pd.read_csv('train.csv')

# Fill missing values in 'crimeaditionalinfo' column
data['crimeaditionalinfo'] = data['crimeaditionalinfo'].fillna('')

# Encode the labels (category and sub_category)
label_encoder_cat = LabelEncoder()
label_encoder_subcat = LabelEncoder()
data['category'] = label_encoder_cat.fit_transform(data['category'])
data['sub_category'] = label_encoder_subcat.fit_transform(data['sub_category'])

# Concatenate category and sub_category to create a combined label
data['combined_label'] = data['category'].astype(str) + '_' + data['sub_category'].astype(str)
label_encoder_combined = LabelEncoder()
data['combined_label'] = label_encoder_combined.fit_transform(data['combined_label'])

# Split the data into training and testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['crimeaditionalinfo'], data['combined_label'], test_size=0.2, random_state=42
)

# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_texts.astype(str))

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_length = 256  # Reduce max_length for faster training
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Build a simplified model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder_combined.classes_), activation='softmax')
])

# Compile the model with a simpler optimizer for faster training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=15, batch_size=64, validation_data=(test_padded, test_labels))  # Fewer epochs and larger batch size for speed

# Evaluate the model
loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy}")

# Save the model
fast_model_path = 'fastmodel.h5'
model.save(fast_model_path, overwrite=True)
print(f"Model saved as {fast_model_path}")
