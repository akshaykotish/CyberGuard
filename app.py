from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Load your saved model
# model_path = 'saved_model/crime_classification_model.h5'
model_path = 'fastmodel.h5'
model = tf.keras.models.load_model(model_path)

# Load the tokenizer and label encoders
tokenizer = tf.keras.preprocessing.text.Tokenizer()
label_encoder_cat = LabelEncoder()
label_encoder_subcat = LabelEncoder()
label_encoder_combined = LabelEncoder()

# Load the training data to reconstruct tokenizer and encoders
data = pd.read_csv('train.csv')
data['crimeaditionalinfo'] = data['crimeaditionalinfo'].fillna('')
data['category'] = label_encoder_cat.fit_transform(data['category'])
data['sub_category'] = label_encoder_subcat.fit_transform(data['sub_category'])
data['combined_label'] = data['category'].astype(str) + '_' + data['sub_category'].astype(str)
data['combined_label'] = label_encoder_combined.fit_transform(data['combined_label'])
tokenizer.fit_on_texts(data['crimeaditionalinfo'].astype(str))

# Set the maximum sequence length used during training
max_length = 512

def predict_category_subcategory(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_length, padding='post', truncating='post'
    )

    # Make the prediction
    prediction = model.predict(padded_sequence)
    predicted_label_id = np.argmax(prediction, axis=1)[0]

    # Decode the predicted combined label
    predicted_combined_label = label_encoder_combined.inverse_transform([predicted_label_id])[0]
    category_id, sub_category_id = predicted_combined_label.split('_')
    category = label_encoder_cat.inverse_transform([int(category_id)])[0]
    sub_category = label_encoder_subcat.inverse_transform([int(sub_category_id)])[0]

    return category, sub_category

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    if text.strip() == '':
        return jsonify({'category': '', 'sub_category': ''})
    else:
        category, sub_category = predict_category_subcategory(text)
        return jsonify({'category': category, 'sub_category': sub_category})

if __name__ == '__main__':
    app.run(debug=True)
