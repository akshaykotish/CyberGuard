import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your saved model
model_path = 'saved_model/crime_classification_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the tokenizers and label encoders
tokenizer = tf.keras.preprocessing.text.Tokenizer()
label_encoder_cat = LabelEncoder()
label_encoder_subcat = LabelEncoder()
label_encoder_combined = LabelEncoder()

# Assuming you have the training data or label encoder mappings saved
data = pd.read_csv('train.csv')
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
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Make the prediction
    prediction = model.predict(padded_sequence)
    predicted_label_id = np.argmax(prediction, axis=1)[0]
    
    # Decode the predicted combined label
    predicted_combined_label = label_encoder_combined.inverse_transform([predicted_label_id])[0]
    category, sub_category = predicted_combined_label.split('_')
    category = label_encoder_cat.inverse_transform([int(category)])[0]
    sub_category = label_encoder_subcat.inverse_transform([int(sub_category)])[0]
    
    return category, sub_category

# Test the function
sample_text = "Online Financial Fraud,Internet Banking Related Fraud,NameManik Varban SO Nirmal Varban"
predicted_category, predicted_sub_category = predict_category_subcategory(sample_text)
print(f"Predicted Category: {predicted_category}, Predicted Sub-category: {predicted_sub_category}")
