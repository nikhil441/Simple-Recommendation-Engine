import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

# Load data (using MovieLens dataset for demonstration purposes)
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as z:
    with z.open('ml-latest-small/ratings.csv') as f:
        df = pd.read_csv(f)

# Preprocess data
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.25)

# Build the recommendation model using SVD
model = SVD()
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
accuracy.rmse(predictions)

# Save the model
import pickle
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create Flask app for serving recommendations
from flask import Flask, request, jsonify

# Load the model
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.json['user_id'])
    item_id = int(request.json['item_id'])
    prediction = model.predict(user_id, item_id)
    return jsonify({'user_id': user_id, 'item_id': item_id, 'rating': prediction.est})

if __name__ == '__main__':
    app.run(debug=True)