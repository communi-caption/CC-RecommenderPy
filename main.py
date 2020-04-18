from flask import Flask, request, jsonify
from train_d2v import setTrainingData
from similarity import mostSimilarDocument

app = Flask(__name__)

@app.route("/train", methods=['POST'])
def train():
    trainFile = request.json
    setTrainingData(trainFile)
    return "ok"

@app.route("/similarity", methods=['POST'])
def similarity():
    docFile = request.json
    return jsonify(
        id = int(mostSimilarDocument(docFile["data"]))
    )

if __name__ == '__main__':
    app.run(debug=False, port=5006)