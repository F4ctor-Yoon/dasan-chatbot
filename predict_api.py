import flask
from flask_cors import CORS, cross_origin
from transformers import TFAutoModelForSeq2SeqLM
import io
import tensorflow as tf
from transformers import AutoTokenizer

app = flask.Flask(__name__)
model = None
tokenizer = AutoTokenizer.from_pretrained("cosmoquester/bart-ko-base")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def load_model():
    global model
    model = TFAutoModelForSeq2SeqLM.from_pretrained('tf_model')

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    data = {"success": False};

    if flask.request.method == "POST":
        quest = flask.request.args["quest"]
        inputs = tokenizer(quest, return_tensors="tf").input_ids
        # outputs = model.generate(inputs, max_new_tokens=46, num_beams=4, do_sample=False, top_k=50, top_p=0.92,)
        outputs = model.generate(inputs, max_new_tokens=60, num_beams=4, do_sample=False, top_k=50, top_p=0.92, num_return_sequences=1)
        # outputs = model.generate(inputs, max_length = 20, num_beams=4, do_sample=False, top_k=10, top_p=0.92, num_return_sequences=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        data["answer"] = answer.split('.')[0]
        data["success"] = True
    
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Flask staring server..."
        "peases wait until server has fully started"))
    load_model()

    app.run(host='0.0.0.0', port='8081')