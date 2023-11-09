from flask import Flask,request, render_template
from data import pd, model

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.form.to_dict()
    features = pd.DataFrame(data, index=[0])
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
