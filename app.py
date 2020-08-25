from flask import Flask,render_template,request
import numpy as np
import pickle


filename = 'iris.pkl'
classifier = pickle.load(open(filename, 'rb'))


app=Flask(__name__)



@app.route('https://krishna-kumar-prathipati.github.io/krishna1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['Name'])
        glucose = int(request.form['Physics'])
        bp = int(request.form['chemistry'])
        st = int(request.form['Mathematics'])
        data = np.array([[preg, glucose, bp, st]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ =="__main__":
    app.debug=True
    app.run()
