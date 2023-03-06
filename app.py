from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import  os
app = Flask(__name__)

model=pickle.load(open('diabetes.pkl','rb'))


@app.route('/')
def hello_world():

    return render_template("diabetes.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
    Pregnancies=request.form['Pregnancies']
    BloodPressure=request.form['BloodPressure']
    Glucose=request.form['Glucose']
    SkinThickness=request.form['SkinThickness']
    Insulin=request.form['Insulin']
    BMI=request.form['BMI']
    DiabetesPedigreeFunction=request.form['DiabetesPedigreeFunction']
    Age=request.form['Age']
    Pregnancies =int(Pregnancies)
    BloodPressure = int(BloodPressure)
    Glucose = int(Glucose)
    SkinThickness = int(SkinThickness)
    Insulin = int(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction =float(DiabetesPedigreeFunction)
    Age = int(Age)
    final_features=np.array([(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)])
    prediction=model.predict(final_features)
    print(prediction)
    output =(prediction[0])

    if output == 1:
        return render_template('result.html',
                               pred='Your health  in Danger.\n we detect you have a diabetes {}'.format(output))
    else:
        return render_template('result.html',
                               pred="   you don't have a diabetes \n your healt is safe {}".format(output))



if __name__ == '__main__':
    app.run(debug=True)
