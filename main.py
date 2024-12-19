from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#importing pickle files
model = pickle.load(open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/fertilizer.pkl','rb'))
ferti = pickle.load(open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/encoder_fert.pkl','rb'))

@app.route('/')
def home():
    return render_template('plantindex.html')

@ app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@ app.route('/Detail')
def Detail():
    return render_template('Detail.html')



@app.route('/predict',methods=['POST'])
def predict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    if None in (temp, humi,soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi,soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

# Convert values to integers
    input = [int(temp), int(humi), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    res = ferti.classes_[model.predict([input])]
    return render_template('Model1.html', x=res)
if __name__ == "__main__":
    app.run(debug=True)