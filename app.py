#import libraries
import os, random               
import numpy as np             
import pandas as pd            
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, Markup
import requests
import io
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import torch
from torchvision import transforms
from utils.model import ResNet9



disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }

crops = {0: 'apple',
 1: 'banana',
 2: 'blackgram',
 3: 'chickpea',
 4: 'coconut',
 5: 'coffee',
 6: 'cotton',
 7: 'grapes',
 8: 'jute',
 9: 'kidneybeans',
 10: 'lentil',
 11: 'maize',
 12: 'mango',
 13: 'mothbeans',
 14: 'mungbean',
 15: 'muskmelon',
 16: 'orange',
 17: 'papaya',
 18: 'pigeonpeas',
 19: 'pomegranate',
 20: 'rice',
 21: 'watermelon'}



features_mean = [ 50.68636364,  53.40965909,  48.13636364,  25.57963982,  71.55999729,
   6.47593982, 102.66125502]   

features_std = [1.39034481e+03, 1.10348388e+03, 2.60749277e+03, 2.58940867e+01,
 4.84781389e+02, 5.97877421e-01, 3.01800594e+03]

features_std = [val**(1/2) for val in features_std]



disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_recommendation_model = tf.keras.models.load_model('models/ann.h5')


app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath(os.getcwd())
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index2.html', title=title)

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)



@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        temperature = int(request.form['temperature'])
        humidity = int(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        normmalize_data = [N, P, K, temperature, humidity, ph, rainfall]
        normmalize_data = [(x-mean_)/std_ for x,mean_,std_ in zip(normmalize_data, features_mean, features_std)]
        
        data = np.array([normmalize_data])
    
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        final_prediction = crops[int(np.argmax(my_prediction, axis=1)[0])]
        return render_template('crop-result.html', prediction=final_prediction, title=title)



# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)




@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filePath)
            print(filePath)
            img = cv2.imread(filePath)
            img  = cv2.resize(img, (256, 256))
            #img = np.array(img)
            #img = img/255
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
            ])
            image = Image.open(filePath)
            img_t = transform(image)
            img_u = torch.unsqueeze(img_t, 0)
            print(filePath, img_t.shape)
            # Get predictions from model
            yb = disease_model(img_u)
            print(yb)
            # Pick index with highest probability
            _, preds = torch.max(yb, dim=1)
            prediction = disease_classes[preds[0].item()]    
            
            
            #img = np.expand_dims(img, axis=0)
            #pred = disease_prediction_model.predict(img)
            #prediction = disease_classes[np.argmax(pred, axis=1)[0]]
            
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)





if __name__ == '__main__':
    app.run(debug=True)
