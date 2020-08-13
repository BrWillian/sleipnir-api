"""
@Author Willian Antunes
Vizentec 2020
"""
from app import app
from flask import request
from numpy import fromstring, uint8, array, squeeze, float32
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from cv2 import imdecode, resize, IMREAD_COLOR, INTER_CUBIC, imencode, putText
from base64 import b64encode, b64decode


model = load_model('tf_model/model.hdf5')

@app.route('/api', methods=['POST', 'GET'])
def index():
	if request.method == "POST":
		try:
			imgb64 = request.data['image']
			img = fromstring(b64decode(imgb64), uint8)
			img = imdecode(img, IMREAD_COLOR)
			orig = img.copy()
			img = resize(img, (28, 28), INTER_CUBIC)
			img = img.astype('float') / 255.0
			img = img_to_array(img)
			img = np.expand_dmis(img, axis=0)

			(usable, unusable) = model.predict(img)[0]

			label = "usable" if usable > unusable else "unusable"
			prob = usable if usable > unusable else unusable

			label_final = "{}: {:.2f}%".format(label, proba * 100)

			output = imutils.resize(orig, width=800)
			putText(output, label_final, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

			_, pred = imencode('.jpg', output)
            pred = pred.tobytes()
            pred = b64encode(pred)

            return {
            	'image': pred.decode('utf-8'),
            	'pred': label
            }

		except:
			return {
				"Error": "Runtime error"
			}

	return {'url': 'teste'}
