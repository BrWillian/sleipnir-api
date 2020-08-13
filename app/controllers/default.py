"""
@Author Willian Antunes
Vizentec 2020
"""
from app import app
from flask import request
from numpy import fromstring, uint8, array, squeeze, float32
from cv2 import imdecode, resize, IMREAD_COLOR, INTER_CUBIC, imencode
from base64 import b64encode, b64decode
from app.tf_model.VtNet import VtNet


model = VtNet.build(32,32, 3, 2)
model.summary()

@app.route('/api', methods=['POST', 'GET'])
def index():
	if request.method == "POST":
		try:
			imgb64 = request.data['image']
			img = fromstring(b64decode(imgb64), uint8)
			img = imdecode(img, IMREAD_COLOR)
			img = resize(img, (32, 32), INTER_CUBIC)

			print(img)
		except:
			return {
				"img": "imagem"
			}

	return {'url': 'teste'}
