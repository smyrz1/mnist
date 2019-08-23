from flask import Flask, render_template, request
import numpy as np
import time
import re
import base64
import keras.models
import cv2
from keras.models import model_from_json
import tensorflow as tf
import recorder

# initalize our flask app
mnist_web = Flask(__name__, template_folder='templates')
global model, graph

available_type = set(['png', 'jpg', 'jpeg'])


def init():
    file = open('m_lenet.json', 'r')
    mnist_model = file.read()
    file.close()
    loaded_model = model_from_json(mnist_model)
    loaded_model.load_weights("m_weights.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img = tf.get_default_graph()

    return loaded_model, img


# initialize variables
model, graph = init()


@mnist_web.route('/')
def index():

    return render_template("index.html")


def convert_img(img_data):

    with open('digit.png', 'wb') as output:
        output.write(base64.b64decode(re.search(b'base64,(.*)', img_data).group(1)))


@mnist_web.route('/predict/', methods=['GET', 'POST'])
def predict():

    img_data = request.get_data()
    convert_img(img_data)
    img = cv2.imread('digit.png', 0)
    img = cv2.resize(img, (28, 28))
    img = np.invert(img)  # change black to white, vice versa.
    img = img.reshape(1, 28, 28, 1)
    with graph.as_default():
        output = model.predict(img)
        response = np.array_str(np.argmax(output, axis=1))[1]

        # record data into cassandra with the Keyspace mnist_space, table recorder.
        ticks = int(round(time.time() * 1000))
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ticks / 1000))
        recorder.insertData(now, response)

        return response


if __name__ == "__main__":
    # run mnist_train.py first to train your model.
    recorder.createKeySpace()
    mnist_web.run(host='0.0.0.0')

