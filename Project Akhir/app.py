from flask import Flask, render_template, Response, request
import cv2
import datetime
import time
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from threading import Thread
from keras.preprocessing import image

app = Flask(__name__)

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 1
switch = 1
rec = 0
predictions = None

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

camera = cv2.VideoCapture(0)

model = keras.models.load_model('saved_model/face_model_madel.h5')
class_names = ['Ali', 'Fikri', 'Irawan', 'Irsyad', 'Salsa']


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if(face):
                frame = detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame = cv2.bitwise_not(frame)
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "temp.png"])
                # p = os.path.sep.join(
                # ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if(rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(
                    frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ambil')
def ambil():
    return render_template('ambil.html')


# @app.route('/absen')
# def absen():
#     return render_template('absen.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ambil/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        # elif request.form.get('grey') == 'Grey':
        #     global grey
        #     grey = not grey
        # elif request.form.get('neg') == 'Negative':
        #     global neg
        #     neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if(face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if(switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if(rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif(rec == False):
                out.release()

    elif request.method == 'GET':
        return render_template('ambil.html')
    return render_template('ambil.html')


@app.route('/absen')
def absen():
    global capture, predictions
    img = tf.keras.preprocessing.image.load_img(
        os.path.sep.join(['shots', "temp.png"]), target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.vstack([img_array])

    predictions = model.predict(img_array)
    predictions = tf.nn.softmax(predictions[0])
    predictions = class_names[np.argmax(predictions)]
    print(predictions)
    # biodata = 0
    if predictions == 'Ali':
        nim = '10000'
    elif predictions == 'Fikri':
        nim = '001'
    elif predictions == 'Irawan':
        nim = '002'
    elif predictions == 'Irsyad':
        nim = '003'
    elif predictions == 'Salsa':
        nim = '004'
    return render_template('absen.html', nama_karyawan=predictions, nim_karyawan=nim)
