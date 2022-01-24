import multiprocessing
import pymongo
import face_recognition
import dlib
import numpy as np
import re
import cv2

client = pymongo.MongoClient(
    host='hostname', port=27017, username='root', password='pass', authSource="admin")
client.admin.command('ping')
db = client["users_db"]
collection = db.webcam_recognize
document_ids = collection.find().distinct('_id')


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def calculate(chunk):
    client = pymongo.MongoClient(
        host='hostname', port=27017, username='root', password='pass', authSource="admin")
    client.admin.command('ping')
    db = client["users_db"]
    collection = db.webcam_recognize
    chunk_result_list = []
    for id in chunk:
        result = collection.find_one(id)
        result1 = result.get('pixelValue')
        flag = 0
        path = "your-path"
        files = "your-image.png"
        filename = str(path+files)
        imgTest = face_recognition.load_image_file(filename)
        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgTest)
        encodesCurFrame = face_recognition.face_encodings(
            imgTest, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            encodeListKnown = tuple(result1[0])
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                flag = 1
                response = "Access Granted"
        if (flag == 0):
            response = "ACCESS DENIED!!"
    chunk_result_list.append(response)
    return chunk_result_list


pool = multiprocessing.Pool(processes=4)
result = pool.map(calculate, list(chunks(document_ids, 1)))
pool.close()
print("Result", result)
