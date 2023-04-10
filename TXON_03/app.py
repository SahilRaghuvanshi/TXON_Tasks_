import face_recognition
import cv2
import pandas as pd
import numpy as np
import csv
import os
import os.path
from datetime import datetime
import uuid
import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as com
with open("style.css") as source:
    design=source.read()
com.html(f"""<style>{design}</style><div class="heading">Face Recognition Based Attendance System</div>""")
i=1
lst=[]
student1=face_recognition.load_image_file("photos/salmankhan.jpg")
student1_encoding=face_recognition.face_encodings(student1)[0]
student2=face_recognition.load_image_file("photos/shahrukh.jpg")
student2_encoding=face_recognition.face_encodings(student2)[0]
student3=face_recognition.load_image_file("photos/hritik.jpg")
student3_encoding=face_recognition.face_encodings(student3)[0]
student4=face_recognition.load_image_file("photos/sahil.jpg")
student4_encoding=face_recognition.face_encodings(student4)[0]
student5=face_recognition.load_image_file("photos/bhavesh.jpg")
student5_encoding=face_recognition.face_encodings(student5)[0]
known_face_encoding=[student1_encoding,student2_encoding,student3_encoding,student4_encoding,student5_encoding]
known_faces_names = ["Salman Khan","Shahrukh Khan","Hritik Roshan","Sahil Raghuvanshi","Bhavesh Badgujar"]
students=known_faces_names.copy()
col1,col2=st.columns(2)
col2.markdown("<h3 style='color:red;'>Present Students</h3>",unsafe_allow_html=True)
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
print("executed")
fname=current_date+'.csv'
if not os.path.exists(fname):
    print("not exists")
    with open(current_date+'.csv','w+',newline='') as f:
        lnwriter = csv.writer(f)
        lnwriter.writerow(["Names","Reporting Time"])
filename="./"+str(current_date)+'.csv'
empty_widget=col2.empty()
img_file_buffer = col1.camera_input("CLICK A PHOTO")
if img_file_buffer is not None:
    bytes_data =img_file_buffer.getvalue()
    image_array=cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)
    image_array_copy=cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)
    face_locations=face_recognition.face_locations(image_array)
    encodeCurFrame = face_recognition.face_encodings(image_array,face_locations)
    for idx,(top,right,bottom,left) in enumerate (face_locations):
        image_array=cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image_array,(left,top),(right,bottom),(255,0,0),3)
    st.subheader("FACE RECOGNIZED")
    st.image(image_array)
    face_names=[]
    for face_encoding in encodeCurFrame:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    with open(current_date+'.csv','a+',newline='') as f:
                        lnwriter = csv.writer(f)
                        lnwriter.writerow([name,current_time])
df=pd.read_csv(filename)
empty_widget.table(df)
                
                    

    
        




