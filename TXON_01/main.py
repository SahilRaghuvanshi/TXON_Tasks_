import streamlit as st
import numpy as np 
import cv2
st.title("Black and White Image Colorization")
st.markdown("""<style> 
                        .css-cio0dv.egzxvld1{visibility:hidden;}
                        body {
                        background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
                        background-size: cover;
                        }
                       .stVideo{border:10px solid white; border-radius:20px}
               </style>""", unsafe_allow_html=True)
print("loading models.....")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]
img=st.file_uploader("Please Upload Image",type=["png","jpg","jpeg"])
if img is not None:
    st.image(img)
    btn=st.button("Convert")
    if btn:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        scaled = image.astype("float32")/255.0
        lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab,(224,224))
        L = cv2.split(resized)[0]
        L -= 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1,2,0))
        ab = cv2.resize(ab, (image.shape[1],image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized,0,1)
        colorized = (255 * colorized).astype("uint8")
        status = cv2.imwrite('C:/Users/Sahil Raghuvanshi/Desktop/Projects/BlackAndWhiteToColor/Library/'+img.name,colorized)
        st.image('./Library/'+img.name)
        cv2.waitKey(0)