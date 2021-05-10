import uuid
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import numpy
import matplotlib.pyplot as plt
import pandas
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import json
import pickle
import csv
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
import time
from firebase import firebase
import pyrebase
from flask import Flask, render_template
from multiprocessing import Value

import joblib

config = {
	"apiKey": "AIzaSyDGn1mGQwArneAsY2SgJEEERRuW277N-64",
    "authDomain": "roadrought.firebaseapp.com",
    "databaseURL": "https://roadrought.firebaseio.com",
    "projectId": "roadrought",
    "storageBucket": "roadrought.appspot.com",
    "messagingSenderId": "663197101652",
    "appId": "1:663197101652:web:e712035d905f4ee4fe87c1",
    "measurementId": "G-932QWBXMXN"
}

config2 = {
	"apiKey": "AIzaSyBaPgv4WK5_hgZj1LbmNAV61riiXbDdqIU",
  "authDomain": "roadcollect-f08f1.firebaseapp.com",
  "databaseURL": "https://roadcollect-f08f1.firebaseio.com",
  "projectId": "roadcollect-f08f1",
  "storageBucket": "roadcollect-f08f1.appspot.com",
  "messagingSenderId": "667959761015",
  "appId": "1:667959761015:web:f41a91b32085d9687a8e72",
  "measurementId": "G-XRP3BTLJFF"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

firebase2 = pyrebase.initialize_app(config2)
db2 = firebase2.database()


model_c=joblib.load('web_model_C.pkl')
model_ac=joblib.load('web_model_AC.pkl')
Stp=0
Sop=1000

counter = Value("i", 0)
roadcode = Value("i", 0)




#Preprocessing Fucntion
def window(a, w = 1000 , o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides*2
    #st=(8,8)
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
def bincount2D_vectorized(a,bsize):    
    N = bsize
    a_offs = a + (N*np.arange(a.shape[0]))[:,None]
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

def Binning_c(ndata):
    print("Binning")
    ndata=numpy.array(ndata)
    print(ndata.shape)
    dasDigi= window(ndata)
    print(dasDigi.shape)
    bins=np.array([-0.005,-0.0001,0.00009,0.0091,0.1,0.3])
    digidas=np.digitize(dasDigi,bins=bins)
    print(digidas.shape)
    C_digitize=bincount2D_vectorized(digidas,bins.shape[0])
    print("End biniing shape :",C_digitize.shape)
    print("Binning EX:",C_digitize[0:5])
    return  C_digitize

def Binning_ac(ndata):
    print("Binning")
    ndata=numpy.array(ndata)
    print(ndata.shape)
    dasDigi= window(ndata)
    print(dasDigi.shape)
    bins=np.array([-1,-0.009,0.01,0.03,0.05])
    digidas=np.digitize(dasDigi,bins=bins)
    print(digidas)
    C_digitize=bincount2D_vectorized(digidas,bins.shape[0])
    print("End biniing shape :",C_digitize.shape)
    print("Binning EX:",C_digitize[0:5])
    return  C_digitize
import numpy as np
def slidding_feature(ndata): 

    ndata=numpy.array(ndata)
    das= window(ndata).mean(axis=1)
    das=das.reshape(das.shape[0],1)
    print("das1=",das.shape)
    
    #max
    das2 = window(ndata).max(axis=1)
    das2=das2.reshape(das2.shape[0],1)
    print("das2=",das2.shape)
    
    #min
    das3 = window(ndata).min(axis=1)
    das3=das3.reshape(das3.shape[0],1)
    print("das3=",das3.shape)

    #variance
    das4 = window(ndata).var(axis=1)
    das4=das4.reshape(das4.shape[0],1)
    print("das4=",das4.shape)
 
    
    return das,das2,das3,das4
def Featurecom_N(das,das2,das3,das4,C_digitize):
    row,col=das.shape
    datat2=np.empty((row,4))
    label=np.empty((row,1))
    for r in range(row):
        for c in range(4):
            if c==0:
                datat2[r][c]=das[r]
            elif c==1:
                datat2[r][c]=das2[r]
            elif c==2:
                datat2[r][c]=das3[r]
            elif c==3:
                datat2[r][c]=das4[r]
            
    datat2=np.concatenate((datat2, C_digitize),axis=1)
  
    Adatatable=datat2
  

    print(Adatatable[0])
    #print(label)
    return Adatatable
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
@app.route('/')
def index():


    """db_events = db.child("coordinates").get().val().values()
    db_events = list(db_events)
    colors= db.child("colorss").get().val()
    colors = list(colors.values())
    """

    db_events = db.get().val()
    #t_db=db.child("roadname").child("r1").child("colorss").get()
    #print(t_db.val())
    data = list(db_events.items())
    #data2 = list(t_db.val().items())
    #print(data2)
    an_array = np.array(data)
    #an_array2 = np.array(data2)
    #print(an_array2[:,1])

    an_arrays=an_array[0][1] #put data out from big array
    labeling=np.fromiter(an_arrays.keys(), dtype='S100')
    labeling=np.array(labeling, dtype=np.str)
    labeling=labeling.tolist()
    print(len(labeling))
    l_code=[]
    l_Province=[]
    for bindex in range(len(labeling)):
           code_split=labeling[bindex].split()
           l_code.append(code_split[0])
           l_Province.append(code_split[1])
    


    res = np.array([list(an_arrays.values()) for an_arrays in an_arrays.values()])

    print(res.shape)
    latlngs=res[:,1]
    color=res[:,0]
    A=latlngs
    A=np.asarray(A)
    num=A.shape[0]
   
    for i in range(A.shape[0]):
        for j in range(len(A[i])):
            if A[i][j] is None:
                A[i][j]=A[i-num][j-num]
    
    A=A.tolist()    

    B=color
    B=np.asarray(B)
    for i in range(B.shape[0]):
        for j in range(len(B[i])):
            if B[i][j] is None:
                B[i][j]=B[i-num][j-num]
     
    B=B.tolist()

    return render_template("index_.html",color1=B,c1=A,label_code=l_code,label_pro=l_Province)
@app.route('/clearcode', methods=['GET', 'POST'])
def clear_code():
    return render_template("clearcode.html")
@app.route('/clearal', methods=['GET', 'POST'])
def clear_al():
    db2.child("nowinput").update({"province":"no input"})
    db2.child("nowinput").update({"roadcode":"0000"})
    db2.child("nowinput").update({"date":"0-0-0"})
    db2.child("nowinput").update({"type road":"no input"})
    return render_template("clearal.html")

@app.route('/liveview', methods=['GET', 'POST'])
def l_view():
    NoneType = type(None)
    roadcode_n=db2.child("nowinput").child("roadcode").get().val()
    province=db2.child("nowinput").child("province").get().val()
    if type(roadcode_n)=='0000':
        return render_template("dashinw.html")
    if type(province)=='no input':
        return render_template("dashinw.html")
    z=db2.child("spedata").child(roadcode_n).child("IRI").get().val()
    if type(z)==NoneType:
        return render_template("dashinw.html")
    z=np.asarray(z)
    print(z.shape)
    num=z.shape[0]

    zlabel=list(range(0, len(z)))   
    for i in range(z.shape[0]):
        if z[i]is None:
            z[i]=z[i-1]
    z=z.astype(float)
    z=z.tolist()
    roadcode_n=json.dumps(roadcode_n)
    province=json.dumps(province)
    return render_template("liveview.html",Acode=roadcode_n,RIRI=z,RLB=zlabel,Pcode=province)


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    

    return render_template("dashin.html")
@app.route('/inputdb', methods=['GET', 'POST'])
def inputdb():
    data = request.form['roadcode']
    province = request.form['province']
    
    print("Now reading...")
    print("province = ",province)
    print("roadcode = ",data)
    db2.child("nowinputda").update({"province":province})
    db2.child("nowinputda").update({"roadcode":data})
    
    #chart 2
    roadcode_n=db2.child("nowinputda").child("roadcode").get().val()
    province=db2.child("nowinputda").child("province").get().val()
   
   
    roadIRI=db2.child("province").child(province).child(roadcode_n).get().val()
    NoneType = type(None)
    if type(roadIRI)==NoneType:
        alert=1
        alert=json.dumps(alert)
        return render_template("dashinw.html")
    

    roadIRI = list(roadIRI.items())
    roadIRI=np.asarray(roadIRI)
    date_c=roadIRI[:,0]
    iri_c=roadIRI[:,1]

    date_c=date_c.tolist()
    iri_c=iri_c.tolist()

    #chart1
    z=db2.child("spedata").child(data).child("IRI").get().val()
    if type(z)==NoneType:
        return render_template("dashinw.html")
    z=np.asarray(z)
    print(z.shape)
    num=z.shape[0]

    zlabel=list(range(0, len(z)))   
    for i in range(z.shape[0]):
        if z[i]is None:
            z[i]=z[i-1]
    z=z.astype(float)
    z=z.tolist()

    #chart 3
    db_events = db2.child("province").child(province).get().val()
    ch3 = list(db_events.items())
    an_array = np.array(ch3)
    lable_c3=np.array(an_array[:,0], dtype=np.str)
    dat_c3=np.asarray(an_array[:,1])
    charar = np.chararray((lable_c3.shape[0],))
    charar[:] = ':'
    charar=np.array(charar, dtype=np.str)
    print(lable_c3)
    key_lable=[]
    value_l=[]
    for i in range(dat_c3.shape[0]):
        idat=dat_c3[i]
        keys = np.fromiter(idat.keys(), dtype='S10')
        vals = np.fromiter(idat.values(), dtype=float)
        keys=np.array(keys, dtype=np.str)
        last=len(keys)-1
        #print(keys[last])
        #print(vals[last])
        key_lable.append(keys[last])
        value_l.append(vals[last])
    
    print(key_lable)
    print(value_l)
    lable_c3 = np.char.add(lable_c3,charar)
    new_array = np.char.add(lable_c3,key_lable)
    print("new array: ",new_array)
    new_array=new_array.tolist()


        
    

    return render_template("dashboard.html",date=date_c,iri=iri_c,Acode=roadcode_n,RIRI=z,
    RLB=zlabel,c3label=new_array,c3data=value_l,Pcode=province)


@app.route('/input', methods=['GET', 'POST'])
def inputroadcode():
    

    return render_template("input.html")

@app.route('/inputAl', methods=['GET', 'POST'])
def inputAl():
    print("beforee=",roadcode.value)
    bef=roadcode.value
    data = request.form['roadcode']
    date = request.form['date']
    province = request.form['province']
    t_road = request.form['typeroad']
    print("province = ",province)
    print("roadcode = ",data)
    print("date = ",date)
    print("type road = ",t_road)
    db2.child("nowinput").update({"province":province})
    db2.child("nowinput").update({"roadcode":data})
    db2.child("nowinput").update({"date":date})
    db2.child("nowinput").update({"type road":t_road})

    data=int(data)
    with roadcode.get_lock():
        roadcode.value = data
    dbn2=roadcode.value
    dbn2=str(dbn2)
    print(db2.child("spedata").child(dbn2).child("date").child("new").get().val())
    if data!=bef or data==bef and date != db2.child("spedata").child(dbn2).child("date").child("new").get().val() :
        with open("buffer.csv", 'w+') as filename:
             writer = csv.DictWriter(filename, fieldnames=column)
             writer.writeheader()
        filename.close()
        with counter.get_lock():
                counter.value = 0

   
    print("After=",dbn2)
    newd=db2.child("spedata").child(dbn2).child("date").child("new").get().val()
    
    db2.child("spedata").child(dbn2).child("date").update({"last":newd})
    db2.child("spedata").child(dbn2).child("date").update({"new":date})
    
    print(db2.child("spedata").child(dbn2).child("date").child("new").get().val())
    
    z=db2.child("spedata").child(dbn2).child("IRI").get().val()
    z=np.asarray(z)
    if z[0]!=None:        
        num=z.shape[0]
        for i in range(z.shape[0]):
            if z[i]is None:
                 z[i]=z[i-1]
        z=z.astype(float)
           
        mz=np.mean(z)
           
        mz=json.dumps(mz)
        l_date=db2.child("spedata").child(dbn2).child("date").child("last").get().val()
        db2.child("province").child("Bangkok").child(dbn2).update({l_date:mz})
  
    db2.child("spedata").child(dbn2).child("IRI").remove()
    db2.child("spedata").child(dbn2).child("IRI").update({'0':'0.00'})
    return render_template("inputAl.html")


@app.route('/acc', methods=['GET', 'POST'])
def acc():
    if request.method == 'POST' and roadcode.value!=0 :
        global Stp,Sop
        c=counter.value
        c=int(c)
        data = request.get_json()
        
        buffer = open('buffer.csv', 'a+', newline='')
        with buffer as f:
            w = csv.DictWriter(f, data.keys())
            w.writerow(data)
        buffer.close()
        buff_r=pandas.read_csv("buffer.csv")
 
        print(buff_r.shape[0])

        if buff_r.shape[0]>=1000:
            fps = 10
            sine = buff_r["accelerometerAccelerationZ"][int(Stp+counter.value):int(Sop+counter.value)]
            sine=numpy.asarray(sine)
            print(sine[0:5])
            filtered_sine = butter_highpass_filter(sine,0.0475,fps)
            filtered_sine = butter_lowpass_filter(filtered_sine,0.0475,fps)
            print("fsine=",filtered_sine[0:5])
            if(db2.child("nowinput").child("type road").get().val()=='C'):
                C_data=Binning_c(filtered_sine)
            else:
                C_data=Binning_ac(filtered_sine)

            dsd1,dsd2,dsd3,dsd4=slidding_feature(filtered_sine)
            Conclude_data=Featurecom_N(dsd1,dsd2,dsd3,dsd4,C_data)
            
            if(db2.child("nowinput").child("type road").get().val()=='C'):
                result=model_c.predict(Conclude_data)
            else:
                result=model_ac.predict(Conclude_data)
            print("IRI:",result)
            result=np.mean(result)
            
            co_start = [buff_r['locationLatitude'][0],
                          buff_r['locationLongitude'][0]]
            co_end = [buff_r['locationLatitude'][999],
                          buff_r['locationLongitude'][999]]
            color_iri='green'
            if 2.5<=result<=3.5:
                color_iri = 'yellow'
            elif 3.5<=result<=5.0:
                color_iri = 'orange'
            elif result >= 5.0:
                color_iri = 'red'
            result=json.dumps(result)

            
            dbn=roadcode.value
            print(dbn)

            
            date=db2.child("spedata").child(dbn).child("date").child("new").get().val()
            dbn=str(dbn)
            A1=0+c
            B2=1+c
            province_get=db2.child("nowinput").child("province").get().val()
            dbno=dbn
            dbn=dbn+" "+province_get
            print('dbno=',dbno)
            
            dbn=str(dbn)
            db.child("roadname").child(dbn).child("coordinate").update({A1:co_start})
            db.child("roadname").child(dbn).child("coordinate").update({B2:co_end})
            db2.child("spedata").child(dbno).child("IRI").update({A1:result})
            db2.child("spedata").child(dbno).child("IRI").update({B2:result})
            
            
            db.child("roadname").child(dbn).child("colorss").update({A1:color_iri})
            db.child("roadname").child(dbn).child("colorss").update({B2:color_iri})
            with counter.get_lock():
                counter.value +=1

            z=db2.child("spedata").child(dbno).child("IRI").get().val()
            z=np.asarray(z)
            
            num=z.shape[0]
            for i in range(z.shape[0]):
                if z[i]is None:
                        z[i]=z[i-1]
            z=z.astype(float)
           
            mz=np.mean(z)
           
            mz=json.dumps(mz)
           
            db2.child("province").child("Bangkok").child(dbno).update({date:mz})
            print(A1,B2)
            print('Record Updated')
    elif request.method == 'POST' and roadcode.value==0 :
        print("Please Enter Road code")


    return jsonify('got it')








if __name__ == '__main__':
    
    filename = "buffer.csv"
    column=['accelerometerAccelerationX', 'accelerometerAccelerationY', 
    'accelerometerAccelerationZ', 'accelerometerTimestamp_sinceReboot', 
    'identifierForVendor', 'locationAltitude', 
    'locationCourse', 'locationFloor', 
    'locationHorizontalAccuracy', 'locationLatitude', 
    'locationLongitude', 'locationSpeed', 'locationTimestamp_since1970', 
    'locationVerticalAccuracy', 'loggingTime']
    with open(filename, 'w+') as filename:
        writer = csv.DictWriter(filename, fieldnames=column)
        writer.writeheader()
    filename.close()
	
    app.run(host='192.168.1.15',port=5000,debug=True)
    #app.run(host='10.90.7.197',port=5000,debug=True)
    

