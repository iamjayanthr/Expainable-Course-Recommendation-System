from __future__ import print_function
from matplotlib.pyplot import draw
import streamlit as st
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components



df=pd.read_csv("datanum1.csv")
l=len(df.index)
df=df.drop('Timestamp',axis = 1)
df=df.drop('Name', axis = 1)
df=df.drop('satisfied', axis = 1)
df=df.drop('12th Stream',axis = 1)
sub=['CIVIL','CSE','ECE','EEE','MECH','BME']

def load_model():
    with open('svc_model.pkl','rb') as file:
        data = dill.load(file)
    return data
    
data = load_model()

svc=data["model"]
explainer=data["exp"]
le_future=data["le_future"]
le_group=data["le_group"]
le_create=data["le_create"]
le_write=data["le_write"]
le_out=data["le_out"]
le_enjoymost=data["le_enjoymost"]
le_enjyleast=data["le_enjyleast"]
le_clubmost=data["le_clubmost"]
le_clubleast=data["le_clubleast"]
le_projectliked=data["le_projectliked"]
le_projectdisliked=data["le_projectdisliked"]
le_eng=data["le_eng"]
le_draw=data["le_draw"]
le_job=data["le_job"]
le_interest=data["le_interest"]

df["creative"]= le_create.transform(df["creative"])
df["writing"]= le_write.transform(df["writing"])
df["outdoorwork"]= le_out.transform(df["outdoorwork"])
df["groupwork"]= le_group.transform(df["groupwork"])
df["enjoymost"]= le_enjoymost.transform(df["enjoymost"])
df["enjoyleast"]= le_enjyleast.transform(df["enjoyleast"])
df["clubmost"]= le_clubmost.transform(df["clubmost"])
df["clubleast"]= le_clubleast.transform(df["clubleast"])
df["projectliked"]= le_projectliked.transform(df["projectliked"])
df["projectdisliked"]= le_projectdisliked.transform(df["projectdisliked"])
df["noengineering"]= le_eng.transform(df["noengineering"])
df["futurejob"]= le_job.transform(df["futurejob"])
df["interest"]= le_interest.transform(df["interest"])
df["drawing"]= le_draw.transform(df["drawing"])

print(df.head(2))


feature=df.drop('Opted',axis = 1)
target=df["Opted"]
X_train, X_test, y_train, y_test = train_test_split(feature,target, test_size = 0.4)
xt=X_train.values
#predict_fn_rf = lambda x: svc.predict_proba(x).astype(float)
#explainer = lime.lime_tabular.LimeTabularExplainer(xt,feature_names =["creative","writing","outdoorwork","future","groupwork","enjoymost","enjoyleast","clubmost","clubleast","projectliked","projectdisliked","noengineering","futurejob","drawing","interest"],mode="classification",kernel_width=7)

def show_predict_page():
    st.title("Hii")


    
    qcreative=("I am very creative","I am somewhat creative","I am not creative")
    
    qoutdoorwork=("I love the outdoors and wish I could work outside every day","Working outside would be okay, but only for short periods of time","I would rather work inside")
    
    qfuture=("Building things with moving parts","Designing buildings","Designing or building sensor based technology",
               "Improving the way we use the world's resources","Making discoveries at the molecular level","Programming apps","Optimizing processes")
    
    qgroupwork=("I enjoy working with others","I occasionally like working with others","I do not like working as part of a team. I would rather work alone")

    qwriting=("Excited! I can share my theories with the world","A bit apprehensive. I get overwhelmed with so many options","Annoyed. I would much rather be given a topic with clear instructions")

    qsubjectlike=("Autoshop","Biology","Business","Chemistry","Computer Science","Geography","Visual Arts","History","Math","Physics","Language Arts")

    qsubjectdislike=("Autoshop","Biology","Business","Chemistry","Computer Science","Geography","Visual Arts","History","Math","Physics","Language Arts")

    qclublike=("Art or design club","Autoshop club","Business club","Consulting club","Environment club","Robotics club","Hacker club","Student council")

    qclubdislike=("Art or design club","Autoshop club","Business club","Consulting club","Environment club","Robotics club","Hacker club","Student council")

    qprojectlike=("Prototyping a musical instrument for children","Designing an Olympic village","Programming a robot that can make you dinner","Building the world's most powerful supercomputer","Designing a water treatment system for Mars","Creating a battery from recycled material","Optimizing the Uber Pool routes")

    qprojectdislike=("Prototyping a musical instrument for children","Designing an Olympic village","Programming a robot that can make you dinner","Building the world's most powerful supercomputer","Designing a water treatment system for Mars","Creating a battery from recycled material","Optimizing the Uber Pool routes")

    qnoeng=("Applied Science","Business","Computer Science","Economics","English Literature","Environmental Studies","Finance","Geography","Graphic Design","Health Studies","Marketing","Math","Political Science","Psychology","Visual Arts")

    qdraw=("Really good, I can draw just about anything","I am not the best, but I am not the worst","I am not very good")

    qfuturejob=("Architecture","Automotive","Entrepreneurship","Construction","Health","Environment","Manufacturing","Technology")

    qinterest=("CIVIL","CSE","ECE","EEE","MECH","BME")

    name=st.text_input("Enter name")
    tenth=st.number_input("Enter 10th percentage ")
    plus2=st.number_input("Enter 12th percentage ")
    creative=st.selectbox("How creative are you ?",qcreative)
    writing=st.selectbox("Feelings about writing skills ?",qwriting)    
    outdoorwork=st.selectbox("How comfortable is working in the outdoors ?",qoutdoorwork)
    future=st.selectbox("Future work",qfuture)
    groupwork=st.selectbox("Does working in groups excite you ?",qgroupwork)
    subjectlike=st.selectbox("Most enjoyed subject in high school ?",qsubjectlike)
    subjectdisliked=st.selectbox("Most diliked subject in high school ?",qsubjectdislike)
    clubliked=st.selectbox("Interested club for joining?",qclublike)
    clubdisliked=st.selectbox("Least interested club for joining?",qclubdislike)
    projectliked=st.selectbox("What project would you want to be a part of ?",qprojectlike)
    projectdisliked=st.selectbox("What project would you dont want to be a part of ?",qprojectdislike)
    noeng=st.selectbox("If engineering didn't exist, which course will you go for ?",qnoeng)
    draw=st.selectbox("How will you rate your drawing skills?",qdraw)
    futurejob=st.selectbox("What industry can you see yourself working in, in the future ?",qfuturejob)
    interest=st.selectbox("Interested course ?",qinterest)

    ok=st.button("Predict")
   # ok1=st.button("PredictArray")
    if ok:
        X1={"10th":[tenth],
        "12th":[plus2],
        "creative":[creative],
        "writing":[writing],
        "outdoorwork":[outdoorwork],
        "future":[future],
        "groupwork":[groupwork],
        "enjoymost":[subjectlike],
        "enjoyleast":[subjectdisliked],
        "clubmost":[clubliked],
        "clubleast":[clubdisliked],
        "projectliked":[projectliked],
        "projectdisliked":[projectdisliked],
        "noengineering":[noeng],
        "drawing":[draw],
        "futurejob":[futurejob],
        "interest":[interest]}
        X=pd.DataFrame(X1) 

        X["creative"]= le_create.transform(X["creative"])
        X["writing"]= le_write.transform(X["writing"])
        X["outdoorwork"]= le_out.transform(X["outdoorwork"])
        X["future"]= le_future.transform(X["future"])
        X["groupwork"]= le_group.transform(X["groupwork"])
        X["enjoymost"]= le_enjoymost.transform(X["enjoymost"])
        X["enjoyleast"]= le_enjyleast.transform(X["enjoyleast"])
        X["clubmost"]= le_clubmost.transform(X["clubmost"])
        X["clubleast"]= le_clubleast.transform(X["clubleast"])
        X["projectliked"]= le_projectliked.transform(X["projectliked"])
        X["projectdisliked"]= le_projectdisliked.transform(X["projectdisliked"])
        X["noengineering"]= le_eng.transform(X["noengineering"])
        X["futurejob"]= le_job.transform(X["futurejob"])
        X["interest"]= le_interest.transform(X["interest"])
        X["drawing"]= le_draw.transform(X["drawing"])
        

        subject=svc.predict(X)
        st.subheader(f"Hi {name}")
        if(subject==interest):
            st.subheader(f"Your choice is correct. i.e, {subject} is the best option for you")
        else:
            st.subheader(f"It's better you choose {subject} than {interest}")
        print(X)
        predict_fn_rf = lambda x: svc.predict_proba(x).astype(float)
        exp = explainer.explain_instance(X.loc[0].values, predict_fn_rf,num_features=17,top_labels=3)
        fig1=exp.show_in_notebook()
        fig = exp.as_pyplot_figure()
        #st.write(fig)
        components.html(exp.as_html(), height=800)
   
    





            

            

           
            


#conda activate ml
#streamlit run app.py