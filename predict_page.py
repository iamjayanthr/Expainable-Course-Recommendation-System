from __future__ import print_function
import streamlit as st
import dill
import pandas as pd
import time
import streamlit.components.v1 as components

def load_model():
    with open('svc_model1.pkl','rb') as file:
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

def show_predict_page():
    st.title("Career Recommendation System")


    
    qcreative=("I am very creative","I am somewhat creative","I am not creative")
    
    qoutdoorwork=("I love the outdoors and wish I could work outside every day","Working outside would be okay, but only for short periods of time","I would rather work inside")
    
    qfuture=("Building things with moving parts","Designing buildings","Designing or building sensor based technology",
               "Improving the way we use the world's resources","Making discoveries at the molecular level","Programming apps","Optimizing processes")
    
    qgroupwork=("I enjoy working with others","I occasionally like working with others","I do not like working as part of a team. I would rather work alone")

    qwriting=("Excited! I can share my theories with the world","A bit apprehensive. I get overwhelmed with so many options","Annoyed. I would much rather be given a topic with clear instructions")

    qsubjectlike=("Biology","Business","Chemistry","Computer Science","Social Studies","Visual Arts","Maths","Physics","Language Arts")

    qsubjectdislike=("Biology","Business","Chemistry","Computer Science","Social Studies","Visual Arts","Maths","Physics","Language Arts")

    qclublike=("Art or design club","Autoshop club","Business club","Consulting club","Environment club","Robotics club","Hacker club","Student council")

    qclubdislike=("Art or design club","Autoshop club","Business club","Consulting club","Environment club","Robotics club","Hacker club","Student council")

    qprojectlike=("Prototyping a musical instrument for children","Designing an Olympic village","Programming a robot that can make you dinner","Building the world's most powerful supercomputer","Designing a low power invertor","Prototyping a Blood pressure monitor","Optimizing the Uber Pool routes")

    qprojectdislike=("Prototyping a musical instrument for children","Designing an Olympic village","Programming a robot that can make you dinner","Building the world's most powerful supercomputer","Designing a low power invertor","Prototyping a Blood pressure monitor","Optimizing the Uber Pool routes")

    qnoeng=("Science","Business","Computer Science","English Literature","Environmental Studies","Graphic Design","Health Studies","Maths","Visual Arts")

    qdraw=("Really good, I can draw just about anything","I am not the best, but I am not the worst","I am not very good")

    qfuturejob=("Automotive","Entrepreneurship","Construction","Health","Environment","Manufacturing","Technology")

    qinterest=("CIVIL","CSE","ECE","EEE","MECH","BME")

    name=st.text_input("Enter name")
    tenth=st.number_input("Enter 10th percentage ")
    plus2=st.number_input("Enter 12th percentage ")
    creative=st.selectbox("How creative are you ?",qcreative)
    writing=st.selectbox("How do you feel about writing an essay ?",qwriting)    
    outdoorwork=st.selectbox("How comfortable is working in the outdoors ?",qoutdoorwork)
    future=st.selectbox("Which one of these choices would you prefer the most?",qfuture)
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
    if ok:
        X1={"creative":[creative],
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
        "futurejob":[futurejob]}
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
        X["drawing"]= le_draw.transform(X["drawing"])
        

        subject=svc.predict(X)
        st.subheader(f"Hi {name}")
        if(subject==interest):
           st.subheader(f"Your choice is correct. i.e, {subject} is the best option for you")
       
        else:
            st.subheader(f"It's better you choose {subject} than {interest}")
        print(X)
        with st.spinner("Processing your input.."):
            time.sleep(3)
        with st.spinner('Fetching Result Explanation'):
            predict_fn_rf = lambda x: svc.predict_proba(x).astype(float)
            exp = explainer.explain_instance(X.loc[0].values, predict_fn_rf,num_features=14,top_labels=3)
            st.subheader("Result Explanation")
            components.html(exp.as_html(), height=1800)
            
          
    





            

            

           
            


#conda activate ml
#streamlit run app.py