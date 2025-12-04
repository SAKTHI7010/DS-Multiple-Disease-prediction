import streamlit as st
import pandas as pd
import pickle


with open('liver_model.pkl','rb') as fi:
  model_liv = pickle.load(fi)

with open('liver_enc1.pkl','rb') as el:
  liver_enc = pickle.load(el)

# kidney model
with open('kidney_model.pkl','rb') as f:
  model_kid = pickle.load(f)

with open('enc_kidney1.pkl','rb') as e:
  kid_enc = pickle.load(e)

# parkinson model
with open('parkison_model.pkl','rb') as n:
  parkison_brain = pickle.load(n)





# -------------------------------
#   NAVIGATION BAR
# -------------------------------
st.set_page_config(page_title="Medical Prediction System", layout="wide")

menu = ["🏠Home", "Prediction","Hosipital Spiscelity","Career","contacts Us"]
choice = st.sidebar.selectbox("Navigation", menu)


#   HOME PAGE

if choice == "🏠Home":
    st.title("🏥 VELAN Medical Disease Prediction System")
    st.write("""
        This application predicts health conditions using Machine Learning models:
        - Kidney Disease  
        - Liver Disease  
        - Parkinson's Disease  
        
        Use the sidebar to navigate between pages.
    """)
    st.image("C:/Users/svel2/Downloads/forstrem.jpg")


#   PREDICTION PAGE

elif choice == "Prediction":

    st.title("🧠 Disease Prediction Panel")

    # Sub-menu inside Prediction
    sub_menu = ["Kidney Prediction", "Liver Prediction", "Parkinson Prediction"]
    sub_choice = st.selectbox("Choose Prediction Type", sub_menu)

    
    # Kidney Prediction
    
    if sub_choice == "Kidney Prediction":

        st.header("🩸 Kidney Disease Prediction")

        with st.form("kidney_form"):
         age = st.number_input("Age", min_value=1.0, max_value=120.0, format="%.1f")
         bp = st.number_input("Blood Pressure (bp)", min_value=50.0, max_value=200.0, format="%.1f")
         al = st.number_input("Albumin (al)", min_value=0.0, max_value=5.0, format="%.1f")
         su = st.number_input("Sugar (su)", min_value=0.0, max_value=5.0, format="%.1f")
         bu = st.number_input("Blood Urea (bu)", min_value=0.0, max_value=400.0, format="%.1f")
         sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=15.0, format="%.2f")
         hemo = st.number_input("Hemoglobin (hemo)", min_value=3.0, max_value=20.0, format="%.1f")

         submitted1 = st.form_submit_button("🔍PREDICT")

        if submitted1:
         input_data = pd.DataFrame({
          "age": [age],
          "bp": [bp],
          "al": [al],
          "su": [su],
          "bu": [bu],
          "sc": [sc],
          "hemo": [hemo]
        })

        prediction = model_kid.predict(input_data)
        result = "NORMAL" if prediction[0] == 1 else "DANGER"

        st.success(f"Kidney result: {result}")





    elif sub_choice == "Liver Prediction":

        st.header("🫁 Liver Disease Prediction")

        with st.form("liver_form"):
            Age = st.number_input("Age", min_value=1, max_value=120, step=1)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0, format="%.2f")
            Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, format="%.2f")
            Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0, format="%.2f")
            ALT = st.number_input("Alamine Aminotransferase (ALT)", min_value=0.0, format="%.2f")
            AST = st.number_input("Aspartate Aminotransferase (AST)", min_value=0.0, format="%.2f")
            Albumin = st.number_input("Albumin", min_value=0.0, format="%.2f")

            submitted2 = st.form_submit_button("🔍 PREDICT")

        if submitted2:
            df = pd.DataFrame([{
                "Age": Age,
                "Gender": Gender,
                "Total_Bilirubin": Total_Bilirubin,
                "Direct_Bilirubin": Direct_Bilirubin,
                "Alkaline_Phosphotase": Alkaline_Phosphotase,
                "Alamine_Aminotransferase": ALT,
                "Aspartate_Aminotransferase": AST,
                "Albumin": Albumin
            }])

            df["Gender"] = liver_enc.transform(df["Gender"])
            pred = model_liv.predict(df)[0]
            result = "NORMAL" if pred == 1 else "DANGER"

            st.success(f"Liver result: {result}")

   
    # Parkinson Prediction
    
    elif sub_choice == "Parkinson Prediction":

        st.header("🧠 Parkinson Disease Prediction")

        with st.form("parkinson_form"):
          Fo= st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.3f")
          Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.3f")
          Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.3f")
          jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.5f")
          shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.5f")
          RPDE= st.number_input("RPDE", min_value=0.0, max_value=1.0, format="%.5f")
          DFA= st.number_input("DFA", min_value=0.0, max_value=1.0, format="%.5f")
          spread1 = st.number_input("Spread1", format="%.5f")
          spread2 = st.number_input("Spread2", format="%.5f")
          PPE = st.number_input("PPE", min_value=0.0, format="%.5f")

          submitted3 = st.form_submit_button("🔍 PREDICT")
        if submitted3:
          data =pd.DataFrame([{
           "MDVP:Fo(Hz)":Fo,
           "MDVP:Fhi(Hz)":Fhi,
           "MDVP:Flo(Hz)": Flo,
           "MDVP:Jitter(%)":jitter,
           "MDVP:Shimmer(dB)":shimmer_db,
           "RPDE":RPDE,
           "DFA":DFA,
           "spread1":spread1,
           "spread2":spread2,
           "PPE":PPE
        }])

        prokin_id = parkison_brain.predict(data)[0]
        result2 = "NORMAL" if prokin_id == 1 else "DANGER"

        st.success(f"Liver result: {result2}")

       










