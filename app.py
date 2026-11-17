import streamlit as st
import joblib
import pandas as pd

FEATURES = ["pclass", "sex", "age", "fare"]

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("Titanic Survival Prediction")

st.write("Enter passenger details to predict the chance of survival.")

pclass = st.selectbox("Ticket Class", [1, 2, 3])

sex = st.radio("Sex", ["female", "male"], horizontal=True)

age = st.slider("Age", 1, 80, 30)

fare = st.slider("Fare", 0, 500, 30)

if st.button("Predict"):

    try:
        model = load_model()

        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=FEATURES
        )

        chance = model.predict_proba(row)[0][1]

        st.metric("Estimated Chance of Survival", f"{chance:.0%}")

        if chance >= 0.5:
            st.success("The model predicts: Likely to Survive")
        else:
            st.warning("The model predicts: Unlikely to Survive")

    except FileNotFoundError:
        st.error("Model file is missing. Please add model.joblib to the project folder.")

    except (ValueError, TypeError):
        st.error("Invalid input. Please check the passenger details.")

    except Exception:
        st.error("Something went wrong. Please check the model and input values.")

st.caption(
    "This Decision Tree model was trained on the Titanic dataset. "
    "This is an educational project, not a real-world prediction or advice."
)