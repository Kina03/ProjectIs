import streamlit as st 

def page1():
    st.title("🍔 Predict food ordering trends and food delivery times 🍕")
    st.header("Data Processing 📈📑")
    st.write("📄Dataset 1: Online Food Dataset")
    st.write("The dataset of Online food it is contains information collected from an online food platform over a period of time. It attributes related to Age, Monthly Income, Feedback etc.")
    
    att = '''**Attributes :**jdkk

    Age: Age of the customer.
    Gender: Gender of the customer.
    Marital Status: Marital status of the customer.
    Occupation: Occupation of the customer.
    Monthly Income: Monthly income of the customer.
    Educational Qualifications: Educational qualifications of the customer.
    Family Size: Number of individuals in the customer's family.
    Location Information:

    Latitude: Latitude of the customer's location.
    Longitude: Longitude of the customer's location.
    Pin Code: Pin code of the customer's location.
    Order Details:

    Output: Current status of the order (e.g., pending, confirmed, delivered).
Feedback: Feedback provided by the customer after receiving the order.
    '''
    st.markdown(att)


    st.write("🔍 Source : https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset/data")


def page2():
    st.title("Second page")

pg = st.navigation([
    st.Page(page1, title="Home", icon="🏚️"),
    st.Page(page2, title="Second page", icon=":material/favorite:"),
])
pg.run()