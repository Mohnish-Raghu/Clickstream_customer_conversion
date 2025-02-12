import streamlit as st
import pandas as pd
import pickle
import numpy as np

label_encoder = pickle.load(open(r"F:\Guvi\Clickstream_customer_conversion\le2_clothing_model.pkl", 'rb'))
classifier = pickle.load(open(r"F:\Guvi\Clickstream_customer_conversion\random_forest_classifier_model.pkl", 'rb'))
regressor = pickle.load(open(r"F:\Guvi\Clickstream_customer_conversion\gradient_boosting_model.pkl", 'rb'))
regressor_scaler = pickle.load(open(r"F:\Guvi\Clickstream_customer_conversion\regression_standard_scaler.pkl", 'rb'))
classifier_scaler = pickle.load(open(r"F:\Guvi\Clickstream_customer_conversion\classification_standard_scaler.pkl", 'rb'))

st.set_page_config(layout="wide")
st.title("Clickstream App")
st.sidebar.header("Options")
    
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
model_type = st.sidebar.radio("Select Model Type", ("Regression", "Classification"))

regression_features = ['page1_main_category', 'page2_clothing_model', 'colour']
classify_features = ['page1_main_category', 'page2_clothing_model', 'colour', 'order', 'price', 'location', 'model_photography']

main_dict = {'Trousers': 1, 'Skirts': 2, 'Blouses': 3, 'Shirts': 4}
Trousers = ['A15', 'A2', 'A39', 'A1', 'A9', 'A7', 'A29', 'A11', 'A12', 'A17',
       'A5', 'A4', 'A22', 'A18', 'A25', 'A27', 'A8', 'A30', 'A6', 'A35',
       'A10', 'A3', 'A33', 'A14', 'A34', 'A20', 'A32', 'A13', 'A21',
       'A28', 'A41', 'A31', 'A24', 'A40', 'A42', 'A37', 'A23', 'A16',
       'A26', 'A43', 'A36', 'A38', 'A19']
Skirts = ['B24', 'B32', 'B3', 'B10', 'B13', 'B19', 'B12', 'B16', 'B4', 'B11',
       'B8', 'B2', 'B15', 'B9', 'B21', 'B31', 'B1', 'B7', 'B14', 'B26',
       'B23', 'B17', 'B20', 'B27', 'B28', 'B25', 'B22', 'B30', 'B6',
       'B33', 'B34', 'B29', 'B5']
Blouses = ['C11', 'C35', 'C14', 'C15', 'C57', 'C40', 'C33', 'C44', 'C9',
       'C50', 'C55', 'C51', 'C21', 'C17', 'C34', 'C43', 'C22', 'C54',
       'C5', 'C12', 'C6', 'C2', 'C46', 'C53', 'C25', 'C29', 'C26', 'C36',
       'C48', 'C10', 'C20', 'C13', 'C30', 'C18', 'C23', 'C1', 'C19',
       'C56', 'C39', 'C59', 'C7', 'C45', 'C37', 'C28', 'C4', 'C27', 'C42',
       'C32', 'C41', 'C16', 'C52', 'C31', 'C3', 'C38', 'C8', 'C24', 'C49',
       'C58', 'C47']
Shirts = ['P48', 'P23', 'P51', 'P33', 'P11', 'P7', 'P82', 'P1', 'P56', 'P62',
       'P12', 'P43', 'P16', 'P4', 'P49', 'P3', 'P6', 'P47', 'P61', 'P17',
       'P15', 'P57', 'P64', 'P20', 'P65', 'P37', 'P19', 'P5', 'P63',
       'P74', 'P69', 'P39', 'P60', 'P40', 'P44', 'P21', 'P46', 'P38',
       'P14', 'P52', 'P26', 'P41', 'P18', 'P68', 'P29', 'P2', 'P34',
       'P36', 'P70', 'P72', 'P77', 'P50', 'P32', 'P78', 'P45', 'P53',
       'P42', 'P58', 'P35', 'P30', 'P8', 'P10', 'P67', 'P59', 'P25', 'P9',
       'P80', 'P75', 'P13', 'P71', 'P73', 'P76', 'P27', 'P55', 'P24',
       'P81', 'P31', 'P66', 'P28']

colors = {
            "Beige": "#F5F5DC",
            "Black": "#000000",
            "Blue": "#0000FF",
            "Brown": "#A52A2A",
            "Burgundy": "#800020",
            "Gray": "#808080",
            "Green": "#008000",
            "Navy": "#000080",
            "Purple": "#800080",
            "Olive": "#808000",
            "Pink": "#FFC0CB",
            "Red": "#FF0000",
            "Violet": "#8A2BE2",
            "White": "#FFFFFF"
        }

def first3():
    page1_main_category = st.sidebar.selectbox("Main Category", ['Trousers', 'Skirts', 'Blouses', 'Shirts'])
        
    page1_main_category = main_dict[page1_main_category]
        
    if page1_main_category == 1:
        page2_clothing_model = st.sidebar.selectbox("Model", Trousers)
    elif page1_main_category == 2:
        page2_clothing_model = st.sidebar.selectbox("Model", Skirts)    
    elif page1_main_category == 3:
        page2_clothing_model = st.sidebar.selectbox("Model", Blouses)
    else:
        page2_clothing_model = st.sidebar.selectbox("Model", Shirts)
        
    page2_clothing_model = label_encoder.transform([page2_clothing_model])
    selected_color = st.sidebar.selectbox("Color", list(colors.keys()))
    hex_code = colors[selected_color]
    st.sidebar.markdown(
        f"<div style='width:100px; height:50px; background-color:{hex_code}; border:1px solid #000'></div>", 
        unsafe_allow_html=True
    )
    colour_dict = {'Beige': 1, 'Black': 2, 'Blue': 3, 'Brown': 4, 'Burgundy': 5, 'Gray': 6, 'Green': 7, 'Navy': 8, 'Purple': 9, 'Olive': 10, 'Pink': 11, 'Red': 12, 'Violet': 13, 'White': 14}
    colour = colour_dict[selected_color]

    return page1_main_category, page2_clothing_model[0], colour

if model_type == "Regression":
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # display df in wide mode
        st.dataframe(df[regression_features + ['price']],  width=1000, height=400)   
        row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
        selected_row = df.loc[[row_index], regression_features]
        selected_row['page2_clothing_model'] = label_encoder.transform(selected_row['page2_clothing_model'])
        selected_row = regressor_scaler.transform(selected_row)
        
    else:
        page1_main_category, page2_clothing_model, colour = first3()
        selected_row = [page1_main_category, page2_clothing_model, colour]
        selected_row = np.array(selected_row).reshape(1, -1)
        selected_row = regressor_scaler.transform(selected_row)
        
    if st.button("Predict"):
            st.subheader("Predicted Price")
            predicted = regressor.predict(selected_row)
            st.write(f"${predicted[0]:.2f}")
        
elif model_type == "Classification":
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df[classify_features + ['price_2']],  width=1000, height=400)        
        row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
        selected_row = df.loc[[row_index], classify_features]
        selected_row['page2_clothing_model'] = label_encoder.transform(selected_row['page2_clothing_model'])
        selected_row = classifier_scaler.transform(selected_row)
        
    else:
        page1_main_category, page2_clothing_model, colour = first3()

        order = st.sidebar.slider("Order", 1, 100)
        price = st.sidebar.slider("Price", 0, 100)

        location = st.sidebar.selectbox("Product Location",['Top Left', 'Top in the Middle', 'Top Right', 'Bottom Left', 'Bottom in the Middle', 'Bottom Right'])
        location_dict = {'Top Left': 1, 'Top in the Middle': 2, 'Top Right': 3, 'Bottom Left': 4, 'Bottom in the Middle': 5, 'Bottom Right': 6}
        location = location_dict[location]

        model_photography = st.sidebar.selectbox("Model Photography",['En Face', 'Profile'])
        model_photography_dict = {'En Face': 1, 'Profile': 2}
        model_photography = model_photography_dict[model_photography]
        
        selected_row = [page1_main_category, page2_clothing_model, colour, order, price, location, model_photography]
        selected_row = np.array(selected_row).reshape(1, -1)
        selected_row = classifier_scaler.transform(selected_row)

    if st.button("Classify"):
        st.subheader("Purchase Classification")
        predicted = classifier.predict(selected_row)
        if predicted[0] == 2:
            st.write("Will Purchase")
        else:
            st.write("Won't Purchase")
