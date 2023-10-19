import streamlit as st

from views import home,Signal_processing,plots,file,image_processing,Motor_Status,machine_learning

st.set_page_config(layout="wide",page_title="WELCOME TO INDUCTION MOTOR")
menu_items = ["Home","File" ,"Plots", "Signal Processing","Image Processing","Machine Learning","Check Motor Status"]

st.sidebar.header("Dashboard")
choice = st.sidebar.selectbox("", menu_items)
if choice == "Home":
    home.load_view()
elif choice == "File":
    file.load_view()
elif choice == "Plots":
    plots.load_view(file.load_data())
     
elif choice == "Signal Processing":
    Signal_processing.load_view(file.load_data())
elif choice == "Image Processing":
    image_processing.load_view(file.load_data())

elif choice == "Machine Learning":
    machine_learning.load_view(file.load_data())
elif choice == "Check Motor Status":
    Motor_Status.load_view(image_processing.statusload())

    
st.set_option('deprecation.showPyplotGlobalUse', False)



