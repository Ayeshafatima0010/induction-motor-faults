import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
    
def load_view(dt):
    st.title('Signal Processing')
    sub_menu = ["Select Technique","Time Domain","FFT","FFTDB"]
    sub_choice = st.selectbox("Please select Siganal Processing technique", sub_menu)

    if sub_choice=="FFTDB":
        FFTDBCH = ["Select Parameter","Current", "Vibration","Bearing_vib"]
        FFTDBCH_subchoice = st.selectbox("Please select Attributes", FFTDBCH)

        if FFTDBCH_subchoice=="Current":

            data = dt.iloc[:830,0]

            fs = 55611

            # Center the data around the mean
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)

            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered), 1/fs)

            # Convert magnitude to dB scale
            magnitude_db = 20 * np.log10(2*np.abs(x)/float(len(x)))

            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(abs(freq), magnitude_db)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            # ax.set_xlim(0,1000)
            # ax.set_ylim(-100,50)
            # Display the plot in Streamlit
            st.pyplot(fig)

        elif FFTDBCH_subchoice=="Vibration":
            data = dt.iloc[:255,1]
            fs = 8326
            # Center the data around the mean
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)

            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered), 1/fs)

            # Convert magnitude to dB scale
            magnitude_db = 20 * np.log10(2*np.abs(x)/float(len(x)))

            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(abs(freq), magnitude_db)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            # ax.set_xlim(0,1000)
            # ax.set_ylim(-100,50)
            # Display the plot in Streamlit
            st.pyplot(fig)
        elif FFTDBCH_subchoice=="Bearing_vib":
       
            data = dt.iloc[:2000,2]
            # Center the data around the mean
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)

            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered))

            # Convert magnitude to dB scale
            magnitude_db = 20 * np.log10(2*np.abs(x)/float(len(x)))

            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(abs(freq), magnitude_db)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            # Display the plot in Streamlit
            st.pyplot(fig)
      
    

    elif sub_choice=="FFT":
        FFTchart = ["Select Parameter","Current", "Vibration","Bearing_vib"]
        FFTchart_subchoice = st.selectbox("Please select Attributes", FFTchart)

        if FFTchart_subchoice=="Current":

            data = dt.iloc[:830,0]
            fs = 55611
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered), 1/fs)
            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(abs(freq), 2*np.abs(x)/len(x))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            # ax.set_xlim(0,100)
            # Display the plot in Streamlit
            st.pyplot(fig)

        elif FFTchart_subchoice=="Vibration":
            data = dt.iloc[:255,1]
            fs = 8326
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered), 1/fs)
            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(abs(freq), 2*np.abs(x)/len(x))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            # ax.set_xlim(0,100)
            # Display the plot in Streamlit
            st.pyplot(fig)

        elif FFTchart_subchoice=="Bearing_vib":
         
            data = dt.iloc[:2000,2]
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            # Compute the FFT
            x = np.fft.fft(current1_centered)
            freq = np.fft.fftfreq(len(current1_centered))
            # Plot frequency-domain spectrum
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(abs(freq), 2*np.abs(x)/len(x))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            # ax.set_xlim(0,100)
            # Display the plot in Streamlit
            st.pyplot(fig)
    


    elif sub_choice=="Time Domain":
        Timedchart = ["Select Parameter","Current", "Vibration","Bearing_vib"]
        Timedchart_subchoice = st.selectbox("Please select Attributes", Timedchart)
        if Timedchart_subchoice=="Current":
            data = dt.iloc[:830,0]
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(current1_centered)
            st.pyplot(fig)
        elif Timedchart_subchoice=="Vibration":
            data = dt.iloc[:255,1]
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(current1_centered)
            st.pyplot(fig)
        elif Timedchart_subchoice=="Bearing_vib":
            data = dt.iloc[:2000,2]
            current1_array = np.array(data)
            mean_current1 = np.mean(current1_array)
            current1_centered = np.subtract(current1_array, mean_current1)
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.plot(current1_centered)
            st.pyplot(fig)



            


















    
    
   
        










    
