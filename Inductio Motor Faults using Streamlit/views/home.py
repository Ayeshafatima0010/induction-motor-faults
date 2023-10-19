import streamlit as st

def load_view():
    # Set page title


    # Set page layout
    st.markdown(
        """
        <style>
        .css-z5fcl4{
        width: 100%;
    padding: 0rem 3rem 0rem;
    min-width: auto;
    max-width: initial;
    .css-1kyxreq {
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    justify-content: center;
}
       


        }}

        .container {
            max-width: 900px;
            padding: 20px;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: #333;
            color: white;
            font-size: 14px;
            margin-top: 185px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Add header
    st.write("""
        # AI-Based Faults Prediction of Induction Motor
    """)

   
    # Add image
    st.image("motor.jpg",width=500)

    # Add description
    st.write("""
        Our project aims to use AI techniques to predict faults in induction motors. By analyzing data from sensors placed on the motor, we can detect early signs of faults before they become serious problems. This can help prevent costly downtime and maintenance, as well as improve the overall efficiency and reliability of industrial processes.
    """)
    

# Add footer
    st.markdown(
        """
        <div class="footer">
            © 2023 AI Faults Prediction Project
        </div>
        """,
        unsafe_allow_html=True
    )

    

    # Add call to action




