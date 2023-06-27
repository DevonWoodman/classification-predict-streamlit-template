"""

    Simple Streamlit webserver application for serving developed classification
    models.
    
    Author: WebTec Solutions 
    Senior Developer: Devon Woodman
    Data
    
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html
import base64
from streamlit_option_menu import option_menu

# Data dependencies
import pandas as pd

# Preprocessing
import re
import string
import nltk
from string import punctuation
import emoji
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import TweetTokenizer 
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Analysis dependancies
import matplotlib.pyplot as plt

df_office_loc = pd.DataFrame(
    [[-33.924487, 18.417017]],
    columns=['lat', 'lon'])

def text_processing(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) # Remove URLs
    text = emoji.demojize(text, delimiters=("", "")) # Emoji to Text 
    text = re.sub("rt[\s]", "", text) # Remove retweet 'rt'
    text = re.sub('[^a-z]', ' ',text) # Removing non-alphabets
    
    hashtags = re.findall(r"#\w+", text) # Extract hashtags
    extracted_hashtags = [tag.strip("#") for tag in hashtags]
    text = re.sub(r"#\w+", '',text)
    
    mentions = re.findall(r"@\w+", text) # Extract mentions using regex pattern matching
    extracted_mentions = [tag.strip("@") for tag in mentions]
    text = re.sub(r"@\w+", '',text)
    
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    lemmatizer = WordNetLemmatizer()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            lem = lemmatizer.lemmatize(i)
            y.append(lem)
            
    return " ".join(y)

def remove_urls(text):   
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # Regular expression pattern to match URLs
    text_without_urls = re.sub(url_pattern, '', text) # Remove URLs from the text

    return text_without_urls

def cleaning(text):   
    text = emoji.demojize(text, delimiters=("", "")) # Emoji to Text
    text = re.sub("rt[\s]", "", text) # Remove retweet 'rt'

    return text

def convert_sentiment(num):
    sentiment = {
        0: 'neutral',
        1: 'positive',
        -1: 'negative',
        2: 'news'
    }
    
    return sentiment.get(num, 'no sentiment')

def explain_sentiment(num):
    sentiment = {
        0: 'Neutral sentiment refers to a lack of strong positive or negative feelings or opinions towards a particular subject. It indicates a state of indifference, objectivity, or a balanced perspective. When analyzing text or data for sentiment analysis, neutral sentiment is often considered as the absence of any explicit positive or negative sentiment.<br><br>Neutral sentiment can occur for various reasons:<br>  1. Lack of emotional attachment: Some topics or situations may not evoke strong emotions in individuals, leading to a neutral stance. For example, discussing mundane or neutral subjects like weather, facts, or simple descriptions may elicit a neutral sentiment.<br>  2. Objective statements: Neutral sentiment can arise when conveying factual information or stating something in an unbiased, impartial manner. Objective statements aim to present information without expressing personal opinions or emotions.<br>  3. Ambiguity or uncertainty: In situations where the information is ambiguous, vague, or unclear, people may adopt a neutral sentiment until further clarification is obtained. This allows individuals to suspend judgment and avoid making strong positive or negative assumptions.<br>  4. Mixed opinions or conflicting emotions: Sometimes, individuals may have a combination of positive and negative feelings towards a subject, resulting in an overall neutral sentiment. This can occur when considering various aspects of a topic and finding both favorable and unfavorable elements.',
        1: 'Positive sentiment towards believing in climate change refers to a favorable or optimistic attitude regarding the acceptance and acknowledgment of the reality and significance of climate change. It reflects a positive perception and acceptance of scientific evidence indicating that human activities contribute to global warming and environmental degradation.<br><br>Here are some key points to understand positive sentiment towards believing in climate change:<br><br>  1. Awareness and understanding: Positive sentiment towards climate change involves being aware of the issue and having a solid understanding of the scientific consensus that supports the existence and impact of climate change. It reflects a recognition of the evidence and data collected by scientists studying the Earth\'s climate system.<br>  2. Trust in scientific research: Positive sentiment towards believing in climate change often stems from placing trust in the scientific community and its expertise. It involves acknowledging the extensive research, studies, and models that have been conducted to understand the causes and consequences of climate change.<br>  3. Concern for the environment: Positive sentiment towards climate change is often driven by a genuine concern for the environment and a desire to mitigate its negative impacts. It reflects a belief that taking action to address climate change is necessary to protect ecosystems, biodiversity, and the well-being of current and future generations.<br>  4. Support for mitigation and adaptation measures: Positive sentiment towards believing in climate change is accompanied by support for measures aimed at mitigating and adapting to its effects. This can include endorsing policies and actions that reduce greenhouse gas emissions, promote renewable energy, conserve resources, and foster sustainable practices.<br>  5. Collaboration and collective action: Positive sentiment towards climate change often involves recognizing the need for collective action and international cooperation to address the global challenge. It reflects a belief that individuals, communities, governments, and businesses all have a role to play in finding solutions and implementing sustainable practices.',
        -1: 'Negative sentiment towards believing in climate change refers to a skeptical or pessimistic attitude regarding the acceptance and acknowledgment of the reality and significance of climate change. It reflects a lack of belief or doubt in the scientific consensus that human activities contribute to global warming and environmental degradation.<br><br>Here are some key points to understand negative sentiment towards believing in climate change:<br><br>  1. Skepticism and denial: Negative sentiment towards climate change often arises from skepticism or denial of the scientific evidence supporting the existence and impact of climate change. Skeptics may question the accuracy of climate models, dispute the reliability of climate data, or challenge the attribution of observed changes to human activities.<br>  2. Alternative explanations: Negative sentiment can stem from the belief in alternative explanations for observed climate patterns and variability. Some individuals attribute climate change to natural processes or argue that the Earth\'s climate has always undergone fluctuations and that human activities play a negligible role.<br>  3. Political or ideological influences: Negative sentiment towards believing in climate change can be influenced by political or ideological factors. Some individuals align their opinions based on political affiliations, economic interests, or a skepticism towards government intervention or regulations associated with climate change mitigation.<br>  4. Misinformation and conspiracy theories: Negative sentiment may be reinforced by the spread of misinformation or conspiracy theories regarding climate change. Misinformation can create confusion, sow doubt, or present misleading information that challenges the scientific consensus, leading to a negative sentiment among certain individuals.<br>  5. Perception of economic implications: Negative sentiment can be driven by concerns about the economic impacts of addressing climate change. Some individuals may fear that adopting measures to mitigate climate change will have adverse effects on industries, jobs, or economic growth.',
        2: ''
    }
    
    return sentiment.get(num, 'no sentiment')

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

font_css = """
            <style>
            button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
              font-size: 24px;
            }
            
            #tabs-bui2-tab-0 {
              background: transparent
            }
            
            #tabs-bui2-tab-1 {
              background: transparent
            }
            
            #tabs-bui2-tab-2 {
              background: transparent
            }
            </style>
            """

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
#    def add_bg_from_url():
#        st.markdown(
#        f"""
#            <style>
#                .stApp {{
#                    background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
#                    background-attachment: fixed;
#                    background-size: cover
#                }}
#            </style>
#        """, unsafe_allow_html=True)

#    add_bg_from_url() 

    #Style aplication background with image from local machine
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
            <style>
                .stApp {{
                    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                    background-size: cover
                }}
            </style>
        """,
        unsafe_allow_html=True
        )
        
    add_bg_from_local('logos/cool-background.png')
    
    
    #Define navigation options, icons and styles
    menu_options = ["Home", "Sentiment", "Analysis", "About Us", "Feedback"]
    menu_icons = ['bi-house-fill', 'bi-twitter', 'bi-graph-up', 'bi-info-square-fill', 'bi-chat-heart-fill']
    nav_bar_style = {
        "container": {"padding": "0!important"},
        "nav-item": {"border-radius": "5px"},
        "nav-link": {"margin":"0px", "--hover-color": "#bebebe", "border-radius": "5px"},
        "nav-link-selected": {"background-color": "#007fe0"}
    }
    
    nav_menu_style = {
        "container": {"padding": "0!important"},
        "nav-link": {"margin":"0px", "--hover-color": "#bebebe"},
        "nav-link-selected": {"background-color": "#007fe0","opacity":0.8}
    }
    
#     #Change navigation menu when navigation bar item changes
#     if st.session_state.get('menu_1', False):
#         st.session_state['menu_option'] = menu_options.index(st.session_state.get('menu_1',1))
#         manual_select = st.session_state['menu_option']
#     else:
#         manual_select = None
        
#     if st.session_state.get('menu_2', False):
#         st.session_state['menu_option'] = menu_options.index(st.session_state.get('menu_2',1))
#         manual_select = st.session_state['menu_option']
#     else:
#         manual_select = None
    
    #Sidebar definition and sidebar contents
    with st.sidebar:
        st.image("logos/EcoPulse_Logo_2-removebg.png", width=300) #EcoPulse logo as title   
        
        #Define sidebar menu import from streamlit_option_menu
        selection = option_menu("Main Menu", menu_options, icons=menu_icons, menu_icon="cast", styles=nav_menu_style, default_index=0)
    
    #Define app content common to all pages
    #EcoPulse logo as title
    st.image("logos/EcoPulse_Logo_2-removebg.png")
        
#     #Define sidebar menu import from streamlit_option_menu
#     selection = option_menu(None, menu_options, icons=menu_icons, menu_icon="cast", default_index=0, orientation="horizontal", styles=nav_bar_style,  manual_select=manual_select, key='menu_2')
    
    st.markdown("<div style='background-color: rgba(90, 90, 90, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align: center><h5 style='padding: 0; color: rgba(255, 255, 255, 1)'>{}</h5><span class='bi-house-fill'></span></div>".format(selection), unsafe_allow_html=True)

    # Building out the "Home" page
    if selection == "Home":
        
        st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>EcoPulse is a revolutionary application designed to understand the sentiments surrounding climate change through the analysis of tweet data. With our cutting-edge technology, we provide useful insights into the public's perception of this critical global issue.</p></div>", unsafe_allow_html=True)
        
        with st.expander("Key Features"):
            st.subheader("Sentiment Analysis:")
            st.markdown("EcoPulse utilizes advanced natural language processing algorithms to analyze tweets related to climate change. It accurately detects sentiments, including positive, negative, and neutral tones, allowing you to gauge the overall sentiment trends.")
            st.divider()
            st.subheader("Visualization and Reports:")
            st.markdown("EcoPulse transforms complex sentiment data into clear and visually appealing charts and reports. Our app presents sentiment trends, distributions, and popular keywords associated with climate change. These visualizations empower you to comprehend sentiment patterns at a glance and make data-driven decisions effortlessly.")

        st.subheader("Latest News:")
        components.html(
        """
        <!DOCTYPE html>
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                * {box-sizing: border-box;}
                body {font-family: Verdana, sans-serif;}
                .mySlides {display: none;}
                img {vertical-align: middle;}

                /* Slideshow container */
                .slideshow-container {
                    max-width: 1000px;
                    position: relative;
                    margin: auto;
                }

                /* Caption text */
                .text {
                    color: #f2f2f2;
                    font-size: 15px;
                    padding: 8px 12px;
                    position: absolute;
                    bottom: 8px;
                    width: 100%;
                    text-align: center;
                }

                /* Number text (1/3 etc) */
                .numbertext {
                    color: #f2f2f2;
                    font-size: 12px;
                    padding: 8px 12px;
                    position: absolute;
                    top: 0;
                }

                /* The dots/bullets/indicators */
                .dot {
                    height: 15px;
                    width: 15px;
                    margin: 0 2px;
                    background-color: #bbb;
                    border-radius: 50%;
                    display: inline-block;
                    transition: background-color 0.6s ease;
                }

                .active {
                    background-color: #717171;
                }

                /* Fading animation */
                .fade {
                    animation-name: fade;
                    animation-duration: 1.5s;
                }

                @keyframes fade {
                    from {opacity: .4} 
                    to {opacity: 1}
                }

                /* On smaller screens, decrease text size */
                @media only screen and (max-width: 300px) {
                    .text {font-size: 11px}
                }
                </style>
            </head>
            <body>
                <div class="slideshow-container">

                    <div class="mySlides fade">
                        <div class="numbertext">1 / 3</div>
                        <img src="https://unsplash.com/photos/GJ8ZQV7eGmU/download?force=true&w=1920" style="width:100%">
                        <div class="text">Caption Text</div>
                    </div>

                    <div class="mySlides fade">
                        <div class="numbertext">2 / 3</div>
                        <img src="https://unsplash.com/photos/eHlVZcSrjfg/download?force=true&w=1920" style="width:100%">
                        <div class="text">Caption Two</div>
                    </div>

                    <div class="mySlides fade">
                        <div class="numbertext">3 / 3</div>
                        <img src="https://unsplash.com/photos/zVhYcSjd7-Q/download?force=true&w=1920" style="width:100%">
                        <div class="text">Caption Three</div>
                    </div>

                </div>
                <br>

                <div style="text-align:center">
                    <span class="dot"></span> 
                    <span class="dot"></span> 
                    <span class="dot"></span> 
                </div>

                <script>
                    let slideIndex = 0;
                    showSlides();

                    function showSlides() {
                        let i;
                        let slides = document.getElementsByClassName("mySlides");
                        let dots = document.getElementsByClassName("dot");
                        for (i = 0; i < slides.length; i++) {
                            slides[i].style.display = "none";  
                        }
                        slideIndex++;
                        if (slideIndex > slides.length) {slideIndex = 1}    
                        for (i = 0; i < dots.length; i++) {
                            dots[i].className = dots[i].className.replace(" active", "");
                        }
                        slides[slideIndex-1].style.display = "block";  
                        dots[slideIndex-1].className += " active";
                        setTimeout(showSlides, 10000); // Change image every 10 seconds
                    }
                </script>

            </body>
        </html> 

        """,
        height=600,
    )
    
    # Building out the "Information" page
    if selection == "Analysis":
        st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px; text-align:justify'><p><h6>Data Analyser:</h6>Welcome to our data analyzer! In the realm of data-driven decision making, understanding and analyzing datasets is paramount. With our data analyzer, we aim to simplify the process of exploring and gaining insights from your data. Simply upload your CSV file contianing tweet messages and sentiments and let the application do the work.</p></div>", unsafe_allow_html=True)
        
        
        analysis_data = st.file_uploader('Upload CSV File', type=['csv'], accept_multiple_files=False, label_visibility="visible")
        if st.button("Classify"):
            st.markdown('Analysis begin')
            
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

        

    # Building out the predication page
    if selection == "Sentiment":
        st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px; text-align:justify'><p><h6>Sentiment Predictor:</h6>Welcome to our advanced sentiment predictor! This powerful tool is designed to determine the sentiment of any given text string and provide valuable insights into the emotional tone and polarity of the content.</p></div>", unsafe_allow_html=True)
        
        sentiment_message = "Please submit a message before a sentiment can be determined..."
        
        col1, col2 = st.columns([0.3, 0.7])

        with col1:
            # Model selector. Model to be used for sentiment prediction
            classifier_opt = st.selectbox(
                'Please select a model:',
                ('Stocastic Gradient Descent', 'Logistic Regression', 'Support Vector', 'Naive Bayes', 'Nearest Neighbours' ))
            bulk_sentiment = st.checkbox('Bulk Sentiment:', value=False, label_visibility="visible")
            
            col3, col4, col5 = st.columns([0.1, 0.8, 0.1])
            with col4:
                st.image("logos/black-306558_1920.png", width=150)  

        with col2:
            if bulk_sentiment == False:
                # Creating a text box for user input
                tweet_text = st.text_area("Enter Message:","Type Here")

                if st.button("Classify"):
                    # Clean text
                    clean_text = text_processing(tweet_text)
                    clean_text = remove_urls(tweet_text)
                    clean_text = cleaning(tweet_text)

                    # Transforming user input with vectorizer
                    vect_text = tweet_cv.transform([clean_text]).toarray()

                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    if classifier_opt == 'Logistic Regression':
                        predictor = joblib.load(open(os.path.join("resources/Logistic_Regression_model.pkl"),"rb"))
                    if classifier_opt == 'Support Vector':
                        predictor = joblib.load(open(os.path.join("resources/Linear_SVM_model.pkl"),"rb"))
                    if classifier_opt == 'Naive Bayes':
                        predictor = joblib.load(open(os.path.join("resources/Naive_Bayes_model.pkl"),"rb"))
                    if classifier_opt == 'Nearest Neighbours':
                        predictor = joblib.load(open(os.path.join("resources/Nearest_Neighbors_model.pkl"),"rb"))
                    if classifier_opt == 'Stocastic Gradient Descent':
                        predictor = joblib.load(open(os.path.join("resources/Stocastic_Gradient_Descent_model.pkl"),"rb"))

                    prediction = predictor.predict(vect_text)

                    if prediction == 2:
                        sentiment_message = 'The message appears to be a News item!'
                    else:
                        sentiment_message = 'The message conveys {} sentiment!'.format(convert_sentiment(prediction[0]))
                        
                st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px; text-align:justify'><p>{}<br><br>{}</p></div>".format(sentiment_message, explain_sentiment(prediction[0])), unsafe_allow_html=True)
                    
            else:
                n = 5
                # Creating a text box for user input
                uploaded_file = st.file_uploader('Upload CSV File', type=['csv'], accept_multiple_files=False, label_visibility="visible")
                if uploaded_file is not None:
                    tweet_data = pd.read_csv(uploaded_file)

                if st.button("Classify"):
                    clean_text = tweet_data['message'].apply(text_processing)
                    clean_text = clean_text.apply(remove_urls)
                    clean_text = clean_text.apply(cleaning)
                    
                    vect_text = tweet_cv.transform(clean_text[:n]).toarray()
                    
                    

                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    if classifier_opt == 'Logistic Regression':
                        predictor = joblib.load(open(os.path.join("resources/Logistic_Regression_model.pkl"),"rb"))
                    if classifier_opt == 'Support Vector':
                        predictor = joblib.load(open(os.path.join("resources/Nearest_Neighbors_model.pkl"),"rb"))
                    if classifier_opt == 'Naive Bayes':
                        predictor = joblib.load(open(os.path.join("resources/Naive_Bayes_model.pkl"),"rb"))
                    if classifier_opt == 'Nearest Neighbours':
                        predictor = joblib.load(open(os.path.join("resources/Nearest_Neighbors_model.pkl"),"rb"))
                    if classifier_opt == 'Stocastic Gradient Descent':
                        predictor = joblib.load(open(os.path.join("resources/Stocastic_Gradient_Descent_model.pkl"),"rb"))

                    prediction = predictor.predict(vect_text)
                    
#                     st.write(prediction)
                    
                    df_sentiment = pd.DataFrame(prediction, columns = ['sentiment'])
                    test_new = pd.DataFrame({"tweetid": tweet_data.copy()['tweetid'][:n].reset_index(drop = True)})
                    sub_file = test_new.join(df_sentiment)
                    sub_file['sentiment'] = df_sentiment.values
                    sub_file['message'] = tweet_data['message'][:n]
                    st.write(sub_file)

        
    # Building out the about us page
    if selection == "About Us":
        
        tab1, tab2, tab3 = st.tabs(["WebTec Solutions", "Testimonials", "Contact Us"])

        with tab1:
           # Creates a main title and subheader on your page    
            st.title("WebTec Solutions")
            st.caption("Your Vision, Our Tech")
            st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>At <b>WebTec Solutions</b>, we are <b>passionate</b> about leveraging the power of data to drive intelligent decision-making and unlock <b>business success</b>. As a leading provider of data science solutions, we combine <b>cutting-edge</b> technologies with our deep expertise to help organizations harness the <b>full potential</b> of their data.<br><br>Founded in 2019, our company has quickly emerged as a trusted partner for businesses across various industries, revolutionizing the way they analyze, interpret, and utilize data. We understand that data is a valuable asset, and we <b>empower our clients</b> with the tools and insights needed to transform raw information into <b>actionable intelligence</b>.<br><br>Our team comprises highly skilled data scientists, machine learning experts, and domain specialists who possess a <b>wealth of knowledge</b> and <b>experience</b> in their respective fields. They are at the forefront of the latest advancements in data science, constantly pushing the boundaries of what is possible.</p></div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("Words of Appreciation: Testimonials that Inspire Us")
            st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>At WebTec Solutions, we take immense pride in the positive impact EcoPulse has on the lives of our users. In this dedicated testimonial section, we present the voices of those who have experienced the transformative power of our app firsthand. Their stories reflect the countless hours of hard work and dedication we have poured into creating a seamless user experience. Join us as we celebrate their achievements and let their words inspire you to embark on your own journey of success with our application.</p></div>", unsafe_allow_html=True)
            
            col6, col7 = st.columns(2)
            with col6:
                st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>\"EcoPulse has completely transformed the way I analyze and interpret climate change discussions on social media. It saves me countless hours of manually sifting through tweets and trying to decipher the sentiment behind them. With just a few clicks, I can instantly obtain accurate predictions of sentiment, ranging from positive to negative, related to climate change topics. The app's user-friendly interface and intuitive design make it a breeze to navigate and utilize. Whether you're a researcher, journalist, or someone interested in understanding public opinion, this app is a game-changer!\"<br> ~Ellen Dusk<br><br><br></p></div>", unsafe_allow_html=True)
            with col7:    
                st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>\"As a business owner, I recognize the importance of staying informed about the public's perception of climate change. EcoPulse has become an essential tool in my decision-making process. By leveraging its predictive capabilities, I can swiftly identify shifts in sentiment and anticipate potential consumer behavior changes. This app has empowered me to align my marketing strategies with prevailing public sentiments, enabling me to engage with my audience more effectively and position my brand as environmentally conscious. I appreciate the app's accuracy, timeliness, and the valuable insights it provides. A must-have for any business concerned about climate change!\"<br> ~ Jeph Boris</p></div>", unsafe_allow_html=True)

        with tab3:
            st.subheader("Contact Us:")
            st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>Thank you for your interest in reaching out to us. If you have any questions, concerns, or need assistance, please don't hesitate to contact us. Our dedicated team is here to provide you with prompt and personalized support. You can contact us directly through the provided phone number or email address. We appreciate your time and look forward to hearing from you. Your satisfaction is our priority, and we are committed to ensuring a seamless experience for you.</p></div>", unsafe_allow_html=True)
        
        
            col8, col9 = st.columns(2)
        
            with col8:
                st.map(df_office_loc)
                
            with col9:
                st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><h5 style='padding: 0'>Contact Details:</h5><br><b>Office phone:</b>&nbsp(+27) 04 626 7000<br><b>Alternative phone:</b>&nbsp(+27) 04 626 7001<br><b>Email:</b>&nbspwebtecsolutions@gmail.com</div>", unsafe_allow_html=True)
                st.image("resources/imgs/WebTec_Logo.png") 

        st.markdown(font_css, unsafe_allow_html=True) 
        
        
        
    # Building out the about us page
    if selection == "Feedback":
        st.markdown("<div style='background-color: rgba(246, 246, 246, 0.4); box-shadow: 2px 2px; padding: 20px; margin: 0px 0px 25px 0px; border-radius: 10px'; text-align:justify><p>Dear User,<br><br>We want to take a moment to express our sincere gratitude for choosing our app and being a valued member of our community. Your support and engagement mean the world to us, and we appreciate your trust in our product.<br><br>We're constantly striving to provide the best possible experience for our users, and your feedback is crucial in helping us achieve that goal. We would be incredibly grateful if you could take a few minutes to share your thoughts and leave a review on your experience with our app.<br><br>Your review not only helps us understand what we're doing well but also provides valuable insights into areas where we can improve. We value your perspective and want to ensure that we meet and exceed your expectations.<br><br>If you have any suggestions or recommendations for app improvements, we would love to hear them. Whether it's a feature you would like to see implemented or an enhancement that could enhance your overall experience, your ideas are vital in shaping the future of our app.<br><br>To leave a review, simply leave a comment in the designated text area, before clicking submit. We genuinely appreciate your time and effort in sharing your thoughts, as it helps us in our mission to create a better app for everyone.<br><br>Once again, thank you for being a part of our app community and for your ongoing support. We look forward to receiving your feedback and working towards making our app even more amazing.<br><br>Best regards,<br><br>WebTec Solutions</p></div>", unsafe_allow_html=True)
        
        # Creating a text box for user input
        feedback_text = st.text_area("User Feedback:","Type Here")

        if st.button("Submit Feedback"):
            st.markdown("Feedback Submitted...")
            st.markdown("Thank you for using the App!")
            
    st.image("resources/imgs/trees-6863878_1920.png")  

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
