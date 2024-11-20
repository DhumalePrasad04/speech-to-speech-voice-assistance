import streamlit as st
import pyttsx3
import speech_recognition as sr
import threading
from rag_model import initialize_rag_pipeline, answer_query

# Initialize the RAG pipeline
vectorstore_path = "./vectorbase"  # Directory where vectorstore is persisted
rag_pipeline = initialize_rag_pipeline(vectorstore_path)

# Initialize TTS engine
engine = pyttsx3.init()

# Function to capture and convert speech to text
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)  # Use Google's speech recognition
            st.write(f"Text recognized: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return ""

# Function to handle text-to-speech in a separate thread
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("AI Question-Answering System with Speech-to-Text and Text-to-Speech")

# Input field for user question or record from voice
user_input_option = st.radio("Choose input method", ('Text', 'Voice'))

user_question = ""  # Initialize user_question to an empty string

if user_input_option == 'Voice':
    if st.button("Record Question"):
        # Record and convert speech to text
        user_question = record_audio()
else:
    user_question = st.text_input("Ask a question:")

# Display answer and convert it to speech if a valid question is provided
if user_question:
    with st.spinner('Processing your question...'):
        # Get answer from the RAG pipeline
        answer = answer_query(rag_pipeline, user_question)
        answer = answer.replace("*", "")
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)


        # Use a separate thread to convert answer to speech
        if st.button('Listen to the Answer'):
            threading.Thread(target=speak_text, args=(answer,)).start()
