!pip install speechrecognition pydub transformers torch
!apt-get install ffmpeg
import speech_recognition as sr
from pydub import AudioSegment
import torch
from transformers import pipeline
def convert_audio_to_text(audio_path):
    # Load your audio file
    audio = AudioSegment.from_file("harvard.wav")
    
    # Export the audio as a .wav file for compatibility with the recognizer
    audio.export("converted.wav", format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile("converted.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text
def analyze_sentiment(text):
    # Load the sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return result
def process_audio_for_sentiment_analysis(audio_path):
    text = convert_audio_to_text(audio_path)
    sentiment = analyze_sentiment(text)
    return sentiment
audio_path = "harvard.wav"  # Replace with your audio file path
sentiment_result = process_audio_for_sentiment_analysis(audio_path)
print(sentiment_result)
