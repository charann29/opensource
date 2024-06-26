import speech_recognition as sr
import pyttsx3

# Create a recognizer object
recognizer = sr.Recognizer()

# Create a microphone object
microphone = sr.Microphone()

# Start listening for audio
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Say something...")
    audio = recognizer.listen(source)

# Recognize the speech
try:
    # Get the recognized text
    text = recognizer.recognize_google(audio)
    print("You said:", text)

    # Save the recognized text to a file
    with open("recognized_text.txt", "w") as file:
        file.write(text)

    # Create a Text-to-Speech engine
    engine = pyttsx3.init()

    # Set the voice
    engine.setProperty('voice', 'en-US')

    # Set the rate
    engine.setProperty('rate', 150)

    # Speak the text
    engine.say(text)

    # Stop the engine
    engine.runAndWait()

except sr.UnknownValueError:
    print('Sorry, I could not understand what you said.')
except sr.RequestError:
    print('Sorry, there was an error with the speech recognition service.')
