import nltk
from nltk.chat.util import Chat, reflections

# Define patterns for the chatbot
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hey there!', 'Hi!']),
    (r'how are you?', ['I am good, thank you!', 'Feeling great, thanks for asking.']),
    (r'what is your name?', ['You can call me ChatBot.', 'I am known as ChatBot.']),
    (r'quit', ['Bye! Take care.', 'Goodbye, have a great day!']),
]

# Create a chatbot using the defined patterns
chatbot = Chat(patterns, reflections)

def main():
    print("Welcome to the ChatBot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        response = chatbot.respond(user_input)
        print("ChatBot:", response)
        if user_input.lower() == 'quit':
            break

if __name__ == "__main__":
    main()
