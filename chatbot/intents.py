GREETING = 'greeting'
GOODBYE = 'goodbye'
THANKS = 'thanks'
HOURS = 'hours'
MOPEDS = 'mopeds'
PAYMENTS = 'payments'
OPENTODAY = 'opentoday'
RENTAL = 'rental'
TODAY = 'today'

INTENT = {
    GREETING: {
         "patterns": [
             "Hi",
             "How are you",
             "Is anyone there?",
             "Hello",
             "Good day"
         ],
         "responses": [
             "Hello, thanks for visiting",
             "Good to see you again",
             "Hi there, how can I help?"
         ],
         "context_set": ""
    },
    GOODBYE: {
         "patterns": [
             "Bye",
             "See you later",
             "Goodbye"
         ],
         "responses": [
             "See you later, thanks for visiting",
             "Have a nice day",
             "Bye! Come back again soon."
         ]
    },
    THANKS: {
         "patterns": [
             "Thanks",
             "Thank you",
             "That's helpful"
         ],
         "responses": [
             "Happy to help!",
             "Any time!",
             "My pleasure"
         ]
    },
    HOURS: {
         "patterns": [
             "What hours are you open?",
             "What are your hours?",
             "When are you open?"
         ],
         "responses": [
             "We're open every day 9am-9pm",
             "Our hours are 9am-9pm every day"
         ]
    },
    MOPEDS: {
         "patterns": [
             "Which mopeds do you have?",
             "What kinds of mopeds are there?",
             "What do you rent?"
         ],
         "responses": [
             "We rent Yamaha, Piaggio and Vespa mopeds",
             "We have Piaggio, Vespa and Yamaha mopeds"
         ]
    },
    PAYMENTS: {
         "patterns": [
             "Do you take credit cards?",
             "Do you accept Mastercard?",
             "Are you cash only?"
         ],
         "responses": [
             "We accept VISA, Mastercard and AMEX",
             "We accept most major credit cards"
         ]
    },
    OPENTODAY: {
         "patterns": [
             "Are you open today?",
             "When do you open today?",
             "What are your hours today?"
         ],
         "responses": [
             "We're open every day from 9am-9pm",
             "Our hours are 9am-9pm every day"
         ]
    },
    RENTAL: {
         "patterns": [
             "Can we rent a moped?",
             "I'd like to rent a moped",
             "How does this work?"
         ],
         "responses": [
             "Are you looking to rent today or later this week?"
         ],
         "context_set": "rentalday"
    },
    TODAY: {
         "patterns": [
             "today"
         ],
         "responses": [
             "For rentals today please call 1-800-MYMOPED",
             "Same-day rentals please call 1-800-MYMOPED"
         ],
         "context_filter": "rentalday"
    }
}
