# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
import requests
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import base64
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
API_KEY = "WR9cz4JUfWoFJhal7ihYnv7HLN2TFORzlPIWinmuPrT1"
PROJECT_ID = "3c47116b-7ce6-484f-b416-1af22b65b9bf"

# Nastav credentials pro TTS
TTS_API_KEY = "4q7t2JKH8r0f-8XkvnV83VrIVEHnHYRxU8MQQdOjkZF1"
TTS_URL = "https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/4f032f09-c30f-42e7-9405-ee7e3c8feb3c"

# Nastav credentials pro STT
STT_API_KEY = "ZdD_ghR1TQekxwzLgx4DHOIZqrnyzO6Mp1FPRpix-IyZ"
STT_URL = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/4f4e6e67-06b4-4e45-8070-db38744f6d87"


# Inicializuj TTS service
tts_authenticator = IAMAuthenticator(TTS_API_KEY)
text_to_speech = TextToSpeechV1(authenticator=tts_authenticator)
text_to_speech.set_service_url(TTS_URL)

# Inicializuj STT service
stt_authenticator = IAMAuthenticator(STT_API_KEY)
speech_to_text_service = SpeechToTextV1(authenticator=stt_authenticator)
speech_to_text_service.set_service_url(STT_URL)

# Define the credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
   "apikey": API_KEY
}

# Specify model_id that will be used for inferencing
# model_id = "meta-llama/llama-2-70b-chat"
model_id = "mistralai/mistral-large"


# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024,
}

# Define the LLM
model = Model(
    model_id=model_id, params=parameters, credentials=credentials, project_id=PROJECT_ID
)

def speech_to_text(audio_binary):
    """
    Převede řeč na text pomocí IBM Watson STT API
    """
    try:
        # Zavolej Watson STT API
        response = speech_to_text_service.recognize(
            audio=audio_binary,
            content_type='audio/wav',  # nebo 'audio/mp3', 'audio/flac'
            model='en-US_BroadbandModel',
            continuous=True,
            word_confidence=True,
            timestamps=True
        ).get_result()
        
        # Parse odpověď
        if response['results']:
            transcript = response['results'][0]['alternatives'][0]['transcript']
            print('Speech-to-Text response:', transcript)
            return transcript.strip()
        else:
            print('No speech detected')
            return 'null'
            
    except Exception as e:
        print(f"STT Error: {e}")
        return 'null'

''''
def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url+'/speech-to-text/api/v1/recognize'
    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }
    # Set up the body of our HTTP request
    body = audio_binary
    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()
    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text
'''
def text_to_speech(text, voice="en-US_MichaelV3Voice"):
    """
    Převede text na řeč pomocí IBM Watson TTS API
    """
    try:
        # Zavolej Watson TTS API
        response = text_to_speech_service.synthesize(  # Použij text_to_speech_service
            text=text,
            voice=voice,
            accept='audio/wav'
        ).get_result()
        
        return response.content
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

'''
def text_to_speech(text, voice=""):
    # def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    base_url = 'https://sn-watson-tts.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP Post reqeust to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content
'''

def watsonx_process_message(user_message):
    try:
        prompt = f"""You are an assistant helping translate sentences from English into Spanish.
        Translate the query to Spanish: ```{user_message}```."""
        response_text = model.generate_text(prompt=prompt)
        print("watsonx response:", response_text)
        return response_text
    except Exception as e:
        print(f"Error in watsonx_process_message: {e}")
        return f"Translation error: Unable to connect to watsonx service. Please try again."
