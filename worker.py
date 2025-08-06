'''
tato verze je upravena pro prime voleni Speech-to-Text a Text-to-Speech
pomoci HTTP volani, coz je jednodussi a flexibilnejsi nez pouziti SDK.

nicmene neni vylade, na vstupu neakceptuje audio
'''
import requests
import json
from ibm_watson import TextToSpeechV1, SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods


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

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 50,  # Změna z 1024 na 50
    GenParams.STOP_SEQUENCES: ["\n", ".", "English:", "Original:"],  # Přidej stop slova
}

# Define the LLM
model = Model(
    model_id=model_id, params=parameters, credentials=credentials, project_id=PROJECT_ID
)

def speech_to_text(audio_binary):
    """
    STT s automatickou detekcí formátu
    """
    # Seznam modelů k vyzkoušení

    models_to_try = [
        'en-US_BroadbandModel',
        'en-US_NarrowbandModel',
        'en-US_Multimedia'
    ]

    for model_name in models_to_try:
        try:
            print(f"Trying model: {model_name}")

            response = speech_to_text_service.recognize(
                audio=audio_binary,
                content_type="audio/wav",
                model=model_name,
                continuous=True,
                word_confidence=True
            ).get_result()
            
        
            if response.get('results') and len(response['results']) > 0:
                transcript = response['results'][0]['alternatives'][0]['transcript']
                confidence = response['results'][0]['alternatives'][0].get('confidence', 0)
                print(f'SUCCESS with {model_name}: {transcript} (confidence: {confidence})')
                return transcript.strip()
                
        except Exception as e:
            print(f"Failed with {model_name}: {e}")
            continue
    
    print("All formats failed")
    return 'null'
    
def text_to_speech(text, voice="en-US_MichaelV3Voice"):
    """
    TTS pomocí přímého HTTP volání
    """
    url = f"{TTS_URL}/v1/synthesize"
    
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json'
    }
    
    data = {
        'text': text,
        'voice': voice
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            auth=('apikey', TTS_API_KEY)
        )
        
        if response.status_code == 200:
            print(f'Text-to-Speech response: Successfully synthesized {len(response.content)} bytes')
            return response.content
        else:
            print(f"TTS Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
    
def watsonx_process_message(user_message):
    try:
        # Jednodušší a přesnější prompt
        prompt = f"Translate this English text to Spanish: {user_message}\n\nSpanish:"
        
        response_text = model.generate_text(prompt=prompt)
        print("watsonx response:", response_text)
        
        # Vyčisti odpověď od zbytečného textu
        cleaned_response = clean_response(response_text)
        return cleaned_response
        
    except Exception as e:
        print(f"Error in watsonx_process_message: {e}")
        return f"Translation error: Unable to connect to watsonx service. Please try again."

def clean_response(response_text):
    """
    Vyčistí odpověď od opakování a zbytečného textu
    """
    if not response_text:
        return "No translation available"
    
    # Odstranění možných prefixů
    response_text = response_text.replace("Spanish:", "").strip()
    response_text = response_text.replace("Translation:", "").strip()
    
    # Rozdělení podle vět a vzít jen první
    sentences = response_text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if first_sentence:
            return first_sentence
    
    # Pokud je odpověď příliš dlouhá, zkrať ji
    if len(response_text) > 200:
        response_text = response_text[:200] + "..."
    
    return response_text.strip()
