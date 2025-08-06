from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Set up the API key and project ID for IBM Watson 
watsonx_API = "WR9cz4JUfWoFJhal7ihYnv7HLN2TFORzlPIWinmuPrT1" # below is the instruction how to get them
project_id= "3c47116b-7ce6-484f-b416-1af22b65b9bf" # like "0blahblah-000-9999-blah-99bla0hblah0"

generate_params = {
    GenParams.MAX_NEW_TOKENS: 250
}

model = Model(
    # model_id = 'ibm/granite-13b-chat-v2',
    model_id = 'meta-llama/llama-2-70b-chat', # you can also specify like: ModelTypes.LLAMA_2_70B_CHAT
    params = generate_params,
    credentials={
        "apikey": watsonx_API,
        # "url": "https://ml.cloud.ibm.com"
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id= project_id
    )

q = "How to be happy?"
generated_response = model.generate(prompt=q)
print(generated_response['results'][0]['generated_text'])