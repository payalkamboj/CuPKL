
# from bardapi import BardCookies
# def get_LLM_answer( question):
#     cookie_dict = {
#     "__Secure-1PSID": "ewiWP8Z-RFN_y-NLrBoX7akKQZV2x-fSQbo8DVBmS_fOE92D3uWuqqb3Z_uzXZ9KNwrnAQ.",
#     "__Secure-1PSIDTS": "sidts-CjIBPVxjSuUTB59a4yuxx5NR0nWfZtNIoE1CMcE_scshhf_dYh3XIRkBjBUkWA7H_YoDehAA",
#     # Any cookie values you want to pass session object.
# }
#     # Create BardCookies instance with provided cookies
#     bard = BardCookies(cookie_dict=cookie_dict)
    
#     # Get answer for the specified question
#     answer = bard.get_answer(question)['content']
    
#     return answer



# # question = "what is SOZ"
# # response = get_LLM_answer(cookie_dict, question)
# # print(response)

#############unofficial BARD chatbot  works fine till this point###########

import pprint
import google.generativeai as genai
import os
#genai.configure(api_key='AIzaSyC_vdoxnOJZs_y0ChUp2GvZXkYoimN5FTc')

def get_LLM_answer(prompt):
    model = genai.GenerativeModel(model_name='gemini-pro')
    genai.configure(api_key='AIzaSyC_vdoxnOJZs_y0ChUp2GvZXkYoimN5FTc')
    generation_config={"temperature":0.8, "top_p": 1} #Temperature controls how creative Gemini is, value closer to 1.0 generate more creative response
    response = model.generate_content(prompt, generation_config=generation_config)
    response = response.text
    return response
    
#     return answer


# # model = genai.GenerativeModel(model_name='gemini-pro')
# # prompt = """Who are you?, what is your model? are you an LLM, answer yes or no"""
# # response = get_LLM_answer(model, prompt)
# # response = model.generate_content(prompt)
# # response = response.text
# # print(response)
