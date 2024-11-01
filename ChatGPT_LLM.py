#from openai import OpenAI
import os


'''example completion with openai > 1.1'''
from openai import OpenAI



def get_LLM_answer_GPT_four(prompt):

    client =  OpenAI(api_key ='xxx')

    completion = client.chat.completions.create( # Change the method
    model = "gpt-4o",
    messages = [ # Change the prompt parameter to messages parameter
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    )

    return completion.choices[0].message.content.strip() # Change message content retrieval

def get_LLM_answer_GPT_Three_five(prompt):

    client =  OpenAI(api_key ='xxx')

    completion = client.chat.completions.create( # Change the method
    model = "gpt-3.5-turbo",
    messages = [ # Change the prompt parameter to messages parameter
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    )

    return completion.choices[0].message.content.strip() # Change message content retrieval
