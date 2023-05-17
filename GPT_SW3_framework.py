import importlib

libraries = ['transformers', 'openpyxl', 'sentencepiece', 'torch','pandas','numpy']

for library in libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is installed and imported successfully.")
    except ImportError:
        print(f"{library} is not installed.")

import pandas as pd
import numpy as np
import torch
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login

def _cleanStringAndAddQuestion(prompt, generated_text):
    """This is a helper function that will remove some artefacts of the generated responses of the LLMs and add a question during conversation.

    Args:
        prompt (String): The new prompt.
        generated_text (String): The previous portion of the conversation.

    Returns:
        String: Cleaned up string with the new question added.
    """
    try: 
        text = str(generated_text).replace('[{\'generated_text\': \'', "").replace('])}.\'}]' , "").replace('\'}]', "").replace('\\' , "")
        text = text + " " + prompt
        return text

    except:
        return prompt

def _cleanString(generated_text):
    """This is a helper function that will remove some artefacts of the generated responses of the LLMs.

    Args:
        generated_text (String): Text that needs cleaning.

    Returns:
        String: Cleaned text.
    """
    return str(generated_text).replace('[{\'generated_text\': \'', "").replace('])}.\'}]' , "").replace('\'}]', "").replace('\\' , "")

class Framework:
    """A class representing a framework.

    This class provides a framework for performing generative operations.

    :param token: The token used for authentication.
    """
    def __init__(self, token):
        self.token = token
        login(token=self.token)
        dataframe_names = ['Model', 'Question', 'Max Tokens', 'Repetition Penalty', 'Generated Response', 'Time and Date']
        self.dataframe = pd.DataFrame(columns=dataframe_names)


    def modelGenerateConversation(self, model_name, prompts, max_new_tokens = 150, repetition_penalty = 2.0, temperature = 0.3, generated_text = ""):
        """The main function for conversations. The function will chain the prompts and reponses to evaluate the entire conversation.

        Args:
            model_name (String): Name of the model on Hugging Face.
            prompts (Array): An array of the prompts that you would like the conversation to contain. 
            max_new_tokens (int, optional): Maximum number of tokens that the responses will contain. Defaults to 150.
            repetition_penalty (float, optional): How much you want to penalise the model for repeating sentences. Defaults to 2.0.
            temperature (float, optional): How creative you want the model to be in its response. Defaults to 0.3.
            generated_text (str, optional): Previous conversation that you would like to continue. Defaults to "".
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model.to(device)


        if isinstance(prompts, str):
            prompts = [prompts]  # Convert the string to a single-item array

        for subprompt in (prompts):
            print(subprompt)
            time = datetime.now()
            input_ids = tokenizer(_cleanStringAndAddQuestion(prompt=subprompt, generated_text=generated_text), return_tensors="pt")["input_ids"].to(device)

            generated_token_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1,
            repetition_penalty = repetition_penalty
            )[0]
            generated_text = tokenizer.decode(generated_token_ids)

            print(generated_text)
            new_row = {'Model' : model_name, 'Question' : subprompt, 'Max Tokens' : max_new_tokens, 'Repetition Penalty' : repetition_penalty, 'Temperature' : temperature, 'Generated Response' : [generated_text],'Time and Date' : time}
            combine = pd.DataFrame(new_row, columns=self.dataframe.columns.tolist())
            self.dataframe = pd.concat([self.dataframe, combine], ignore_index=True)



    def modelGenerateQuestion(self, model_name, prompts, max_new_tokens = 150, repetition_penalty = 2.0, temperature = 0.3):
        """The main function for conversations. The function will answer each question separately in prompts.

        Args:
            model_name (String): Name of the model on Hugging Face.
            prompts (Array or String): An array of the questions that you would like the model to answer. Can also be a single String.
            max_new_tokens (int, optional): Maximum number of tokens that the responses will contain. Defaults to 150.
            repetition_penalty (float, optional): How much you want to penalise the model for repeating sentences. Defaults to 2.0.
            temperature (float, optional): How creative you want the model to be in its response. Defaults to 0.3.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model.to(device)


        if isinstance(prompts, str):
            prompts = [prompts]  # Convert the string to a single-item array

        for subprompt in (prompts):
            print(subprompt)
            time = datetime.now()
            input_ids = tokenizer(subprompt, return_tensors="pt")["input_ids"].to(device)

            generated_token_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1,
            repetition_penalty = repetition_penalty
            )[0]
            generated_text = tokenizer.decode(generated_token_ids)

            print(generated_text)
            new_row = {'Model' : model_name, 'Question' : subprompt, 'Max Tokens' : max_new_tokens, 'Repetition Penalty' : repetition_penalty, 'Temperature' : temperature, 'Generated Response' : [generated_text],'Time and Date' : time}
            combine = pd.DataFrame(new_row, columns=self.dataframe.columns.tolist())
            self.dataframe = pd.concat([self.dataframe, combine], ignore_index=True)

    def save(self):
        """Saves the dataframe to an Excel sheet in the form "output"day_month_year_hour_minute_seconds.xlsx in the same folder as the framework. 
        """
        self.dataframe.to_excel("output" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx", index=False)
