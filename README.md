
<br />
<p align="center">

  <h3 align="center">A Prompting Framework for Natural Language Processing in the Medical Field </h3>

  <p align="center">
    Thesis work by Anim Mondal
  </p>
</p>

##  What is part of this repo?
This repository contains the work and results of the master's thesis written by me. 

## Things to know 

The framework file itself is not a large contributor to the project, but if you would like to use the same structure as me, simply create an object and use the functions within the class. 
The main work came from choosing the prompts to assess the models and tinkering with the parameters. If you want to use the same questions and prompts as me, load the .npy files into your project. They are arrays of Strings that contain the questions, medical reasoning tasks, and medical conversations. Pass those arrays into the framework functions to obtain results. You could pass your own questions and tasks using these functions. 

### Organisation


- **Old/**: folder with the old results from earlier runs of the project.
- **GPT_SW3_framework.py**: class to utilise the functions in the framework. 
- **.npy**: numpy objects with the questions, reasoning tasks, and conversations. 
- **Results.xlsx**: Excel sheet with all of the results from the final runs and the results that were used for analysis for the thesis.

## Citation
If you use this work, please use the following reference:
```
@mastersthesis{Mondal_2023,
series={TRITA-CBH-GRU},
title={A Prompting Framework for Natural Language Processing in the Medical Field : Assessing the Potential of Large Language Models for Swedish Healthcare},
url={https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-328900},
abstractNote={The increasing digitisation of healthcare through the use of technology and artificial intelligence has affected the medical field in a multitude of ways. Generative Pre-trained Transformers (GPTs) is a collection of language models that have been trained on an extensive data set to generate human-like text and have been shown to achieve a strong understanding of natural language. This thesis aims to investigate whether GPT-SW3, a large language model for the Swedish language, is capable of responding to healthcare tasks accurately given prompts and context. To reach the goal, a framework was created. The framework consisted of general medical questions, an evaluation of medical reasoning, and conversations between a doctor and patient has been created to evaluate GPT-SW3’s abilities in the respective areas. Each component has a ground truth which is used when evaluating the responses. Based on the results, GPT-SW3 is capable of dealing with specific medical tasks and shows, in particular instances, signs of understanding. In more basic tasks, GPT-SW3 manages to provide adequate answers to some questions. In more advanced scenarios, such as conversation and reasoning, GPT-SW3 struggles to provide coherent answers reminiscent of a human doctor’s conversation. While there have been some great advancements in natural language processing, further work into a Swedish model will have to be conducted to create a model that is useful for healthcare. Whether the work is in fine-tuning the weights of the models or retraining the models with domain-specific data is left for subsequent works. },
author={Mondal, Anim},
year={2023},
collection={TRITA-CBH-GRU} }
```

## Contact

Anim Mondal - animmondal1997@gmail.com
