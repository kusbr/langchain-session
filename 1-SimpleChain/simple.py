import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.language_models.chat_models import BaseChatModel

from azure.identity import DefaultAzureCredential

def main():

    # Load keys from .env file
    load_dotenv()

    # Create a PromptTemplate instance
    prompt = PromptTemplate(
        input_variables=["question"],
        template="I want you to answer the following question: {question}",
    )

    # query
    query = "Who created LangChain?"

    # LLM Instances
    
    # openai 
    openai_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Azure OpenAI
    azure_llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        azure_credentials=DefaultAzureCredential(),
    )

    # Local PHI4
    phi4_llm = OllamaLLM(model="phi4")

    # Local Llama3
    llam3_llm = OllamaLLM(model="llama3")

    # Local deepseek-r1
    deepseek_r1_llm = OllamaLLM(model="deepseek-r1") 

    # Invoke the LLMs
    InvokeLLM(openai_llm.model_name, openai_llm, prompt, query)
    InvokeLLM(azure_llm.model_name, openai_llm, prompt, query)
    InvokeLLM(phi4_llm.model, phi4_llm, prompt, query)
    InvokeLLM(llam3_llm.model, llam3_llm, prompt, query)
    InvokeLLM(deepseek_r1_llm.model, deepseek_r1_llm, prompt, query)

    
def InvokeLLM(model: str, llm: BaseChatModel, prompt: PromptTemplate, query: str):

    # Create a chain with the prompt and LLM  
    chain = prompt | llm

    # Invoke the chain with the query
    result = chain.invoke(input={"question": query})
    
    # Print the result
    print(f"\n\nModel: {model}: Result: {result}")

if __name__ == "__main__":
    main()