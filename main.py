import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

st.title("🦜🔗 Name Generator App")

openai_api_key = st.sidebar.text_input("OpenAI API Key")


# function to generate text using llm model
def generate_text_from_openai(input_text):
    # prompts
    name_template = "What are some good names for {subject}?"
    explain_template = "Explain the meaning of these names:{names}"
    name_prompt_template = PromptTemplate(
        input_variables=["subject"], template=name_template
    )
    explain_prompt_template = PromptTemplate(
        input_variables=["names"], template=explain_template
    )

    # chains
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    name_chain = LLMChain(llm=llm, prompt=name_prompt_template, output_key="names")
    explain_chain = LLMChain(
        llm=llm, prompt=explain_prompt_template, output_key="explanation"
    )

    # sequential chains
    overall_chain = SequentialChain(
        chains=[name_chain, explain_chain],
        input_variables=["subject"],
        # Here we return multiple variables
        output_variables=["names", "explanation"],
        verbose=True,
    )
    response = overall_chain({"subject": input_text})
    st.info(response["names"])
    st.info(response["explanation"])


with st.form("my_form"):
    # textarea
    text = st.text_area(
        "Tell us the subject that you want to generate names for:",
        "My dog",
    )

    # submit button
    submitted = st.form_submit_button("Submit")

    # check if there is a correct openapi key
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_text_from_openai(text)
