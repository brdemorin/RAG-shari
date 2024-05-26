# streamlit run chat.py; to close, ctrl+C in terminal first then close browser
# >3K files chunked into 2K chunks; 8 chunks retrieved
# With translation. Choice of similarity search or .as_retriever (which seem identical except I can choose to retrieve more than 4 chunks if I want with similarity search) and finally able to get retrieved docs while still using the chain.invoke sequence 

import os
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain.prompts import ChatPromptTemplate # deprecated
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from google.cloud import translate_v2 as translate
from langdetect import detect
import streamlit as st
import tiktoken
# Set verbosity using the new approach
from langchain import globals as langchain_globals
langchain_globals.set_verbose(True)
from langchain_groq import ChatGroq
import json
from google.oauth2 import service_account

# if using secrets in .env in root folder:
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
# # I had to take the .json I downloaded from google cloud service account and convert into a single-line string then update .env with it
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# if using secrets in secrets.toml (which is what streamlit.io cloud supports), must create .streamlit subfolder and put secrets.toml in it then also load as environment variables below
OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["secrets"]["GROQ_API_KEY"]
PINECONE_API_KEY = st.secrets["secrets"]["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["secrets"]["PINECONE_API_ENV"]

# google service cloud API for translation fixed by gemini 1.5 pro
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["secrets"]["GOOGLE_APPLICATION_CREDENTIALS"]
credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS) 
credentials = service_account.Credentials.from_service_account_info(credentials_info)
translate_client = translate.Client(credentials=credentials)

# Now set SOME of the secrets from secrets.toml as environment variables. It doesn't appear openAI needs this but I did it anyhow. If I try to do this for google, google will stop working
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_API_ENV 


# I get faster and shorter responses API'ing into GROQ and using the llama3 70b model vs openAI GPT 3.5 Turbo and price is about same ... but sacrifice a decent amount of context window. I change to GPT3.5 when retrieval is more than Groq can handle for a given question
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o") # or gpt-4o-2024-05-13
# response = model.invoke("This is a test. Simply respond with, 'GPT initialized'")
# print(response)



# pip install langchain-groq (I added to requirements.txt for future) # langchain documentation for groq: https://python.langchain.com/v0.1/docs/integrations/chat/groq/
model2 = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192") # 8192 token limit on groq but actually can't even reach that since the API limit on groq with this model is 6,000 tokens per minute
# response = model2.invoke("This is a test. Simply respond with, 'Llama3 70B on Groq initialized'")
# print(response)

# Don't run this. Groq has a bigger context window with mistral (Context Window: 32,768 tokens) but can't use it since groq limits API to 5,000 tokens per min with this model which is less than the 6K it allows for Llama3 70b
#model = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

def is_english(raw_query):
    try:
        lang = detect(raw_query)
        if lang == 'en':
            print(f'English detected')
            return True
        else:
            print(f'Not English detected.')
            return False
    except:
        return False
def get_translation(text, detected_language = None):
    #translate_client = translate.Client() # commented out since new script defines in the beginning and makes global
    result = translate_client.detect_language(text)
    # this is path for incoming question from human to detect language and translate to English if not English
    if not detected_language:
        detected_language = result['language']
        print("Detected language:", detected_language)
        if detected_language == 'en':  # If language is English, do nothing
            translated_text = text
            print(f'Question is in "{detected_language}", do nothing.')
            return translated_text, detected_language
        else: # if not English, translate to English so we can do the semantic search in English
            translated_text = translate_client.translate(text, target_language='en')['translatedText']
            print("Translated text for semantic search:", translated_text)
            return translated_text, detected_language

    # this is path to translate the response if user does not speak English
    if detected_language:
        translated_text = translate_client.translate(text, target_language=detected_language)['translatedText']
        #print("Translated text:", translated_text)
        return translated_text

class ConditionalModelStep:
    def __init__(self, token_threshold, model_below_threshold, model_above_threshold):
        self.token_threshold = token_threshold
        self.model_below_threshold = model_below_threshold
        self.model_above_threshold = model_above_threshold
        self.selected_model = None

    def __call__(self, prompt):
        if len(tokens) < self.token_threshold:
            self.selected_model = "model2"
            return self.model_below_threshold.invoke(prompt)
        else:
            self.selected_model = "model"
            return self.model_above_threshold.invoke(prompt)

    def get_selected_model(self):
        return self.selected_model

def count_tokens(input_string: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    global tokens
    tokens = tokenizer.encode(input_string)
    print(f'number of tokens: {len(tokens)}')
    return len(tokens)

# def calculate_input_cost(original_prompt):
#     # Convert the "prompt" to a string for token counting
#     input_string = str(original_prompt)
#     num_tokens = count_tokens(input_string)
#     # GPT-4o cost per M input tokens
#     cost_per_million_tokens: float = 5
#     total_cost = (num_tokens / 1_000_000) * cost_per_million_tokens
#     print(f"The total cost for using gpt-4o is: ${total_cost:.6f}")

def get_response(query, model_selection_user, detected_language):
    """Answers a question based on context from a Pinecone vector store."""  
    index_name = "shari"
    embeddings = OpenAIEmbeddings()
    pinecone = PineconeVectorStore.from_existing_index(index_name, embeddings)
    #model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    parser = StrOutputParser()
    template = """
    Provide a deep-dive explanation to the below question based upon the "context" section. If you can't answer the question based on the context, then say so.

    Context: {context}

    Question: Give me as much info as you can on this subject: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieve the documents using similarity search. 
    retrieved_docs = pinecone.similarity_search(query, k=8)  # Adjust k as needed which not something I can change when using retriever.invoke(query). I can put in k=5 but I will only get 4 which is the default

    # # Retrieve the documents using Maximal Marginal Relevance Searches (mmr). I notice that when I use the one that defines search_type="mmr", I get very poor retrieval results compared to without it or similarity search
    # #retriever = pinecone.as_retriever(search_type="mmr") # do not use. Terrible
    # retriever = pinecone.as_retriever() # this seems to retrieve same as the similarity search
    # retrieved_docs = retriever.invoke(query) # when I try to add k=5 here, I still get 4 even though I will get 5 if I use the similarity search method

    # Join the retrieved documents' page_content into a single string
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    #print(context)


    # def print_prompt(prompt):
    #     print(f"Prompt: {prompt}")
    #     return prompt

    def copy_prompt(prompt):
        global original_prompt
        original_prompt = prompt
        #calculate_input_cost(original_prompt)
        count_tokens(str(original_prompt))
        return prompt
    
    # Define your models
    model_below_threshold = model2
    model_above_threshold = model

    # if the user selected GPT3.5, the model for <6K tokens is now GPT 3.5 as well. If the user selects "Auto" or Llama3, the function runs normally where >6K token prompts are handed to GPT3.5 so it doesn't error
    if model_selection_user == "GPT 3.5 Turbo":
        model_below_threshold = model
        model_above_threshold = model

    # Create an instance of the ConditionalModelStep
    conditional_model_step = ConditionalModelStep(6000, model_below_threshold, model_above_threshold)

    # Define the chain
    chain = prompt | copy_prompt | conditional_model_step | parser

    # Invoke the chain
    result = chain.invoke({
        "context": context,
        "question": query
    })

    if not model_selection_user == "GPT 3.5 Turbo":
        # Get the selected model
        selected_model = conditional_model_step.get_selected_model()
        # Print or log the selected model
        print(f'The selected model was: {selected_model}')
    else:
        selected_model = "model"
        # Print or log the selected model
        print(f'The selected model was: {selected_model}')

    if selected_model == "model":
        # Token count + cost
        num_input_tokens = count_tokens(str(original_prompt))
        input_cost = (num_input_tokens / 1_000_000) * .5 # GPT-3.5 Turbo is $0.5 per M token (input)
        #input_cost = (num_input_tokens / 1_000_000) * 5 # GPT-4o is $5 per M token (input)
        num_output_tokens = count_tokens(str(result))
        output_cost = (num_output_tokens / 1_000_000) * 1.5 # GPT-3.5 Turbo is $1.5 per M token (output)
        #output_cost = (num_output_tokens / 1_000_000) * 15 # GPT-4o is $15 per M tokens (output)
        total_cost = input_cost + output_cost
    elif selected_model == "model2":
        # Token count + cost
        num_input_tokens = count_tokens(str(original_prompt))
        input_cost = (num_input_tokens / 1_000_000) * .59 # Llama3 70b on Groq is $0.59 per M token (input)
        num_output_tokens = count_tokens(str(result))
        output_cost = (num_output_tokens / 1_000_000) * .79 # Llama3 70b on Groq is $.79 per M token (output)
        total_cost = input_cost + output_cost
    if detected_language != "en":
        translation = get_translation(result, detected_language)
        result = translation
        return result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, selected_model
    else: 
        return result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, selected_model

def sources_to_print(retrieved_docs):
    """Prints unique URLs and titles from retrieved documents."""
    unique_urls = set()
    unique_sources = set()
    for doc in retrieved_docs:

        # print(f"Document {i}:")
        # print("Content:")
        # print(doc.page_content)
        # print("Metadata:")
        # print(doc.metadata)

        source = doc.metadata.get("source")
        if source:
            unique_sources.add(source)
        url = doc.metadata.get("url")
        if url:
            unique_urls.add(url)

    return unique_urls, unique_sources
        #print(context)

# # Custom CSS to make the sidebar width more dynamic to fit the content it contains
# st.markdown("""
# <style>
#     .css-1d391kg {
#         width: auto !important; /* Allow the width to adjust automatically */
#         padding-right: 10px; /* Add padding to the right */
#         padding-left: 10px; /* Add padding to the left */
#     }
#     .css-1d391kg .css-1n76uvr {
#         width: auto !important; /* Ensure inner elements also adjust */
#     }
#     .css-1d391kg .css-1n76uvr div {
#         max-width: 250px; /* Set a maximum width for the content */
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
    .css-1d391kg {
        width: auto !important; /* Allow the width to adjust automatically */
        padding-right: 10px; /* Add padding to the right */
        padding-left: 10px; /* Add padding to the left */
    }
    .css-1d391kg .css-1n76uvr {
        width: auto !important; /* Ensure inner elements also adjust */
    }
    .css-1d391kg .css-1n76uvr div {
        max-width: 250px; /* Set a maximum width for the content */
    }
    
    /* Move main content up by adjusting the negative value */
    .main .block-container {
        margin-top: -4rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Shari's Burnout Bot 🤖")
    st.write("**by Shari's brother** | **Model: Mixture of GPT3.5 Turbo and Llama3 70b** 🧠") 
    st.write("This genAI model has been trained on copyrighted materials by Shari Morin.")
    #st.write("**by digitalcxpartners.com**\n\nModel: GPT3.5 Turbo")

    model_selection_user = st.selectbox("Select foundation model:", ["Auto", "GPT 3.5 Turbo", "Llama3 70b"], index=0)

    #query = st.text_area("Enter your prompt: :pencil2:")
    query = st.text_area(":pencil2: Enter your prompt:")



    # For onscreen variables that need ability to change with each run. Check if session variables are already initialized
    if 'session_cost' not in st.session_state:
        st.session_state.session_cost = 0.0
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    # Initialize a new session state variable to store the previous session cost:
    if 'previous_session_cost' not in st.session_state:
        st.session_state.previous_session_cost = 0.0
    # if 'selected_model' not in st.session_state:
    #     st.session_state.selected_model = "default"


    # Generate output
    if st.button("Run 🚀"):
        if query:
            with st.spinner("Generating response... :hourglass_flowing_sand:"):

                # Add your logic here to use the selected model
                if model_selection_user == "Auto":
                    # Logic for auto model selection
                    pass
                elif model_selection_user == "GPT 3.5 Turbo":
                    # Logic for GPT 3.5 Turbo
                    pass
                elif model_selection_user == "Llama3 70b":
                    # Logic for Llama3 70b
                    pass
                # Replace the pass statements with the actual model handling logic
                #st.success("Response generated successfully!")


                # I introduced is_english function to run a local True/False on English to eliminate unecessary API call to Google translate. If I remove is_english, the script will still work perfectly but just means burning a lot of API calls for English detection
                language_check = is_english(query)
                if language_check: # English path
                    #print('English path taken')
                    detected_language = "en"
                    #st.markdown("Generating response...wait just a few seconds")
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, selected_model = get_response(query, model_selection_user, detected_language)
                else: # non-English path
                    #print('non-English path taken')
                    query2, detected_language = get_translation(query)
                    result, retrieved_docs, num_input_tokens, input_cost, num_output_tokens, output_cost, total_cost, selected_model = get_response(query2, model_selection_user, detected_language)
                #print(result)
                unique_urls, unique_sources = sources_to_print(retrieved_docs)
                #st.write(result)

                url_list = "\n".join(f"{url}," for url in unique_urls)
                title_list = "\n".join(f"- {title}" for title in unique_sources)  # Using bullet points for clarity

                if detected_language == "en":
                    st.markdown(f"{result}\n\n**Check out these links for more:**\n\n{url_list}\n\n**Titles sourced:**\n\n{title_list}")
                    #with st.expander("📈 Expand to see cost for query"):
                    with st.expander("🔽 Expand for query cost"):
                        st.markdown(f'\n\n**Cost:**\n\n- Input tokens = {num_input_tokens}\n\n- Cost for question: ${input_cost:.6f}')
                        st.markdown(f'- Number of output tokens = {num_output_tokens}\n\n- Cost for response: ${output_cost:.6f}')
                        st.markdown(f'- Total cost for question + answer = ${total_cost:.6f}')
                        #st.write(f"{result}\nCheck out these links for more:\n{unique_urls}\n{unique_titles}")

                else:
                    st.markdown(f"{result}\n\n{url_list}")
                    st.markdown(f'\n\n**Cost:**\n\n- Input tokens = {num_input_tokens}\n\n- Cost for question: ${input_cost:.6f}')
                    st.markdown(f'- Number of output tokens = {num_output_tokens}\n\n- Cost for response: ${output_cost:.6f}')
                    st.markdown(f'- Total cost for question + answer = ${total_cost:.6f}')

                # using "session state" to change onscreen variables
                st.session_state.session_cost += total_cost
                st.session_state.question_count += 1
                st.session_state.selected_model = selected_model


                session_cost_delta = st.session_state.session_cost - st.session_state.previous_session_cost
                st.session_state.previous_session_cost = st.session_state.session_cost

                # # Display the session variables in the sidebar
                # #st.sidebar.markdown(f"**Session Cost:** ${st.session_state.session_cost:.6f}")
                st.sidebar.markdown(f"**Session Cost:**")
                st.sidebar.metric(label="USD", value=f"${st.session_state.session_cost:.6f}", delta=f"${session_cost_delta:.6f}")
                st.sidebar.markdown(f"** **")
                st.sidebar.markdown(f"**Question Count:** {st.session_state.question_count} 💬")
                st.sidebar.markdown(f"** **")
                if selected_model == "model":
                    st.sidebar.caption(f"**Last model invoked:**\n\nGPT-3.5 🧠")
                if selected_model == "model2":
                    st.sidebar.caption(f"**Last model invoked:**\n\nLlama3 70b 🧠")





                # st.sidebar.write("**Session Cost:**")
                # st.sidebar.write(f"${st.session_state.session_cost:.6f}")
                # st.sidebar.write(f"Delta: ${session_cost_delta:.6f}")
                # st.sidebar.write(f"**Question Count:** {st.session_state.question_count} 💬")



        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
