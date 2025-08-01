import os
import io
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent,AgentType
from dotenv import load_dotenv
import requests
from langchain.tools import tool
import base64
load_dotenv()

# os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
# os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
# os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
# os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Set environment variables from Streamlit secrets
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
os.environ['NVIDIA_API_KEY'] = st.secrets['NVIDIA_API_KEY']
os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_API_KEY']
os.environ['SERPER_API_KEY'] = st.secrets['SERPER_API_KEY']
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
nvidia_api_key = st.secrets['NVIDIA_API_KEY']

st.set_page_config(page_title='Welcome to OmniBot',page_icon='🐋')
# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'messages' in st.session_state:
            del st.session_state['messages']
        st.rerun()
st.markdown("### 🌊 **OmniBot-AI** — Your Intelligent Assistant for Coding, Creative Tasks, and Meaningful Interactions")

st.sidebar.title("🌟 About OmniBot")

st.sidebar.markdown("""
Meet **OmniBot**, your ultimate AI companion designed to assist with a wide range of tasks!

Whether you're a:
- 👨‍💻 Developer seeking **coding assistance**
- 🎨 Creative thinker looking for **image inspiration**
- 💬 Conversationalist enjoying **meaningful interactions**

OmniBot is here to make your life easier.

### 🧠 Key Capabilities:
- 💻 Intelligent coding support  
- 🖼️ Creative image generation  
- 🌐 Real-time information access  
- 📄 Visual Q&A on documents and images  
- ✨ Smart summarization of content  

Boost your **productivity**, fuel your **creativity**, and enjoy rich, interactive conversations — all in one place with **OmniBot**!
""")

st.sidebar.title("🔑 Task Keywords Guide")

st.sidebar.markdown("""
Use the following **keywords** in your prompt to trigger specific AI abilities:

- 🔍 **Web Search**: Use keywords like `"search"`, `"find"`, `"look up"`  
- 🖼️ **Image Generation**: Use `"generate image"`, `"create image"`, or `"draw"`  
- 🖼️ **Image Analysis**: Use `"uploaded image"`, `"show image"`, `"describe image"`, `"analyze image"`
- 👨‍💻 **Code Help**: Use `"code"`, `"program"`, or `"script"`  
- 💬 **General Q&A or Chat**: Just type your question or start chatting!

---

🎯 Try something like:
- `"Search latest AI trends"`
- `"Find place to eat in Gandhinagar"`
- `"Search image of a mountain sunset"`
- `"Generate image of a futuristic city"`
- `"Write a Python script to scrape a website"`
- `"describe the uploaded image"`

""")



#----------------------------------------------------Tools----------------

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
wiki_tool = Tool.from_function(
    name='Wikipedia',
    func=wiki.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)

tavily = TavilySearchResults(
    max_results=2,
    include_answer=True,
    include_images=True,
    search_depth='advanced',
)
tavily_tool = Tool.from_function(
    name='tavily',
    func=tavily.run,
    description='A tool for getting real time world infromation'
)

serper = GoogleSerperAPIWrapper()
serper_tool = Tool.from_function(
    name='serper',
    func=serper.run,
    description='Use the Google Serper API to search for places, news, images, videos, or perform an image search from a URL.'
)


template = """
You are Omnibot-AI, an intelligent assistant developed by **Bhautik Vadaliya**.

Instructions:
- Always introduce yourself as: *"I am Omnibot-AI, developed by Bhautik Vadaliya."*
- Never mention Google, OpenAI, or any other company as your creator.
- Respond in a helpful, logical, and friendly manner.
- Format responses clearly, using bullet points where appropriate.
- If a tool fails, rely on your own knowledge.
- If the question is **code-related**, answer like an expert coder (NLP, AI, ML, web dev, etc.), and include code snippets and explanations.
- Otherwise, respond conversationally as a smart assistant.

User: {question}
Omnibot-AI:
"""

prompt = PromptTemplate(
    input_variables=['question'],
    template=template
)

# Chain setup
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
# output_parser = StrOutputParser()  # Keeps output clean
chain = prompt | llm

# Tool definition
llm_tool = Tool.from_function(
    name='Omnibot-AI Assistant',
    func=lambda q: chain.invoke({"question": q}).content,
    description=(
        'An assistant that answers both general and code-related questions. '
        'Responds with detailed, structured information and helpful explanations.'
    )
)



def gen_img(question: str) -> str:
    """Generates an image using the prompt provided in the question string."""
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
    }

    payload = {
        "prompt": question,
        "cfg_scale": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "steps": 50,
        "negative_prompt": ""
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        response_body = response.json()

        if "image" in response_body:
            image_data = response_body["image"]
            image_bytes = base64.b64decode(image_data)

              # Display image directly from memory
            st.image(io.BytesIO(image_bytes), caption="Generated Image", use_container_width=True)
            return "Image generated successfully."

        else:
            return "Error: No image data found in the response."

    except Exception as e:
        return f"Request failed: {str(e)}"

gen_img_tool = Tool.from_function(
    name="Generate image",
    func=gen_img,
    description="Generates an image based on the input question or prompt using NVIDIA's Stable Diffusion 3 Medium model."
)

# llm_code = ChatGroq(model='qwen-qwq-32b')

# template_code = """
# You are a coding assistant. Given the user's question, return **only the answer or code**. Do not include explanations or markdown formatting.

# Question: {question}
# Answer:
# """
# prompt_code = PromptTemplate(
#     input_variables=['question'],
#     template=template_code
# )
# chain_code = prompt_code|llm_code
# llm_code_tool = Tool.from_function(
#     name='Expert Coder',
#     func=chain_code.invoke,
#     description='A tool for answering code related question.'
# )


st.sidebar.write(' ')
st.sidebar.write('')
st.sidebar.write('🌩️Visual Question Answer')
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"],label_visibility='visible')
if uploaded_file:
    st.sidebar.image(uploaded_file, caption='uploaded file')

def visual_qa(question: str) -> str:

    if uploaded_file is not None:
        # Read and convert to base64
        image_bytes = uploaded_file.read()
        image_b64 = base64.b64encode(image_bytes).decode()

        if len(image_b64) > 4_194_304:  # >3 MB
            st.warning("Image too large. Please upload a smaller image.")
        else:
            # st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

            # API Setup
            invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            stream = False  # Set to True if you want streamed output

            headers = {
                "Authorization": f"Bearer {nvidia_api_key}",  
                "Accept": "text/event-stream" if stream else "application/json"
            }

            content_prompt = question if question else "What is in this image?"
            content_prompt += f'\n<img src="data:image/png;base64,{image_b64}" />'

            payload = {
                "model": 'meta/llama-4-scout-17b-16e-instruct',
                "messages": [
                    {
                        "role": "user",
                        "content": content_prompt
                    }
                ],
                "max_tokens": 512,
                "temperature": 1.0,
                "top_p": 1.0,
                "stream": stream
            }

            # Send request
            response = requests.post(invoke_url, headers=headers, json=payload)

            # Display response
            if stream:
                st.subheader("Streamed Response")
                for line in response.iter_lines():
                    if line:
                        st.write(line.decode("utf-8"))
            else:
                result = response.json()
                try:
                    content = result['choices'][0]['message']['content']
                    # st.subheader("Description:")
                    # st.write(content)
                    return content
                except Exception as e:
                    st.error(f"Error extracting content: {e}")

visual_tool = Tool.from_function(
    name='Visual Question Answer',
    func=visual_qa,
    description='if user question is related to uploaded image then Answer questions about an uploaded image.'
)




#---------------------------------------------------------Build Agent-------------------
final_llm = llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
# final_llm = ChatGroq(model="llama-3.1-8b-instant")
# final_llm = ChatNVIDIA(model='nvidia/llama-3.1-nemotron-ultra-253b-v1')

def build_agent(tools, llm_model=final_llm, max_iter=3):
    return initialize_agent(
        tools=tools,
        llm=llm_model,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=max_iter,
        early_stopping_method="generate",
    )
qa_agent = build_agent([wiki_tool, tavily_tool, llm_tool], max_iter=3)
visual_agent = build_agent([visual_tool], max_iter=2)
image_agent = build_agent([gen_img_tool], max_iter=2)
serper_agent = build_agent([serper_tool], max_iter=3)

def route_query(query, callback_manager):
    content = query[-1]['content'].lower()
    if any(k in content for k in ["search", "find", "look up"]):
        return serper_agent.run(content, callbacks=[callback_manager])
    elif any(k in content for k in ["generate image", "draw", "create image"]):
        return image_agent.run(query, callbacks=[callback_manager])
    elif any(k in content for k in ["provided image", "describe", "show image", "uploaded image", "analyze image"]):
        return visual_agent.run(query, callbacks=[callback_manager])
    else:
        return qa_agent.run(query, callbacks=[callback_manager])



#------------------------------------------------------ Let's build chatbot---------------------------------

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'assistant','content':'How Can I Help You Today?'}]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

user_input = st.chat_input(placeholder='What is Generative AI?')

if user_input:
    st.session_state.messages.append({'role':'user','content':user_input})
    st.chat_message('user').write(user_input)
    # print(st.session_state.messages)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = route_query(st.session_state.messages, st_cb)
        st.session_state.messages.append({'role':'assistant','content':response})
        st.success(response)
else:
    st.info('Please provide the question..')
