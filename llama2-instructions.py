import sys
import time

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, LLMPredictor, SimpleDirectoryReader, VectorStoreIndex, \
    StorageContext, load_index_from_storage, PromptHelper
from langchain.llms import LlamaCpp
import gradio as gr

# Initialize and configure parameters
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
llm_predictor = LLMPredictor(
    llm=LlamaCpp(
        model_path='YOUR LLAMA2 MODEL PATH', n_ctx=4096,verbose=True,
        input={"temperature": 0.75, "max_length": 2000, "top_p": 1}
    )
)
service_context = ServiceContext.from_defaults(
    context_window=4000,
    chunk_size=1024,
    llm_predictor=llm_predictor,
    embed_model=embed_model
)

def construct_index(directory_path):
    # Load documents and create index
    documents = SimpleDirectoryReader(directory_path).load_data()
    #create llama_index
    index = VectorStoreIndex.from_documents(
        documents=documents, service_context=service_context, show_progress=True
    )
    #save index
    index.storage_context.persist(persist_dir='YOUR STORAGE PATH')
    return index

# Process input query and return the response
# It might take a long time to generate your response, depending on your device conditions and file size
def process_query(input_text, history):

    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(input_text)
        print(response)
        return (
                str(response.response)
        )


    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return "An error occurred. Please try again."

##option 1: Create a new index from your file folder
index=construct_index("docs")

##option 2:
#IF you donnot want to create index again, you can also load the index that we have saved
storage_context = StorageContext.from_defaults(persist_dir='YOUR STORAGE PATH')
index = load_index_from_storage(storage_context=storage_context,service_context=service_context)


# Create Gradio interface
iface = gr.ChatInterface(
    fn=process_query,
    chatbot=gr.Chatbot(height=400),
    title="LLAMA2 AI Chatbot",
    description="LLAMA2 AI Chatbot",
    theme="soft",
    #enable this, if you want to set some examples for your chatbot
    #examples=["Who are the most prominent authors in the International Networked Learning Conference?"],
    #cache_examples=True,

)
# Launch the interface
iface.launch(share=True,enable_queue=True)