# Chat with PDF
In this project, we will build a standard RAG pipeline to chat with your local PDF files. We will be using LangChain with OpenAI models, ChromaDB, unstructured, and Streamlit.


## Tech Stack
- Python
- Langchain
- ChromaDb
- unstructured
- Open AI API
- Streamlit

## Installation
1. Clone the repository

```
git clone https://github.com/MiguelSteph/rag-chat-with-pdf.git
```

2. Install the requirements
```
pip install -r requirements.txt
```

3. Create a `.env` file with the following keys
    - **OPEN_AI_MODEL_NAME**: The open AI LLM model to use. 
    - **EMBEDDING_MODEL_NAME**: The open AI embedding model to use.
    - **OPEN_AI_API_KEY** The open AI API key

4. Run the Streamlit app:

```
streamlit run streamlit_ui.py
```

## References
- [unstructured - partition_pdf](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition_pdf)
- [chroma](https://docs.trychroma.com/docs/collections/manage-collections)
- [Develop Web Apps with Streamlit](https://www.educative.io/courses/develop-web-apps-streamlit/streamlit-cloud)
- [Fundamentals of Retrieval-Augmented Generation with LangChain](https://www.educative.io/courses/rag-llm/understanding-retrieval-and-generative-models)
- [Multimodal RAG: Chat with PDFs (Images & Tables)](https://youtu.be/uLrReyH5cu0)
