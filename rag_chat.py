import os
from document_db import DocumentDb
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import IO
from dotenv import load_dotenv
load_dotenv()


MAX_DISTANCE = 1.0
TEMPLATE = """Use the following pieces of context and images if there are to answer the question at the end. Keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""


class RagChat:
  def __init__(self):
    self.messages = []
    self.llm = ChatOpenAI(model=os.environ['OPEN_AI_MODEL_NAME'], api_key=os.environ['OPEN_AI_API_KEY'])
    self.augmented_prompt_template = PromptTemplate.from_template(TEMPLATE)
    self.document_db = DocumentDb()


  def handle_pdf(self, file: IO[bytes], file_name: str) -> None:
    self.document_db.handle_pdf(file, file_name)


  def answer_user_questions(self, query: str) -> str:
    if len(query) == 0:
      return ''
    
    query_result = self.document_db.query(query)
    relevant_indexes = [index for index, dist in enumerate(query_result['distances'][0]) if dist < MAX_DISTANCE]
    if len(relevant_indexes) == 0:
      self.messages.append(HumanMessage(query))
    else:
      context = []
      img_contents = []
      for index in relevant_indexes:
        if query_result['metadatas'][0][index]['type'] == 'image':
          img_content = {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{query_result['metadatas'][0][index]['base_64']}"},
            }
          img_contents.append(img_content)
        else:
          current_doc_content = query_result['documents'][0][index]
          context.append(current_doc_content)
      human_message = HumanMessage(content=[
          {
              "type": "text",
              "text": self.augmented_prompt_template.format(context=context, question=query)
          }, *img_contents])
      self.messages.append(human_message)
  
    ai_message = llm_response = self.llm.invoke(self.messages)
    self.messages.append(ai_message)
    return ai_message.content
