import os
from typing import IO
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


def chunk_pdf(file: IO[bytes], file_name: str) -> list[Document]:
  llm = ChatOpenAI(model=os.environ['OPEN_AI_MODEL_NAME'], api_key=os.environ['OPEN_AI_API_KEY'])

  elements = partition_pdf(file=file,
                          strategy="hi_res",
                          languages=["eng"],
                          infer_table_structure=True,
                          extract_images_in_pdf=True,
                          extract_image_block_types=["Image"],
                          extract_image_block_to_payload=True,
                          chunking_strategy="by_title",
                          chunk_combine_text_under_n_chars=2_000,
                          chunk_max_characters=10_000,
                          chunk_new_after_n_chars=8_000)

  texts = []
  images = []
  tables = []

  # Get the text documents
  for element in elements:
    if len(element.text) > 0:
      document = Document(
          page_content = element.text,
          id=uuid.uuid4().hex,
          metadata={
              "page_number": element.metadata.to_dict()["page_number"],
              "source": file_name,
              "type": "text",
          }
      )
      texts.append(document)

  # Get the images
  for element in elements:
    # file_name = element.metadata.to_dict()["filename"]
    for orig_element in element.metadata.orig_elements:
      orig_element_type = orig_element.to_dict()['type']
      if orig_element_type == 'Image':
        base64_image = orig_element.metadata.to_dict()["image_base64"]
        mime_type = orig_element.metadata.to_dict()["image_mime_type"]
        img_summary = get_img_summary(llm, base64_image, mime_type)
        document = Document(
            page_content = img_summary,
            id=uuid.uuid4().hex,
            metadata={
                "base_64": base64_image,
                "page_number": orig_element.metadata.to_dict()["page_number"],
                "image_mime_type": mime_type,
                "source": file_name,
                "type": "image",
            }
        )
        images.append(document)

  # Get the tables
  for element in elements:
    # file_name = element.metadata.to_dict()["filename"]
    for orig_element in element.metadata.orig_elements:
      orig_element_type = orig_element.to_dict()['type']
      if orig_element_type == 'Table':
        document = Document(
            page_content = orig_element.metadata.to_dict()["text_as_html"],
            id=uuid.uuid4().hex,
            metadata={
                "page_number": orig_element.metadata.to_dict()["page_number"],
                "source": file_name,
                "type": "table",
            }
        )
        tables.append(document)

  return texts + images + tables


def get_img_summary(llm: ChatOpenAI, base64_image: str, mime_type: str) -> str:
  message = HumanMessage(content=[
      {
          "type": "text",
          "text": "Provide a concise summary of the provided image. Ensure the summary covers all key points and main ideas, without including external information."
       },
      {
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
      },
  ])
  llm_response = llm.invoke([message])
  return llm_response.content
