from dotenv import load_dotenv
import os
from .utils import get_chatbot_response, get_embedding
from openai import OpenAI
from copy import deepcopy
from pinecone import Pinecone
load_dotenv()

class DetailsAgent():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.embedding_client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"), 
            base_url=os.getenv("RUNPOD_EMBEDDING_URL")
        )
        self.model_name = os.getenv("MODEL_NAME")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
    
    def get_closest_results(self, pinecone_index_name, query_embedding, top_k=2):
        index = self.pc.Index(pinecone_index_name)
        
        results = index.query(
            namespace="ns1",
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return results

    def get_response(self, dialog_messages):
        dialog_messages = deepcopy(dialog_messages)

        user_text = dialog_messages[-1]['content']
        generated_vector = get_embedding(self.embedding_client, self.model_name, user_text)[0]
        nearest = self.get_closest_results(self.index_name, generated_vector)
        source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in nearest['matches'] ])

        prompt = f"""
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_text}
        """

        system_prompt = """ You are a customer support agent for a coffee shop called Merry's way. You should answer every question as if you are waiter and provide the neccessary information to the user regarding their orders """
        dialog_messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + dialog_messages[-3:]

        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        final_payload = self.postprocess(chatbot_output)
        return final_payload

    def postprocess(self, raw_text):
        packaged = {
            "role": "assistant",
            "content": raw_text,
            "memory": {"agent": "details_agent"
                      }
        }
        return packaged

    
