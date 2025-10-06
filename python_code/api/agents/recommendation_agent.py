import json
import pandas as pd
import os
from .utils import get_chatbot_response
from openai import OpenAI
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()


class RecommendationAgent():
    def __init__(self, apriori_recommendation_path, popular_recommendation_path):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        with open(apriori_recommendation_path, 'r') as file:
            self.apriori_recommendations = json.load(file)

        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = self.popular_recommendations['product_category'].tolist()
    
    def get_apriori_recommendation(self, purchased_items, top_k=5):
        candidate_rules = []
        for purchased in purchased_items:
            if purchased in self.apriori_recommendations:
                candidate_rules += self.apriori_recommendations[purchased]
        
        # Sort recommendation list by "confidence"
        candidate_rules = sorted(candidate_rules, key=lambda x: x['confidence'], reverse=True)

        shortlisted = []
        per_category_counter = {}
        for rec in candidate_rules:
            # If Duplicated recommendations then skip
            if rec in shortlisted:
                continue 

            # Limit 2 recommendations per category
            product_catory = rec['product_category']
            if product_catory not in per_category_counter:
                per_category_counter[product_catory] = 0
            
            if per_category_counter[product_catory] >= 2:
                continue

            per_category_counter[product_catory] += 1

            # Add recommendation
            shortlisted.append(rec['product'])

            if len(shortlisted) >= top_k:
                break

        return shortlisted 

    def get_popular_recommendation(self, product_categories=None, top_k=5):
        recommendations_df = self.popular_recommendations
        
        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendations_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        recommendations_df = recommendations_df.sort_values(by='number_of_transactions', ascending=False)
        
        if recommendations_df.shape[0] == 0:
            return []

        suggestions = recommendations_df['product'].tolist()[:top_k]
        return suggestions

    def recommendation_classification(self, dialog_messages):
        system_prompt = """ You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

        1. Apriori Recommendations: These are recommendations based on the user's order history. We recommend items that are frequently bought together with the items in the user's order.
        2. Popular Recommendations: These are recommendations based on the popularity of items in the coffee shop. We recommend items that are popular among customers.
        3. Popular Recommendations by Category: Here the user asks to recommend them product in a category. Like what coffee do you recommend me to get?. We recommend items that are popular in the category of the user's requested category.
        
        Here is the list of items in the coffee shop:
        """+ ",".join(self.products) + """
        Here is the list of Categories we have in the coffee shop:
        """ + ",".join(self.product_categories) + """

        Your task is to determine which type of recommendation to provide based on the user's message.

        Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
        {
        "chain of thought": Write down your critical thinking about what type of recommendation is this input relevant to.
        "recommendation_type": "apriori" or "popular" or "popular by category". Pick one of those and only write the word.
        "parameters": This is a  python list. It's either a list of of items for apriori recommendations or a list of categories for popular by category recommendations. Leave it empty for popular recommendations. Make sure to use the exact strings from the list of items and categories above.
        }
        """

        input_messages = [{"role": "system", "content": system_prompt}] + dialog_messages[-3:]

        model_text = get_chatbot_response(self.client, self.model_name, input_messages)
        parsed = self.postprocess_classfication(model_text)
        return parsed

    def get_response(self, dialog_messages):
        dialog_messages = deepcopy(dialog_messages)

        rec_type_payload = self.recommendation_classification(dialog_messages)
        rec_kind = rec_type_payload['recommendation_type']
        picked: list[str] = []
        if rec_kind == "apriori":
            picked = self.get_apriori_recommendation(rec_type_payload['parameters'])
        elif rec_kind == "popular":
            picked = self.get_popular_recommendation()
        elif rec_kind == "popular by category":
            picked = self.get_popular_recommendation(rec_type_payload['parameters'])
        
        if picked == []:
            return {"role": "assistant", "content": "Sorry, I can't help with that. Can I help you with your order?"}
        
        # Respond to User
        readable = ", ".join(picked)
        
        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. And put it an unordered list with a very small description.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {dialog_messages[-1]['content']}

        Please recommend me those items exactly: {readable}
        """

        dialog_messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + dialog_messages[-3:]

        model_reply = get_chatbot_response(self.client, self.model_name, input_messages)
        packaged = self.postprocess(model_reply)

        return packaged



    def postprocess_classfication(self, raw_text):
        parsed = json.loads(raw_text)

        shaped = {
            "recommendation_type": parsed['recommendation_type'],
            "parameters": parsed['parameters'],
        }
        return shaped
    
    def get_recommendations_from_order(self, dialog_messages, order_items):
        purchased = []
        for product in order_items:
            purchased.append(product['item'])

        picked = self.get_apriori_recommendation(purchased)
        readable = ", ".join(picked)

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their order.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {dialog_messages[-1]['content']}

        Please recommend me those items exactly: {readable}
        """

        dialog_messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + dialog_messages[-3:]

        model_reply = get_chatbot_response(self.client, self.model_name, input_messages)
        packaged = self.postprocess(model_reply)

        return packaged
    
    def postprocess(self, raw_text):
        packaged = {
            "role": "assistant",
            "content": raw_text,
            "memory": {"agent": "recommendation_agent"
                      }
        }
        return packaged


