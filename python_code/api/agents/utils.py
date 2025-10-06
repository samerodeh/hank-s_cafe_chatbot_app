def get_chatbot_response(client, model_name, messages, temperature=0):
    queued_messages = []
    for entry in messages:
        queued_messages.append({"role": entry["role"], "content": entry["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=queued_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response


def get_embedding(embedding_client, model_name, text_input):
    output = embedding_client.embeddings.create(input=text_input, model=model_name)
    
    vector_list = []
    for embedding_object in output.data:
        vector_list.append(embedding_object.embedding)

    return vector_list


def double_check_json_output(client, model_name, json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    reviewed = get_chatbot_response(client, model_name, messages)

    return reviewed