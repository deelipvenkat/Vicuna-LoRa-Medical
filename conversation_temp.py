conversation_history = """
A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""

while True:
    user_input = input("User: ")
    conversation_history += f'User: {user_input}\n'
    
    response=ask_vicuna(conversation_history)
    
    response_list=response.split(' ')

    if response_list[-1]== 'User ':
        response_list=response_list[:-1]

    for i in response_list :
        if response[0]=='AI:' or response[0]=='AI' :
            response_list=response_list[1:]
        else :
            pass    
            
    conversation_history += f'AI: {response}\n'    

    print(f'AI: {response}')

    if len(conversation_history.split(" ")) > 2048:
        conversation_history = " ".join(conversation_history.split(" ")[-2048:])
    