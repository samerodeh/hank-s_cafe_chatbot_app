from agents import (
    GuardAgent,
    ClassificationAgent,
    DetailsAgent,
    OrderTakingAgent,
    RecommendationAgent,
    AgentProtocol,
)
import os

def main():
    guardian_agent = GuardAgent()
    selector_agent = ClassificationAgent()
    reco_agent = RecommendationAgent(
        'recommendation_objects/apriori_recommendations.json',
        'recommendation_objects/popularity_recommendation.csv',
    )
    available_agents: dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "order_taking_agent": OrderTakingAgent(reco_agent),
        "recommendation_agent": reco_agent,
    }

    chat_history = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n\nPrint Messages ...............")
        for entry in chat_history:
            print(f"{entry['role'].capitalize()}: {entry['content']}")

        # Get user input
        user_prompt = input("User: ")
        chat_history.append({"role": "user", "content": user_prompt})

        # Get GuardAgent's response
        guard_result = guardian_agent.get_response(chat_history)
        if guard_result["memory"]["guard_decision"] == "not allowed":
            chat_history.append(guard_result)
            continue

        # Get ClassificationAgent's response
        routing_result = selector_agent.get_response(chat_history)
        chosen_key = routing_result["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_key)

        # Get the chosen agent's response
        chosen_agent = available_agents[chosen_key]
        agent_reply = chosen_agent.get_response(chat_history)

        chat_history.append(agent_reply)


if __name__ == "__main__":
    main()
