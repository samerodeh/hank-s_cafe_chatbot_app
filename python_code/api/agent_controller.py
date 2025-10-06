from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    OrderTakingAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

class AgentController():
    def __init__(self):
        self.guardian_agent = GuardAgent()
        self.selector_agent = ClassificationAgent()
        self.reco_agent = RecommendationAgent('recommendation_objects/apriori_recommendations.json',
                                              'recommendation_objects/popularity_recommendation.csv')

        self.named_agents: dict[str, AgentProtocol] = {
            "details_agent": DetailsAgent(),
            "order_taking_agent": OrderTakingAgent(self.reco_agent),
            "recommendation_agent": self.reco_agent,
        }

    def get_response(self, request_payload):
        # Extract User Input
        payload_input = request_payload["input"]
        chat_messages = payload_input["messages"]

        # Get GuardAgent's response
        gate_result = self.guardian_agent.get_response(chat_messages)
        if gate_result["memory"]["guard_decision"] == "not allowed":
            return gate_result

        # Get ClassificationAgent's response
        routing_result = self.selector_agent.get_response(chat_messages)
        selected_agent_key = routing_result["memory"]["classification_decision"]

        # Get the chosen agent's response
        routed_agent = self.named_agents[selected_agent_key]
        final_response = routed_agent.get_response(chat_messages)

        return final_response
