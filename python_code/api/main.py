from agent_controller import AgentController
import runpod

def main():
    controller_instance = AgentController()
    runpod.serverless.start({"handler": controller_instance.get_response})


if __name__ == "__main__":
    main()