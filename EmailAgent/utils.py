def make_agent_node(agent_instance):
    def node_fn(state):
        print(f"Running agent: {agent_instance.__class__.__name__}")
        return agent_instance.run(state)
    return node_fn

