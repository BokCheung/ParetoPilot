"""Optional local extension; uses no API or network. Default config uses DummyAgent directly."""
from paretopilot.agent import DummyAgent


def create_agent(options):
    return DummyAgent()
