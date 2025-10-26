import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import time
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class AgentContext:
    agent_id: str
    role: AgentRole
    capabilities: List[str]
    memory: List[Message]
    tools: List[str]


class MCPAgent:
    """Representa a un agente"""

    def __init__(self, agent_id: str, role: AgentRole, api_key: str = None):
        self.agent_id = agent_id
        self.role = role
        self.api_key = api_key
        self.memory = []
        self.context = AgentContext(
            agent_id=agent_id,
            role=role,
            capabilities=self._init_capabilities(),
            memory=[],
            tools=self._init_tools()
        )

        try:
            genai.configure(api_key=api_key)               
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            print(f"Gemini configuration failed: {e}")
            exit(1)
        

    def _init_capabilities(self) -> List[str]:
        """Inicializar las capabilities de cada rol"""
        capabilities_map = {
            AgentRole.COORDINATOR: ["task_decomposition", "agent_orchestration", "priority_management"],
            AgentRole.RESEARCHER: ["data_gathering", "web_search", "information_synthesis"],
            AgentRole.ANALYZER: ["pattern_recognition", "data_analysis", "insight_generation"],
            AgentRole.EXECUTOR: ["action_execution", "result_validation", "output_formatting"]
        }
        return capabilities_map.get(self.role, [])

    def _init_tools(self) -> List[str]:
        """Inicializar tools disponibles para cada rol"""
        tools_map = {
            AgentRole.COORDINATOR: ["task_splitter", "agent_selector", "progress_tracker"],
            AgentRole.RESEARCHER: ["search_engine", "data_extractor", "source_validator"],
            AgentRole.ANALYZER: ["statistical_analyzer", "pattern_detector", "visualization_tool"],
            AgentRole.EXECUTOR: ["code_executor", "file_handler", "api_caller"]
        }
        return tools_map.get(self.role, [])

    def process_message(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Procesa un mensaje con conciencia del contexto"""

        msg = Message(
            role="user",
            content=message,
            timestamp=datetime.now(),
            metadata=context
        )
        self.memory.append(msg)

        prompt = self._generate_contextual_prompt(message, context)

        try:
            response = self._generate_response_gemini(prompt)

            response_msg = Message(
                role="assistant",
                content=response,
                timestamp=datetime.now(),
                metadata={"agent_id": self.agent_id, "role": self.role.value}
            )
            self.memory.append(response_msg)

            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "response": response,
                "capabilities_used": self._analyze_capabilities_used(message),
                "next_actions": self._suggest_next_actions(response),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"error": str(e)}

    def _generate_response_gemini(self, prompt: str) -> str:
        """Llama a la API de Gemini para obtener una inferencia"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            exit(1)


    def _generate_contextual_prompt(self, message: str, context: Optional[Dict]) -> str:
        """Genera un prompt contextualizado basado en el rol del agente y el mensaje"""

        base_prompt = f"""
        You are an advanced AI agent with the role: {self.role.value}
        Your capabilities: {', '.join(self.context.capabilities)}
        Available tools: {', '.join(self.context.tools)}

        Recent conversation context:
        {self._get_recent_context()}

        Current request: {message}
        """

        role_instructions = {
            AgentRole.COORDINATOR: """
            Focus on breaking down complex tasks, coordinating with other agents,
            and maintaining overall project coherence. Consider dependencies and priorities.
            Provide clear task decomposition and agent assignments.
            """,
            AgentRole.RESEARCHER: """
            Prioritize accurate information gathering, source verification,
            and comprehensive data collection. Synthesize findings clearly.
            Focus on current trends and reliable sources.
            """,
            AgentRole.ANALYZER: """
            Focus on pattern recognition, data interpretation, and insight generation.
            Provide evidence-based conclusions and actionable recommendations.
            Highlight key correlations and implications.
            """,
            AgentRole.EXECUTOR: """
            Concentrate on practical implementation, result validation,
            and clear output delivery. Ensure actions are completed effectively.
            Focus on quality and completeness of execution.
            """
        }

        return base_prompt + role_instructions.get(self.role, "")

    def _get_recent_context(self, limit: int = 3) -> str:
        """Get recent conversation context"""
        if not self.memory:
            return "No previous context"

        recent = self.memory[-limit:]
        context_str = ""
        for msg in recent:
            context_str += f"{msg.role}: {msg.content[:100]}...\n"
        return context_str

    def _analyze_capabilities_used(self, message: str) -> List[str]:
        """Analiza qué capacidades fueron utilizadas"""
        used_capabilities = []
        message_lower = message.lower()

        capability_keywords = {
            "task_decomposition": ["break down", "divide", "split", "decompose"],
            "data_gathering": ["research", "find", "collect", "gather"],
            "pattern_recognition": ["analyze", "pattern", "trend", "correlation"],
            "action_execution": ["execute", "run", "implement", "perform"],
            "agent_orchestration": ["coordinate", "manage", "organize", "assign"],
            "information_synthesis": ["synthesize", "combine", "merge", "integrate"]
        }

        for capability, keywords in capability_keywords.items():
            if capability in self.context.capabilities:
                if any(keyword in message_lower for keyword in keywords):
                    used_capabilities.append(capability)

        return used_capabilities

    def _suggest_next_actions(self, response: str) -> List[str]:
        """Sugiere acciones lógicas a seguir basadas en la respuesta"""
        suggestions = []
        response_lower = response.lower()

        if "need more information" in response_lower or "research" in response_lower:
            suggestions.append("delegate_to_researcher")
        if "analyze" in response_lower or "pattern" in response_lower:
            suggestions.append("delegate_to_analyzer")
        if "implement" in response_lower or "execute" in response_lower:
            suggestions.append("delegate_to_executor")
        if "coordinate" in response_lower or "manage" in response_lower:
            suggestions.append("initiate_multi_agent_collaboration")
        if "subtask" in response_lower or "break down" in response_lower:
            suggestions.append("task_decomposition_required")

        return suggestions if suggestions else ["continue_conversation"]
     




class MCPAgentSwarm:
    """Sistema de coordinación multi-agente (enjambre)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.agents = {}
        self.task_history = []
        self.results = {}

    def create_agent(self, agent_id: str, role: AgentRole) -> MCPAgent:
        """Crea y registra un nuevo agente"""
        agent = MCPAgent(agent_id, role, self.api_key)
        self.agents[agent_id] = agent
        print(f"Creado agente con id {agent_id} y rol {role.value}")
        return agent

    def coordinate_task(self, task: str) -> Dict[str, Any]:
        """Coordina tareas complejas entre múltiples agentes"""

        print(f"\nCoordinando tarea: {task}")
        print("=" * 60)

        if "coordinator" not in self.agents:
            self.create_agent("coordinator", AgentRole.COORDINATOR)

        coordinator = self.agents["coordinator"]

        print("\nPaso 1: Descomposición de Tareas")
        decomposition = coordinator.process_message(
            f"Decompose this complex task into subtasks and identify which specialized agents are needed: {task}"
        )
        print(f"✅ Coordinador: {decomposition['response']}")

        self._ensure_required_agents()

        print("\nPaso 2: Colaboración entre Agentes")
        results = {}
        for agent_id, agent in self.agents.items():
            if agent_id != "coordinator":
                print(f"\n{agent_id.upper()} trabajando...")
                result = agent.process_message(
                    f"Handle your specialized part of this task: {task}\n"
                    f"Coordinator's guidance: {decomposition['response'][:5000]}..."
                    f"I expect a response with the information you have, dont ask for more information.\n"
                )
                results[agent_id] = result
                print(f"✅ Respuesta de {agent_id}: {result['response'][:5000]}...")
        
        print("\nPaso 3: Sintesis final")
        agents_responses = ""
        for k, v in results.items():
            agents_responses += f"\n- Answer from agent {k}: {v['response'][:5000]}..."
        final_result = coordinator.process_message(
            f"Synthesize these agent results into a comprehensive final output for the task '{task}':\n"
            f"Results summary: {agents_responses}"
        )
        print(f"✅ Resultado Final: {final_result['response']}")

        task_record = {
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "final_synthesis": final_result,
        }
        self.task_history.append(task_record)

        return task_record

    def _ensure_required_agents(self):
        """Se asegura de que existan todos los tipos de agentes requeridos"""
        required_roles = [AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.EXECUTOR]

        for role in required_roles:
            agent_id = role.value
            if agent_id not in self.agents:
                self.create_agent(agent_id, role)

    def get_swarm_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del enjambre de agentes"""
        return {
            "total_agents": len(self.agents),
            "tasks_completed": len(self.task_history),
            "last_task": self.task_history[-1]["task"] if self.task_history else "None"
        }
    


def demo():
    
    # ---------------------------------------------------
    # ---------------------------------------------------
    API_KEY = "xxxxxxxxxxxxx"
    # ---------------------------------------------------
    # ---------------------------------------------------


    if not API_KEY:
        print("No hay API key configurada, imposible continuar.")
        exit(1)

    swarm = MCPAgentSwarm(API_KEY)

    complex_task = """
    Analyze the impact of AI agents on software development productivity.
    Include research on current tools, performance metrics, future predictions,
    and provide actionable recommendations for development teams.
    """

    swarm.coordinate_task(complex_task)


if __name__ == "__main__":
    demo()
