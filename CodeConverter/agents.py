import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from states import CodeConvertState, manage_memory
from output_structure import ConvertStructure, FeedbackStructure
from utils import retry

#Load environment variables
load_dotenv(find_dotenv())
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "mistral-large-latest")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "mistral-large-latest")

GENERATION_MODEL_PROVIDER = os.getenv("GENERATION_MODEL_PROVIDER")
REFLECTION_MODEL_PROVIDER = os.getenv("REFLECTION_MODEL_PROVIDER")
GENERATION_MODEL_INFERENCE_ARN = os.getenv("GENERATION_MODEL_INFERENCE_ARN")
REFLECTION_MODEL_INFERENCE_ARN = os.getenv("REFLECTION_MODEL_INFERENCE_ARN")


# Delete this after testing
# with open("data/testdata4.md", "r", encoding="utf-8") as f:
#            original_code = f.read()
# state = CodeConvertState(
#                             original_code=original_code,
#                             generation=[],
#                             reflection=[],
#                             judgement='FAIL',
#                             iterations=0
#                         )
# model_name: str = GENERATION_MODEL
# temperature: float = 0.1

class CodeConverterAgent:
    """Agent responsible for converting and revising code to PySpark."""
    def __init__(self, model_name: str = "mistral-large-latest", model_provider: str = "mistralai", temperature: float = 0.1):
        self.name = "Code Converter"
        self.llm = init_chat_model(model=model_name, 
                                   model_provider=model_provider, 
                                   temperature=temperature).with_structured_output(ConvertStructure)
        # self.llm = ChatBedrock(model_id = model_id,
        #                        provider = provider,
        #                        model_kwargs={"temperature": temperature}).with_structured_output(ConvertStructure)
    
    def convert_code(self, state: CodeConvertState) -> Dict[str, Any]:
        """Convert code from DataStage to PySpark."""
        original_code = state["original_code"]
        previous_feedback = state.get("reflection", [])

        #Create prompt based on graph flow
        if previous_feedback and state["iterations"] > 0:
            latest_feedback = previous_feedback[-1] if previous_feedback else ""
            system_prompt, prompt = self._create_revision_prompt(original_code, state["generation"][-1], latest_feedback)
        else:
            system_prompt, prompt = self._create_generation_prompt(original_code)
        
        #Call the LLM to generate or revise code
        generated_code = self.llm.invoke([SystemMessage(system_prompt),
                                          HumanMessage(prompt)])

        #Update state with new generation
        new_generation = state["generation"] + [generated_code.generation]
        updated_state = {
            **state,
            "generation": new_generation
        }
        
        return manage_memory(updated_state)

    def _create_generation_prompt(self, original_code: str) -> str:
        with open("sysprompts/datastage.md", "r", encoding="utf-8") as f:
            generation_prompt = f.read()
        return f"{generation_prompt}", f"Here is the code to be converted:\n{original_code}"
    
    def _create_revision_prompt(self, original_code: str, generation: str, feedback: str) -> str:
        with open("sysprompts/revision.md", "r", encoding="utf-8") as f:
            revision_prompt = f.read()
        return f"{revision_prompt}", f"Original Code:\n{original_code}\n\Latest Generation:\n{generation}\n\nLatest Feedback:\n{feedback}"   


class ReflectionAgent:
    """Agent responsible for reflecting on code revisions."""
    def __init__(self, model_name: str = "mistral-large-latest", model_provider: str = "mistralai", temperature: float = 0.2, max_retries: int = 3):
        self.name = "Reflection Agent"
        self.llm = init_chat_model(model=model_name, 
                                   model_provider=model_provider, 
                                   temperature=temperature).with_structured_output(FeedbackStructure)
        self.max_retries = max_retries

    @retry(max_retries=3, delay=1) #Reflection agent can be unpredictable with outputs
    def reflect_on_code(self, state: CodeConvertState) -> Dict[str, Any]:
        """Reflect on the latest code generation."""
        original_code = state["original_code"]
        latest_generation = state["generation"][-1] if state["generation"] else ""
        feedback = state.get("reflection", [])

        #Create prompt for reflection
        system_prompt, prompt = self._create_reflection_prompt(original_code, feedback, latest_generation)
        
        #Call the LLM to generate reflection
        reflection = self.llm.invoke([SystemMessage(system_prompt),
                                            HumanMessage(prompt)])

        
        #Update state with new reflection
        new_reflection = state["reflection"] + [reflection.feedback]
        updated_state = {
            **state,
            "judgement": reflection.judgement,
            "reflection": new_reflection,
            "iterations": state["iterations"] + 1 
        }
        
        return manage_memory(updated_state)

    def _create_reflection_prompt(self, original_code: str, feedback:list[str], generation: str) -> str:
        with open("sysprompts/reflection.md", "r", encoding="utf-8") as f:
            reflection_prompt = f.read()
        return f"{reflection_prompt}", f"Original Code:\n{original_code}\nPrevious feedback:\n{feedback}\nPlease review the following code:\n{generation}"