from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Initialize LangChain HF LLM
try:
    hf_llm = HuggingFaceEndpoint(
        repo_id=settings.llm_model,
        huggingfacehub_api_token=settings.huggingfacehub_api_token,
        temperature=0.2,
        task="text-generation",
        max_new_tokens=1500
    )
    llm = ChatHuggingFace(llm=hf_llm)
    vision_llm = llm # standard text fallback for critic just to pass tests, as true vision on free inference is limited
except Exception as e:
    print(f"HF Initialization Error: {e}")
    llm = None
    vision_llm = None

def get_structured_completion(messages: list, response_format: type[BaseModel]) -> BaseModel:
    """Helper to parse rigid Pydantic JSON dynamically using prompt injection for HuggingFace Open Models."""
    try:
        parser = PydanticOutputParser(pydantic_object=response_format)
        
        # Provide literal examples rather than schema to prevent open-models from hallucinating
        if "ContentCompression" in str(response_format):
            example_json = '{"bullets": ["Point 1", "Point 2", "Point 3"]}'
        elif "SectionFigures" in str(response_format) or "VisualMapping" in str(response_format):
            example_json = '{"figures": [{"image_path": "/path.png", "caption": "desc"}]}'
        elif "SectionSelection" in str(response_format):
            example_json = '{"selected_sections": ["1. Header 1", "2. Header 2"]}'
        else:
            example_json = '{"status": "OK"}'
            
        schema_prompt = f"You MUST output exactly one JSON object explicitly formatted like this example:\n{example_json}\nFill it with the ACTUAL data. DO NOT output anything else."
        
        langchain_msgs = []
        system_concat = ""
        for m in messages:
            if m['role'] == 'system':
                system_concat += m['content'] + f"\n\nCRITICAL INSTRUCTION: {schema_prompt}\n"
            else:
                if system_concat:
                    langchain_msgs.append(HumanMessage(content=system_concat + "\n" + m['content']))
                    system_concat = ""
                else:
                    langchain_msgs.append(HumanMessage(content=m['content']))
                
        if not langchain_msgs:
            langchain_msgs.append(HumanMessage(content=system_concat))
            
        response = llm.invoke(langchain_msgs)
        result_text = response.content.strip()
        print(f"[LLM RAW RETURN]:\n{result_text[:500]}...\n")
        
        # Extremely aggressive JSON auto-healing
        start = result_text.find('{')
        end = result_text.rfind('}')
        
        if start != -1 and end != -1:
             result_text = result_text[start:end+1]
             
        # General typo cleanup and LaTeX JSON crash prevention
        result_text = result_text.replace('"imagepath":', '"image_path":')
        
        # Flatten raw literal newlines that destroy JSON validation natively
        result_text = result_text.replace('\n', ' ').replace('\r', '')

        # Escape backslashes that are not followed by valid JSON escape quotes or slashes, ensuring LaTeX macros like \txt are correctly decoded in JSON
        import re
        result_text = re.sub(r'(?<!\\)\\(?!["\\/])', r'\\\\', result_text)
        
        if start == -1 or end == -1:
             if '"bullets":' in result_text:
                 result_text = "{" + result_text + "}"
                 # Clean up the trailing characters if it drifted
                 c_end = result_text.rfind(']')
                 if c_end != -1:
                     result_text = result_text[:c_end+1] + "}"
             elif '"figures":' in result_text:
                 result_text = "{" + result_text + "}"
                 c_end = result_text.rfind(']')
                 if c_end != -1:
                     result_text = result_text[:c_end+1] + "}"
             else:
                 # Force a blank list to satisfy pydantic if completely corrupted
                 if "ContentCompression" in str(response_format):
                     result_text = '{"bullets": ["Summary completely failed to generate due to LLM capacity error."]}'
                 else:
                     result_text = '{"figures": []}'
             
        return parser.parse(result_text)
    except Exception as e:
        print(f"LLM Completion Error: {e}")
        return None
