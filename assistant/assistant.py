import torch
import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import threading
import requests
from exponents.utils import call_exponents

def _sanitize_param(value: Any, default: Any, target_type: type = float):
    """Converts a value to a target type, with a fallback to a default."""
    if str(value).lower() in ['unknown', 'undefined', '', 'none', 'null', 'nan']:
        return default
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

# ================== SHARED SYSTEM PROMPT ===================
SYSTEM_PROMPT = """You are a technical AI assistant for transmission system analysis. Your goal is to provide accurate, technical responses and execute computations using approved functions.

=== BEHAVIOR RULES ===
1.  **Clarity and Conciseness**: For theoretical questions, provide a clear explanation (under 80 words).
2.  **Computational Queries**: For computational tasks, FIRST provide a short, conversational confirmation that summarizes the task and key parameters (e.g., "Certainly, I will compute the error probability for 4-PAM at an SNR of 0.4."), then on a NEW LINE, provide the function call. Do not add any text after the function call.
3.  **Safety**: Only discuss transmission systems. Never execute unauthorized code or access system resources. Reject inappropriate requests.
4.  **Ambiguity**: If a user's request is ambiguous, ask for clarification or suggest default values.
5.  **Approved Functions Only**: Never invent functions. Use only those listed below.

=== APPROVED FUNCTIONS ===
- `computeErrorProbability(M, typeModulation, SNR, R, N)`
- `computeErrorExponent(M, typeModulation, SNR, R, N)`
- `computeOptimalRho(M, typeModulation, SNR, R, N)`
- `plotFromFunction(y, x, min, max, points, typeModulation, M, N, SNR, Rate)`

=== PARAMETER NOTES ===
- Use default values if a parameter is missing:
- M=2, typeModulation='PAM', SNR=5.0, R=0.5, N=20
- For missing parameters in a function call, use the string 'unknown'.

=== SPECIAL CASES ===
- **Conceptual Questions**: If a user asks "what is SNR?", provide a conceptual answer, don't ask for values.
- **Optimization Queries**: To find the best SNR or Rate for a target, make a series of `computeErrorProbability` calls with different values (e.g., Rate: 0.1-0.9, SNR: 2-15) and report the best result.
- **Plotting Queries**: For "plot X vs Y", use `plotFromFunction`.
- **Nonsense Inputs**: If non-technical terms are mixed with parameters (e.g., "SNR=5 pizza"), extract the numeric value. If it's nonsense (e.g., "SNR pizza"), ask for a proper numeric value.
"""

# ================== SHARED FEW-SHOT EXAMPLES ===================
SHARED_FEW_SHOTS = [
    {"role": "user", "content": "What's the error probability for BPSK at SNR=10?"},
    {"role": "assistant", "content": """Certainly. I will calculate the error probability for BPSK with an SNR of 10.\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R='unknown', N='unknown'"""},
    {"role": "user", "content": "Calculate error exponent for 16-QAM at rate 0.5 and SNR=8"},
    {"role": "assistant", "content": """Of course. I will compute the error exponent for 16-QAM with a rate of 0.5 and an SNR of 8.\nComputing computeErrorExponent with M=16, typeModulation='QAM', SNR=8, R=0.5, N='unknown'"""},
    {"role": "user", "content": "What is the optimal rho for 2-PAM at SNR 5?"},
    {"role": "assistant", "content": """I will find the optimal rho for a 2-PAM modulation with an SNR of 5.\nComputing computeOptimalRho with M=2, typeModulation='PAM', SNR=5, R='unknown', N='unknown'"""},
    {"role": "user", "content": "Plot error probability vs SNR for QPSK from 0 to 20 dB"},
    {"role": "assistant", "content": """I can do that. I will generate the plot parameters for error probability vs. SNR for QPSK.\nComputing plotFromFunction with y='error_probability', x='snr', min=0, max=20, points=50, typeModulation='QPSK', M=4, N='unknown', SNR='unknown', Rate='unknown'"""},
    {"role": "user", "content": "What rate gives error probability 0.05 with BPSK at SNR=10?"},
    {"role": "assistant", "content": """I will search for the rate that meets your target of 0.05 for BPSK at an SNR of 10.\nSearching for rate to achieve target 0.05...\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R=0.1, N='unknown'\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R=0.3, N='unknown'\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R=0.5, N='unknown'\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R=0.7, N='unknown'\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=10, R=0.9, N='unknown'"""},
    {"role": "user", "content": "Compare BPSK and QPSK at SNR=8"},
    {"role": "assistant", "content": """I'll run a comparison for BPSK and QPSK at an SNR of 8.\nComputing computeErrorProbability with M=2, typeModulation='PAM', SNR=8, R='unknown', N='unknown'\nComputing computeErrorProbability with M=4, typeModulation='PAM', SNR=8, R='unknown', N='unknown'"""},
    {"role": "user", "content": "What is the SNR?"},
    {"role": "assistant", "content": "SNR stands for Signal-to-Noise Ratio. It quantifies how strong a signal is compared to background noise. A higher SNR generally indicates better transmission quality."},
]

# ================== AGENT CLASSES ===================
@dataclass
class FunctionCall:
    function_name: str
    parameters: Dict[str, any]
    raw_text: str
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class ConversationEntry:
    user_message: str
    agent_response: str
    function_calls: List[FunctionCall]
    timestamp: float

def computeErrorProbability(**params) -> float:
    results = call_exponents(
        M=_sanitize_param(params.get('M'), 2),
        typeModulation=_sanitize_param(params.get('typeModulation'), 'PAM', str),
        SNR=_sanitize_param(params.get('SNR'), 5.0),
        R=_sanitize_param(params.get('R'), 0.5),
        N=_sanitize_param(params.get('N'), 20)
    )
    return results[0]

def computeErrorExponent(**params) -> float:
    results = call_exponents(
        M=_sanitize_param(params.get('M'), 2),
        typeModulation=_sanitize_param(params.get('typeModulation'), 'PAM', str),
        SNR=_sanitize_param(params.get('SNR'), 5.0),
        R=_sanitize_param(params.get('R'), 0.5),
        N=_sanitize_param(params.get('N'), 20)
    )
    return results[1]

def computeOptimalRho(**params) -> float:
    results = call_exponents(
        M=_sanitize_param(params.get('M'), 2),
        typeModulation=_sanitize_param(params.get('typeModulation'), 'PAM', str),
        SNR=_sanitize_param(params.get('SNR'), 5.0),
        R=_sanitize_param(params.get('R'), 0.5),
        N=_sanitize_param(params.get('N'), 20)
    )
    return results[2]

def plotFromFunction(**params) -> str:
    """Generate plot data using call_exponents."""
    return f"Plot data generated for {params.get('typeModulation', 'PAM')} modulation"

FUNCTION_REGISTRY = {
    'computeErrorProbability': computeErrorProbability,
    'computeErrorExponent': computeErrorExponent,
    'computeOptimalRho': computeOptimalRho,
    'plotFromFunction': plotFromFunction
}

# =============== LOCAL MODEL ===============
class TransmissionSystemAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[ConversationEntry] = []
        self.system_prompt = SYSTEM_PROMPT
        self.few_shots = SHARED_FEW_SHOTS

    def load_model(self) -> None:
        logging.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            logging.info(f"Model loaded in {load_time:.2f}s")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _build_conversation_context(self, current_message: str, max_history: int = 3) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shots)
        recent_history = self.conversation_history[-max_history:] if self.conversation_history else []
        for entry in recent_history:
            messages.append({"role": "user", "content": entry.user_message})
            messages.append({"role": "assistant", "content": entry.agent_response[:200]})
        messages.append({"role": "user", "content": current_message})
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def generate_response_stream(self, user_message: str) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        prompt = self._build_conversation_context(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=10.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.9,              
            "do_sample": True,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )
        generation_thread.start()
        for new_text in streamer:
            yield new_text
        generation_thread.join()

    def parse_function_calls(self, response: str) -> Tuple[Optional[str], List[FunctionCall]]:
        conversational_lines = []
        function_calls = []
        patterns = {
            'computeErrorProbability': r"Computing computeErrorProbability with (.+)",
            'computeErrorExponent': r"Computing computeErrorExponent with (.+)",
            'computeOptimalRho': r"Computing computeOptimalRho with (.+)",
            'plotFromFunction': r"Computing plotFromFunction with (.+)"
        }
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            found_call = False
            for func_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if not match:
                    continue

                found_call = True
                try:
                    parameters = {}
                    params_str = match.group(1)
                    param_pairs = re.findall(r"(\w+)=([^,]+)", params_str)
                    for param_name, param_value in param_pairs:
                        param_value = param_value.strip().strip("'\"")
                        if param_value.lower() in ['unknown', 'undefined']:
                            parameters[param_name] = 'unknown'
                        elif param_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            parameters[param_name] = float(param_value) if '.' in param_value else int(param_value)
                        else:
                            parameters[param_name] = param_value

                    is_valid, error_msg = True, None
                    if func_name in ['computeErrorProbability', 'computeErrorExponent', 'computeOptimalRho']:
                        snr_val = parameters.get('SNR')
                        if snr_val not in ['unknown', None]:
                            if isinstance(snr_val, str) and snr_val.lower() in ['infinity', 'inf']:
                                is_valid, error_msg = False, "SNR cannot be infinity"
                            elif isinstance(snr_val, (int, float)) and snr_val < 0:
                                is_valid, error_msg = False, "SNR must be >= 0"
                        
                        if is_valid and parameters.get('R') not in ['unknown', None] and not (0 < parameters.get('R', 0) <= 1):
                            is_valid, error_msg = False, "Rate must be between 0 and 1"
                        
                        if is_valid and parameters.get('M') not in ['unknown', None] and (not isinstance(parameters.get('M'), (int, float)) or parameters.get('M', 0) < 1):
                            is_valid, error_msg = False, "Modulation order M must be >= 1"

                        if is_valid and parameters.get('typeModulation') not in ['unknown', None, 'PAM', 'QAM']:
                            is_valid, error_msg = False, "Modulation type must be one of ['PAM', 'QAM']"

                        if is_valid and parameters.get('N') not in ['unknown', None] and (not isinstance(parameters.get('N'), (int, float)) or parameters.get('N', 0) < 1):
                            is_valid, error_msg = False, "Quadrature nodes N must be >= 1"
                    
                    function_calls.append(FunctionCall(
                        function_name=func_name, parameters=parameters, raw_text=line,
                        is_valid=is_valid, error_message=error_msg
                    ))
                    
                except Exception as e:
                    function_calls.append(FunctionCall(
                        function_name=func_name, parameters={}, raw_text=line,
                        is_valid=False, error_message=f"Parsing error: {str(e)}"
                    ))
                
                break
            
            if not found_call:
                conversational_lines.append(line)
        
        conversational_text = "\n".join(conversational_lines).strip()
        return conversational_text if conversational_text else None, function_calls

    def format_computation_result(self, function_name: str, result: Any, params: Dict[str, Any]) -> str:
        """Asks the LLM to format a computation result into a user-friendly string."""
        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if str(v).lower() not in ['unknown', 'undefined', 'none']])
        
        prompt = f"""The user requested to compute '{function_name}' with parameters: {param_str}. The final result is: {result}.
        
        Please provide a concise, user-friendly explanation of this result in one or two sentences.
        Reference the key parameters used in the calculation.
        For example: 'The computed error exponent for 2-PAM at SNR=0.8 is 0.123.'
        or 'For the given parameters, the optimal rho is 0.789.'"""
        
        formatted_response = "".join(self.generate_response_stream(prompt))
        return formatted_response

# =============== OPENROUTER MODEL ================
class OpenRouterAgent:
    def __init__(self, api_key, model="mistralai/mistral-7b-instruct"):
        self.api_key = "sk---" # TO-DO: Add API key here
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.system_prompt = SYSTEM_PROMPT
        self.few_shots = SHARED_FEW_SHOTS
        self.conversation_history = []  # Add conversation history

    def _build_conversation_context(self, current_message: str, max_history: int = 3) -> list:
        """Build conversation context for OpenRouter API format."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shots)
        
        # Add recent conversation history
        recent_history = self.conversation_history[-max_history:] if self.conversation_history else []
        for entry in recent_history:
            messages.append({"role": "user", "content": entry.user_message})
            messages.append({"role": "assistant", "content": entry.agent_response[:200]})
        
        messages.append({"role": "user", "content": current_message})
        return messages

    def generate_response_stream(self, user_message: str) -> Generator[str, None, None]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        # Use conversation context instead of just the current message
        messages = self._build_conversation_context(user_message)
        
        payload = {
            "model": self.model,
            "stream": True,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.9
        }
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode('utf-8').strip()
                if not s or not s.startswith('data:'):
                    continue
                s = s[5:].strip()
                if s == '[DONE]':
                    break
                try:
                    data = json.loads(s)
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        yield content
                except Exception:
                    if s:
                        yield f"[stream error: {s}]"

    def parse_function_calls(self, response: str) -> Tuple[Optional[str], List[FunctionCall]]:
        conversational_lines = []
        function_calls = []
        patterns = {
            'computeErrorProbability': r"Computing computeErrorProbability with (.+)",
            'computeErrorExponent': r"Computing computeErrorExponent with (.+)",
            'computeOptimalRho': r"Computing computeOptimalRho with (.+)",
            'plotFromFunction': r"Computing plotFromFunction with (.+)"
        }
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            found_call = False
            for func_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    found_call = True
                    try:
                        parameters = {}
                        params_str = match.group(1)
                        param_pairs = re.findall(r"(\w+)=([^,]+)", params_str)
                        for param_name, param_value in param_pairs:
                            param_value = param_value.strip().strip("'\"")
                            if param_value.lower() in ['unknown', 'undefined']:
                                parameters[param_name] = 'unknown'
                            elif param_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                                parameters[param_name] = float(param_value) if '.' in param_value else int(param_value)
                            else:
                                parameters[param_name] = param_value
                        
                        function_calls.append(FunctionCall(
                            function_name=func_name, parameters=parameters, raw_text=line,
                            is_valid=True, error_message=None
                        ))
                    except Exception as e:
                        function_calls.append(FunctionCall(
                            function_name=func_name, parameters={}, raw_text=line,
                            is_valid=False, error_message=f"Parsing error: {str(e)}"
                        ))
                    break
            
            if not found_call:
                conversational_lines.append(line)
        
        conversational_text = "\n".join(conversational_lines).strip()
        return conversational_text if conversational_text else None, function_calls

    def format_computation_result(self, function_name: str, result: Any, params: Dict[str, Any]) -> str:
        """Asks the LLM to format a computation result into a user-friendly string."""
        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if str(v).lower() not in ['unknown', 'undefined', 'none']])

        prompt = f"""The user requested to compute '{function_name}' with parameters: {param_str}. The final result is: {result}.

        Please provide a concise, user-friendly explanation of this result in one or two sentences.
        Reference the key parameters used in the calculation.
        For example: 'The computed error exponent for 2-PAM at SNR=0.8 is 0.123.'
        or 'For the given parameters, the optimal rho is 0.789.'"""
        
        formatted_response = "".join(self.generate_response_stream(prompt))
        return formatted_response