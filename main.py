from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Literal, Any, Dict
from pydantic import BaseModel
import os
import ctypes
import numpy as np
import math
import logging

# Importar mòdul assistant
import sys

sys.path.append("assistant")
from assistant.assistant import OpenRouterAgent, TransmissionSystemAgent, plotFromFunction, plotContour

API_KEY = os.environ.get('API_KEY')  # Make sure this is set in your environment
openrouter_agent = OpenRouterAgent(api_key=API_KEY)
local_agent = TransmissionSystemAgent()

# Global conversation store to maintain context
conversation_store = {}

# To make them accessible in the /compute endpoint
async def computeErrorProbability(**params) -> Any:
    """Compute error probability by calling the /exponents endpoint."""
    # Check if any parameter is an array (for comparisons)
    param_arrays = {k: v for k, v in params.items() if isinstance(v, list)}
    
    if param_arrays:
        # Handle comparison case - multiple scenarios
        # Get the length of the first array to determine number of scenarios
        first_array = next(iter(param_arrays.values()))
        num_scenarios = len(first_array)
        
        results = []
        for i in range(num_scenarios):
            # Build parameters for this scenario
            scenario_params = {}
            for key, value in params.items():
                if isinstance(value, list):
                    scenario_params[key] = value[i] if i < len(value) else value[0]  # Use first value as fallback
                else:
                    scenario_params[key] = value
            
            # Convert parameters to query string format
            query_params = {
                'M': scenario_params.get('M', 2),
                'typeM': scenario_params.get('typeModulation', 'PAM'),
                'SNR': scenario_params.get('SNR', 1.0),
                'R': scenario_params.get('R', 0.5),
                'N': scenario_params.get('N', 20),
                'n': scenario_params.get('n', 128),
                'th': scenario_params.get('th', 1e-6)
            }
            
            # Call the /exponents endpoint internally
            result = await exponents(**query_params)
            results.append({
                "scenario": i + 1,
                "parameters": scenario_params,
                "error_probability": result["Probabilidad de error"]
            })
        
        return results
    else:
        # Single computation case (original behavior)
        # Convert parameters to query string format
        query_params = {
            'M': params.get('M', 2),
            'typeM': params.get('typeModulation', 'PAM'),
            'SNR': params.get('SNR', 1.0),
            'R': params.get('R', 0.5),
            'N': params.get('N', 20),
            'n': params.get('n', 128),
            'th': params.get('th', 1e-6)
        }
        
        # Call the /exponents endpoint internally
        result = await exponents(**query_params)
        return result["Probabilidad de error"]

async def computeErrorExponent(**params) -> Any:
    """Compute error exponent by calling the /exponents endpoint."""
    # Check if any parameter is an array (for comparisons)
    param_arrays = {k: v for k, v in params.items() if isinstance(v, list)}
    
    if param_arrays:
        # Handle comparison case - multiple scenarios
        # Get the length of the first array to determine number of scenarios
        first_array = next(iter(param_arrays.values()))
        num_scenarios = len(first_array)
        
        results = []
        for i in range(num_scenarios):
            # Build parameters for this scenario
            scenario_params = {}
            for key, value in params.items():
                if isinstance(value, list):
                    scenario_params[key] = value[i] if i < len(value) else value[0]  # Use first value as fallback
                else:
                    scenario_params[key] = value
            
            # Convert parameters to query string format
            query_params = {
                'M': scenario_params.get('M', 2),
                'typeM': scenario_params.get('typeModulation', 'PAM'),
                'SNR': scenario_params.get('SNR', 1.0),
                'R': scenario_params.get('R', 0.5),
                'N': scenario_params.get('N', 20),
                'n': scenario_params.get('n', 128),
                'th': scenario_params.get('th', 1e-6)
            }
            
            # Call the /exponents endpoint internally
            result = await exponents(**query_params)
            results.append({
                "scenario": i + 1,
                "parameters": scenario_params,
                "error_exponent": result["error_exponent"]
            })
        
        return results
    else:
        # Single computation case (original behavior)
        # Convert parameters to query string format
        query_params = {
            'M': params.get('M', 2),
            'typeM': params.get('typeModulation', 'PAM'),
            'SNR': params.get('SNR', 1.0),
            'R': params.get('R', 0.5),
            'N': params.get('N', 20),
            'n': params.get('n', 128),
            'th': params.get('th', 1e-6)
        }
        
        # Call the /exponents endpoint internally
        result = await exponents(**query_params)
        return result["error_exponent"]

async def computeOptimalRho(**params) -> Any:
    """Compute optimal rho by calling the /exponents endpoint."""
    # Check if any parameter is an array (for comparisons)
    param_arrays = {k: v for k, v in params.items() if isinstance(v, list)}
    
    if param_arrays:
        # Handle comparison case - multiple scenarios
        # Get the length of the first array to determine number of scenarios
        first_array = next(iter(param_arrays.values()))
        num_scenarios = len(first_array)
        
        results = []
        for i in range(num_scenarios):
            # Build parameters for this scenario
            scenario_params = {}
            for key, value in params.items():
                if isinstance(value, list):
                    scenario_params[key] = value[i] if i < len(value) else value[0]  # Use first value as fallback
                else:
                    scenario_params[key] = value
            
            # Convert parameters to query string format
            query_params = {
                'M': scenario_params.get('M', 2),
                'typeM': scenario_params.get('typeModulation', 'PAM'),
                'SNR': scenario_params.get('SNR', 1.0),
                'R': scenario_params.get('R', 0.5),
                'N': scenario_params.get('N', 20),
                'n': scenario_params.get('n', 128),
                'th': scenario_params.get('th', 1e-6)
            }
            
            # Call the /exponents endpoint internally
            result = await exponents(**query_params)
            results.append({
                "scenario": i + 1,
                "parameters": scenario_params,
                "optimal_rho": result["rho óptima"]
            })
        
        return results
    else:
        # Single computation case (original behavior)
        # Convert parameters to query string format
        query_params = {
            'M': params.get('M', 2),
            'typeM': params.get('typeModulation', 'PAM'),
            'SNR': params.get('SNR', 1.0),
            'R': params.get('R', 0.5),
            'N': params.get('N', 20),
            'n': params.get('n', 128),
            'th': params.get('th', 1e-6)
        }
        
        # Call the /exponents endpoint internally
        result = await exponents(**query_params)
        return result["rho óptima"]

async def computeAllMetrics(**params) -> Dict[str, float]:
    """Compute all three metrics and return them as a dictionary."""
    # Convert parameters to query string format
    query_params = {
        'M': params.get('M', 2),
        'typeM': params.get('typeModulation', 'PAM'),
        'SNR': params.get('SNR', 1.0),
        'R': params.get('R', 0.5),
        'N': params.get('N', 20),
        'n': params.get('n', 128),
        'th': params.get('th', 1e-6)
    }
    
    # Call the /exponents endpoint internally
    result = await exponents(**query_params)
    return {
        "error_probability": result["Probabilidad de error"],
        "error_exponent": result["error_exponent"],
        "optimal_rho": result["rho óptima"]
    }

async def plotContour(**params) -> dict:
    """Generate contour plot data by calling the contour plot endpoint."""
    # Convert parameters to contour plot endpoint format
    contour_params = {
        'y': params.get('y', 'error_probability'),
        'x1': params.get('x1', 'snr'),
        'x2': params.get('x2', 'rate'),
        'rang_x1': params.get('rang_x1', [0, 20]),
        'rang_x2': params.get('rang_x2', [0.1, 0.9]),
        'points1': params.get('points1', 20),
        'points2': params.get('points2', 20),
        'typeModulation': params.get('typeModulation', 'PAM'),
        'M': params.get('M', 2),
        'SNR': params.get('SNR', 5.0),
        'Rate': params.get('Rate', 0.5),
        'N': params.get('N', 20),
        'n': params.get('n', 128),
        'th': params.get('th', 1e-6)
    }
    
    # Call the contour plot endpoint
    result = await generate_contour_plot(ContourPlotRequest(**contour_params))
    return result

FUNCTION_REGISTRY = {
    'computeErrorProbability': computeErrorProbability,
    'computeErrorExponent': computeErrorExponent,
    'computeOptimalRho': computeOptimalRho,
    'computeAllMetrics': computeAllMetrics,
    'plotFromFunction': plotFromFunction,
    'plotContour': plotContour
}

# Load the local model on startup
print("[INFO] Loading local model on startup...", flush=True)
try:
    local_agent.load_model()
    print("[INFO] Local model loaded successfully", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to load local model: {e}", flush=True)
    # Continue without the local model - it will be loaded on first use as fallback

app = FastAPI()

# Cargar la biblioteca compartida
lib = ctypes.CDLL('./build/libfunctions.so')

# Servir archivos estáticos (CSS, imágenes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------ HELPERS --------------------------------------------------------------------------
def sanitize_query_param(value: Any, default: Any, target_type: type = float) -> Any:
    """Sanitize a query parameter value, handling problematic strings."""
    if value is None or str(value).lower() in ['unknown', 'undefined', '', 'none', 'null', 'nan']:
        return default
    try:
        if target_type == str:
            return str(value)
        # For float/int, use float conversion for flexibility
        return float(value)
    except (ValueError, TypeError):
        return default

def call_cpp_exponents(params: dict) -> list:
    """Helper to call the C++ exponents function with a parameter dictionary."""
    result_buffer = (ctypes.c_float * 3)()
    
    # Ensure all parameters are of the correct type before calling C++ function
    m_val = float(params.get("M", 2))
    type_m_val = str(params.get("typeModulation", "PAM"))
    snr_val = float(params.get("SNR", 1.0))
    r_val = float(params.get("Rate", 0.5))
    n_val_c = float(params.get("N", 20))
    n_val_len = float(params.get("n", 128))
    th_val = float(params.get("th", 1e-6))

    lib.exponents(
        ctypes.c_float(m_val),
        type_m_val.encode('utf-8'),
        ctypes.c_float(snr_val),
        ctypes.c_float(r_val),
        ctypes.c_float(n_val_c),
        ctypes.c_float(n_val_len),
        ctypes.c_float(th_val),
        result_buffer
    )
    return list(result_buffer)

# ------------------------ ENDPOINTS ------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_path = "static/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}

@app.get("/exponents")
async def exponents(
        M: str = Query("2", description="Modulation"),
        typeM: str = Query("PAM", description="Tipo de modulación: PAM, QAM, etc."),
        SNR: str = Query("1.0", description="Signal to Noise Ratio"),
        R: str = Query("0.5", description="Rate"),
        N: str = Query("20", description="quadrature"),
        n: str = Query("128", description="Code length"),
        th: str = Query("1e-6", description="Threshold"),
):
    """  
    Calcula l'exponent `Pe`, 'E' i `RHO`.
    """
    # Sanitize all parameters to handle problematic values
    params = {
        "M": sanitize_query_param(M, 2),
        "typeModulation": sanitize_query_param(typeM, "PAM", str),
        "SNR": sanitize_query_param(SNR, 1.0),
        "Rate": sanitize_query_param(R, 0.5),
        "N": sanitize_query_param(N, 20),
        "n": sanitize_query_param(n, 128),
        "th": sanitize_query_param(th, 1e-6)
    }
    
    values = call_cpp_exponents(params)

    return {
        "Probabilidad de error": values[0],
        "error_exponent": values[1],
        "rho óptima": values[2]
    }

# ------------------------ GRAPHICS -------------------------------------------------------------------------
class FunctionPlotRequest(BaseModel):
    y: str
    x: str
    rang_x: list[float]
    points: int
    typeModulation: str
    M: float
    SNR: float
    Rate: float
    N: float
    n: float
    th: float

@app.post("/plot_function")
async def generate_plot_from_function(plot_data: FunctionPlotRequest):
    try:
        # Generate x values, ensuring integers for specific parameters
        if plot_data.x in ["M", "N", "n"]:
            raw = np.linspace(plot_data.rang_x[0], plot_data.rang_x[1], plot_data.points)
            x_vals = np.unique(np.round(raw).astype(int))
        else:
            x_vals = np.linspace(plot_data.rang_x[0], plot_data.rang_x[1], plot_data.points)

        y_vals = []
        # Create a base dictionary of parameters
        base_params = plot_data.dict()

        for x_point in x_vals:
            # Update the parameter for the current iteration
            current_params = base_params.copy()
            current_params[plot_data.x] = x_point
            
            # Call the C++ library and get results
            results = call_cpp_exponents(current_params)
            y_map = {
                "error_probability": results[0],
                "error_exponent": results[1],
                "rho": results[2]
            }
            y_vals.append(y_map.get(plot_data.y))

        return {"x": x_vals.tolist(), "y": y_vals}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot data: {repr(e)}")

class ContourPlotRequest(BaseModel):
    y: str
    x1: str
    x2: str
    rang_x1: list[float]
    rang_x2: list[float]
    points1: int
    points2: int
    typeModulation: str
    M: float
    SNR: float
    Rate: float
    N: float
    n: float
    th: float

@app.post("/plot_contour")
async def generate_contour_plot(plot_data: ContourPlotRequest):
    try:
        # Generate x1 values
        if plot_data.x1 in ["M", "N", "n"]:
            raw_x1 = np.linspace(plot_data.rang_x1[0], plot_data.rang_x1[1], plot_data.points1)
            x1_vals = np.unique(np.round(raw_x1).astype(int))
        else:
            x1_vals = np.linspace(plot_data.rang_x1[0], plot_data.rang_x1[1], plot_data.points1)

        # Generate x2 values
        if plot_data.x2 in ["M", "N", "n"]:
            raw_x2 = np.linspace(plot_data.rang_x2[0], plot_data.rang_x2[1], plot_data.points2)
            x2_vals = np.unique(np.round(raw_x2).astype(int))
        else:
            x2_vals = np.linspace(plot_data.rang_x2[0], plot_data.rang_x2[1], plot_data.points2)

        z_matrix = []
        base_params = plot_data.dict()

        for val1 in x1_vals:
            row = []
            for val2 in x2_vals:
                current_params = base_params.copy()
                current_params[plot_data.x1] = val1
                current_params[plot_data.x2] = val2

                results = call_cpp_exponents(current_params)
                y_map = {
                    "error_probability": results[0],
                    "error_exponent": results[1],
                    "rho": results[2]
                }
                row.append(y_map.get(plot_data.y))
            z_matrix.append(row)

        return {"x1": x1_vals.tolist(), "x2": x2_vals.tolist(), "z": z_matrix}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating contour plot: {str(e)}")

# ------------------------ CHATBOT -------------------------------------------------------------------------
class ChatbotRequest(BaseModel):
    message: str
    model_choice: str
    session_id: str = "default"  # Add session_id to track conversations

@app.post("/chatbot")
async def chatbot_with_bot(request: ChatbotRequest):
    try:
        agent_to_use = local_agent if request.model_choice == 'local' else openrouter_agent
        
        # Get or create conversation history for this session
        if request.session_id not in conversation_store:
            conversation_store[request.session_id] = []
        
        conversation_history = conversation_store[request.session_id]
        
        # Update the agent's conversation history with the session history
        if hasattr(agent_to_use, 'conversation_history'):
            agent_to_use.conversation_history = conversation_history
        
        response_text = "".join(agent_to_use.generate_response_stream(request.message))
        logging.info(f"LLM raw response: '{response_text}'")

        intro_text, function_calls = agent_to_use.parse_function_calls(response_text)
        logging.info(f"Parsed intro: '{intro_text}', Parsed function calls: {function_calls}")
        
        # Debug: Log the raw response to see what the LLM actually generated
        logging.info(f"Raw LLM response lines: {[line.strip() for line in response_text.split('\\n') if line.strip()]}")
        
        # Add print statements for more visible debugging
        print(f"=== DEBUG: Raw LLM response ===")
        print(f"Response text: '{response_text}'")
        print(f"Intro text: '{intro_text}'")
        print(f"Function calls: {function_calls}")
        print(f"=== END DEBUG ===")
        
        # Additional debug for comparison requests
        if 'compare' in request.message.lower() or 'comparison' in request.message.lower():
            print(f"=== COMPARISON DEBUG ===")
            print(f"User message: '{request.message}'")
            print(f"LLM response: '{response_text}'")
            print(f"Function calls: {function_calls}")
            if function_calls:
                print(f"First function call parameters: {function_calls[0].parameters}")
            print(f"=== END COMPARISON DEBUG ===")
            
            # Debug: Check what context is being sent to the LLM
            if hasattr(agent_to_use, '_build_conversation_context'):
                try:
                    context = agent_to_use._build_conversation_context(request.message)
                    print(f"=== LLM CONTEXT DEBUG ===")
                    print(f"Context length: {len(context)}")
                    print(f"Context ends with: {context[-200:] if len(context) > 200 else context}")
                    print(f"=== END LLM CONTEXT DEBUG ===")
                except Exception as e:
                    print(f"Error getting context: {e}")
        
        # Update conversation history
        from assistant.assistant import ConversationEntry
        import time
        conversation_entry = ConversationEntry(
            user_message=request.message,
            agent_response=response_text,
            function_calls=function_calls,
            timestamp=time.time()
        )
        conversation_history.append(conversation_entry)
        
        # Keep only last 10 conversations to prevent memory bloat
        if len(conversation_history) > 10:
            conversation_store[request.session_id] = conversation_history[-10:]
            # Also update the agent's history to match
            if hasattr(agent_to_use, 'conversation_history'):
                agent_to_use.conversation_history = conversation_store[request.session_id]

        if function_calls:
            task = function_calls[0]
            
            # Special handling for plot requests
            if task.function_name == 'plotFromFunction':
                # Convert plotFromFunction parameters to plot endpoint format
                plot_params = task.parameters
                
                # Extract plotting parameters with proper mapping
                y_var = plot_params.get('y', 'error_probability')
                x_var = plot_params.get('x', 'snr')
                min_val = plot_params.get('min', 0)
                max_val = plot_params.get('max', 20)
                points = plot_params.get('points', 50)
                
                # Handle rang_x if it's provided (for cases where LLM uses rang_x instead of min/max)
                rang_x = plot_params.get('rang_x', [min_val, max_val])
                if isinstance(rang_x, list) and len(rang_x) == 2:
                    # Check if rang_x contains 'unknown' values and provide defaults
                    if rang_x[0] == 'unknown' or rang_x[1] == 'unknown':
                        # Provide sensible defaults based on the variable being plotted
                        if x_var.lower() == 'rho':
                            rang_x = [0.0, 1.0]  # rho typically ranges from 0 to 1
                        elif x_var.lower() == 'snr':
                            rang_x = [0.0, 20.0]  # SNR typically ranges from 0 to 20 dB
                        elif x_var.lower() == 'rate':
                            rang_x = [0.1, 0.9]  # Rate typically ranges from 0.1 to 0.9
                        elif x_var.lower() == 'm':
                            rang_x = [2.0, 64.0]  # M typically ranges from 2 to 64
                        elif x_var.lower() == 'n':
                            rang_x = [10.0, 100.0]  # N typically ranges from 10 to 100
                        else:
                            rang_x = [0.0, 10.0]  # Default range for other variables
                    else:
                        # Convert to float if they're valid numbers
                        try:
                            rang_x = [float(rang_x[0]), float(rang_x[1])]
                        except (ValueError, TypeError):
                            rang_x = [0.0, 10.0]  # Fallback default
                else:
                    # Use min/max values if rang_x is not provided
                    rang_x = [min_val, max_val]
                
                # Set default values for missing parameters, handling 'unknown' values
                type_modulation = plot_params.get('typeModulation', 'PAM')
                M = plot_params.get('M', 2)
                
                # Handle 'unknown' values by setting defaults based on what's being plotted
                SNR = plot_params.get('SNR', 5.0)
                if SNR == 'unknown' or SNR is None:
                    SNR = 5.0 if x_var != 'snr' else 5.0
                
                # Handle both 'R' and 'Rate' parameter names
                Rate = plot_params.get('Rate', plot_params.get('R', 0.5))
                if Rate == 'unknown' or Rate is None:
                    Rate = 0.5 if x_var != 'rate' else 0.5
                
                N = plot_params.get('N', 20)
                if N == 'unknown' or N is None:
                    N = 20 if x_var != 'n' else 20
                
                n = plot_params.get('n', 128)
                if n == 'unknown' or n is None:
                    n = 128 if x_var != 'n' else 128
                
                th = plot_params.get('th', 1e-6)
                if th == 'unknown' or th is None:
                    th = 1e-6
                
                # Safely convert to float with error handling
                def safe_float(value, default):
                    if value == 'unknown' or value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                # Create plot request payload
                plot_payload = {
                    "y": y_var,
                    "x": x_var,
                    "rang_x": rang_x,
                    "points": points,
                    "typeModulation": type_modulation,
                    "M": safe_float(M, 2.0),
                    "SNR": safe_float(SNR, 0.5),
                    "Rate": safe_float(Rate, 0.5),
                    "N": safe_float(N, 20.0),
                    "n": safe_float(n, 128.0),
                    "th": safe_float(th, 1e-6)
                }
                
                initial_message = intro_text if intro_text else f"Generating plot for {y_var} vs {x_var} from {rang_x[0]} to {rang_x[1]}..."
                
                logging.info(f"Sending plot request: {plot_payload}")
                return {
                    "response": initial_message,
                    "task": {
                        "function_name": "plotFromFunction",
                        "parameters": plot_payload,
                        "is_plot_request": True
                    }
                }
            elif task.function_name == 'plotContour':
                # Convert plotContour parameters to contour plot endpoint format
                plot_params = task.parameters
                
                # Extract plotting parameters with proper mapping
                y_var = plot_params.get('y', 'error_probability')
                x1_var = plot_params.get('x1', 'snr')
                x2_var = plot_params.get('x2', 'rate')
                min_x1 = plot_params.get('min_x1', 0)
                max_x1 = plot_params.get('max_x1', 20)
                min_x2 = plot_params.get('min_x2', 0.1)
                max_x2 = plot_params.get('max_x2', 0.9)
                points1 = plot_params.get('points1', 20)
                points2 = plot_params.get('points2', 20)
                
                # Handle rang_x1 and rang_x2 if provided
                rang_x1 = plot_params.get('rang_x1', [min_x1, max_x1])
                rang_x2 = plot_params.get('rang_x2', [min_x2, max_x2])
                
                # Provide sensible defaults for unknown values
                if rang_x1[0] == 'unknown' or rang_x1[1] == 'unknown':
                    if x1_var.lower() == 'snr':
                        rang_x1 = [0.0, 20.0]
                    elif x1_var.lower() == 'rate':
                        rang_x1 = [0.1, 0.9]
                    elif x1_var.lower() == 'm':
                        rang_x1 = [2.0, 16.0]
                    else:
                        rang_x1 = [0.0, 10.0]
                
                if rang_x2[0] == 'unknown' or rang_x2[1] == 'unknown':
                    if x2_var.lower() == 'snr':
                        rang_x2 = [0.0, 20.0]
                    elif x2_var.lower() == 'rate':
                        rang_x2 = [0.1, 0.9]
                    elif x2_var.lower() == 'm':
                        rang_x2 = [2.0, 16.0]
                    else:
                        rang_x2 = [0.0, 10.0]
                
                # Set default values for missing parameters
                type_modulation = plot_params.get('typeModulation', 'PAM')
                M = plot_params.get('M', 2)
                SNR = plot_params.get('SNR', 0.5)
                Rate = plot_params.get('Rate', plot_params.get('R', 0.5))
                N = plot_params.get('N', 20)
                n = plot_params.get('n', 128)
                th = plot_params.get('th', 1e-6)
                
                # Handle 'unknown' values by setting defaults based on what's being plotted
                if M == 'unknown' or M is None:
                    M = 2 if x1_var != 'm' and x2_var != 'm' else 2
                
                if SNR == 'unknown' or SNR is None:
                    SNR = 5.0 if x1_var != 'snr' and x2_var != 'snr' else 5.0
                
                if Rate == 'unknown' or Rate is None:
                    Rate = 0.5 if x1_var != 'rate' and x2_var != 'rate' else 0.5
                
                if N == 'unknown' or N is None:
                    N = 20 if x1_var != 'n' and x2_var != 'n' else 20
                
                if n == 'unknown' or n is None:
                    n = 128 if x1_var != 'n' and x2_var != 'n' else 128
                
                if th == 'unknown' or th is None:
                    th = 1e-6
                
                # Safely convert to float with error handling
                def safe_float(value, default):
                    if value == 'unknown' or value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                # Create contour plot request payload
                contour_payload = {
                    "y": y_var,
                    "x1": x1_var,
                    "x2": x2_var,
                    "rang_x1": rang_x1,
                    "rang_x2": rang_x2,
                    "points1": points1,
                    "points2": points2,
                    "typeModulation": type_modulation,
                    "M": safe_float(M, 2.0),
                    "SNR": safe_float(SNR, 5.0),
                    "Rate": safe_float(Rate, 0.5),
                    "N": safe_float(N, 20.0),
                    "n": safe_float(n, 128.0),
                    "th": safe_float(th, 1e-6)
                }
                
                initial_message = intro_text if intro_text else f"Generating contour plot for {y_var} vs {x1_var} and {x2_var}..."
                
                logging.info(f"Sending contour plot request: {contour_payload}")
                return {
                    "response": initial_message,
                    "task": {
                        "function_name": "plotContour",
                        "parameters": contour_payload,
                        "is_plot_request": True,
                        "is_contour_plot": True
                    }
                }
            else:
                # Regular computation tasks
                # Use LLM-generated text if available, otherwise use a more natural fallback
                print(f"=== DEBUG: Message selection ===")
                print(f"Intro text: '{intro_text}'")
                print(f"Intro text type: {type(intro_text)}")
                print(f"Intro text is None: {intro_text is None}")
                print(f"Intro text is empty: {intro_text == ''}")
                print(f"Intro text is whitespace: {intro_text.strip() == '' if intro_text else 'N/A'}")
                
                if intro_text and intro_text.strip():
                    initial_message = intro_text
                    logging.info(f"Using LLM-generated conversational text: '{initial_message}'")
                    print(f"Using LLM-generated text: '{initial_message}'")
                else:
                    # Only use hardcoded message if LLM didn't generate any conversational text
                    initial_message = f"Understood. Starting the calculation for `{task.function_name}`. This may take a moment..."
                    logging.info(f"Using fallback message because LLM generated no conversational text")
                    print(f"Using fallback message: '{initial_message}'")
                
                print(f"=== END DEBUG ===")
                
                logging.info(f"Sending initial confirmation: '{initial_message}' and task: {task}")
                return {
                    "response": initial_message,
                    "task": {
                        "function_name": task.function_name,
                        "parameters": task.parameters,
                        "is_plot_request": False
                    }
                }
        else:
            # If no function call, just return the LLM's text
            logging.info("No function call detected. Sending raw text response.")
            return {"response": response_text}
            
    except Exception as e:
        logging.error(f"Error in /chatbot endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")

class ComputeRequest(BaseModel):
    function_name: str
    parameters: Dict[str, Any]

@app.post("/compute")
async def compute_task(request: ComputeRequest):
    logging.info(f"Received /compute request for function: {request.function_name} with parameters: {request.parameters}")
    try:
        func_name = request.function_name
        params = request.parameters
        
        if func_name not in FUNCTION_REGISTRY:
            logging.error(f"Function '{func_name}' not found in registry.")
            raise HTTPException(status_code=404, detail="Function not found")
        
        # For computation functions, always compute all three metrics for better LLM context
        if func_name in ['computeErrorProbability', 'computeErrorExponent', 'computeOptimalRho']:
            # Execute the function (this handles both single and comparison cases)
            calculation_function = FUNCTION_REGISTRY[func_name]
            raw_result = await calculation_function(**params)
            logging.info(f"Raw result for {func_name}: {raw_result}")
            
            # Check if this is a comparison result (list of scenarios)
            if isinstance(raw_result, list):
                # Comparison case - multiple scenarios
                logging.info("Processing comparison results with LLM...")
                formatted_result = local_agent.format_comparison_result(raw_result)
                logging.info(f"Formatted comparison result: '{formatted_result}'")
            else:
                # Single computation case
                # Compute all metrics for context (for single computation)
                all_metrics = await computeAllMetrics(**params)
                logging.info(f"All metrics computed: {all_metrics}")
                
                # Get the specific requested result
                if func_name == 'computeErrorProbability':
                    requested_result = all_metrics["error_probability"]
                elif func_name == 'computeErrorExponent':
                    requested_result = all_metrics["error_exponent"]
                elif func_name == 'computeOptimalRho':
                    requested_result = all_metrics["optimal_rho"]
                
                # Use the local agent to format the result with all three metrics for context
                logging.info("Formatting result with LLM using all metrics for context...")
                formatted_result = local_agent.format_computation_result_with_context(func_name, requested_result, all_metrics, params)
                logging.info(f"Formatted result for {func_name}: '{formatted_result}'")
        else:
            # For non-computation functions (like plots), use the original approach
            calculation_function = FUNCTION_REGISTRY[func_name]
            raw_result = await calculation_function(**params)
            logging.info(f"Raw result for {func_name}: {raw_result}")
            
            # Use the local agent to format the result nicely
            logging.info("Formatting result with LLM...")
            formatted_result = local_agent.format_computation_result(func_name, raw_result, params)
            logging.info(f"Formatted result for {func_name}: '{formatted_result}'")
        
        return {"result": formatted_result}

    except Exception as e:
        logging.error(f"Error during computation for {request.function_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during computation: {str(e)}")

@app.delete("/chatbot/session/{session_id}")
async def clear_conversation_session(session_id: str):
    """Clear conversation history for a specific session."""
    if session_id in conversation_store:
        del conversation_store[session_id]
        # Also clear the agent's conversation history if it matches
        if hasattr(local_agent, 'conversation_history'):
            local_agent.conversation_history = []
        if hasattr(openrouter_agent, 'conversation_history'):
            openrouter_agent.conversation_history = []
    return {"message": f"Conversation session {session_id} cleared"}
