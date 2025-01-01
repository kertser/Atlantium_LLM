from typing import Dict, List, Optional, Union
import ctypes
import os
import json
from pathlib import Path
import platform
import logging
from utils.LLM_utils import openai_post_request
from config import CONFIG
from models.prompt_manager import PromptLoader
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class REDLibrary:
    def __init__(self, api_key: str):
        """Initialize connection to the RED calculation library"""
        try:
            self.api_key = api_key
            self.lib = self._load_library()
            if self.lib is None:
                raise RuntimeError("Failed to load library")

            if not self._setup_functions():
                raise RuntimeError("Failed to setup functions")

            self.supported_systems = self._get_supported_systems()
            if not self.supported_systems:
                raise RuntimeError("No supported systems found")

            self.prompt_loader = PromptLoader()

        except Exception as e:
            logging.error(f"Failed to initialize RED library: {e}")
            raise

    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Load the appropriate library based on the operating system"""
        try:
            lib_path = self._get_library_path()
            if lib_path is None:
                return None

            if platform.system() == 'Windows':
                resources_dir = str(lib_path.parent.absolute())

                # Add resources directory to PATH and current directory
                os.environ['PATH'] = resources_dir + os.pathsep + os.environ['PATH']
                original_dir = os.getcwd()
                os.chdir(resources_dir)

                # Get kernel32 functions
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                load_library_ex = kernel32.LoadLibraryExW
                load_library_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
                load_library_ex.restype = ctypes.c_void_p

                try:
                    # Load flags
                    LOAD_WITH_ALTERED_SEARCH_PATH = 0x8

                    # Try loading JSON DLL first
                    json_path = str(lib_path.parent / "libjson-c.dll")
                    json_handle = load_library_ex(json_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
                    if not json_handle:
                        logging.error("Could not load JSON DLL")
                        return None

                    # Try loading RED API DLL
                    red_handle = load_library_ex(str(lib_path), None, LOAD_WITH_ALTERED_SEARCH_PATH)
                    if not red_handle:
                        logging.error("Could not load RED API DLL")
                        return None

                    # Create CDLL objects
                    json_lib = ctypes.CDLL(json_path, handle=json_handle)
                    red_lib = ctypes.CDLL(str(lib_path), handle=red_handle)

                    os.chdir(original_dir)
                    return red_lib

                except Exception as e:
                    logging.error(f"Error during DLL loading: {e}")
                    os.chdir(original_dir)
                    return None
                finally:
                    os.chdir(original_dir)

            else:
                return ctypes.CDLL(str(lib_path))

        except Exception as e:
            logging.error(f"Error in load_library: {e}")
            return None

    def _get_library_path(self) -> Optional[Path]:
        """Get the appropriate library path based on OS"""
        try:
            base_dir = Path(__file__).parent
            resources_dir = base_dir / 'resources'
            lib_name = "red_api.dll" if platform.system() == 'Windows' else "libred_api.so"
            lib_path = resources_dir / lib_name

            if not lib_path.exists():
                logging.error(f"Library not found at {lib_path}")
                return None

            return lib_path
        except Exception as e:
            logging.error(f"Failed to get library path: {e}")
            return None

    def _get_n_lamps(self, system_type: str) -> Dict[str, Union[int, str]]:
        """Get the number of lamps for a specific UV system from the DLL"""
        try:
            if self.lib is None:
                return {"error": "Library not initialized"}

            # Call your DLL function to get number of lamps
            # Example (adjust according to your actual DLL function):
            n_lamps = self.lib.get_n_lamps(system_type)
            return {
                "system_type": system_type,
                "n_lamps": n_lamps
            }
        except Exception as e:
            logging.error(f"Error in _get_n_lamps: {str(e)}")
            return {"error": f"Failed to get number of lamps: {str(e)}"}

    def _get_supported_systems(self) -> List[str]:
        """Get list of supported UV systems"""
        try:
            size = ctypes.c_size_t()
            systems_ptr = self.get_supported_systems_func(ctypes.byref(size))
            if not systems_ptr or size.value == 0:
                logging.error("No systems found or empty pointer returned")
                return []

            logging.debug(f"Found {size.value} systems")
            systems = []
            for i in range(size.value):
                if systems_ptr[i]:
                    try:
                        system = systems_ptr[i].decode('utf-8')
                        systems.append(system)
                        logging.debug(f"Found system: {system}")
                    except Exception as e:
                        logging.error(f"Error decoding system at index {i}: {e}")
            return systems
        except Exception as e:
            logging.error(f"Error getting supported systems: {e}")
            return []

    def _init_system_config(self, config_path: str) -> bool:
        """Initialize system configuration"""
        try:
            if not Path(config_path).exists():
                logging.error(f"Config file not found: {config_path}")
                return False

            init_config = self.lib.init_system_config
            init_config.argtypes = [ctypes.c_char_p]
            init_config.restype = ctypes.c_bool
            return init_config(config_path.encode('utf-8'))
        except Exception as e:
            logging.error(f"Error initializing system config: {e}")
            return False

    def _setup_functions(self) -> bool:
        """Setup the C function interfaces"""
        if self.lib is None:
            logging.error("Library is None, cannot setup functions")
            return False

        try:
            # First initialize system configuration
            config_path = str(Path(__file__).parent / 'resources' / 'supported_systems.json')
            if not self._init_system_config(config_path):
                return False

            # Define function types
            RED_FUNC = ctypes.CFUNCTYPE(
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_uint32
            )

            # Store function references as class attributes
            self.get_supported_systems_func = self.lib.get_supported_systems
            self.get_supported_systems_func.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
            self.get_supported_systems_func.restype = ctypes.POINTER(ctypes.c_char_p)

            self.get_lamp_count_func = self.lib.get_lamp_count
            self.get_lamp_count_func.argtypes = [ctypes.c_char_p]
            self.get_lamp_count_func.restype = ctypes.c_uint32

            self.get_red_function = self.lib.getREDFunction
            self.get_red_function.argtypes = [ctypes.c_char_p]
            self.get_red_function.restype = RED_FUNC

            self.validate_parameters_func = self.lib.validate_parameters
            self.validate_parameters_func.argtypes = [
                ctypes.c_char_p,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double
            ]
            self.validate_parameters_func.restype = ctypes.c_bool

            self.get_lamp_power_func = self.lib.get_lamp_power
            self.get_lamp_power_func.argtypes = [ctypes.c_char_p]
            self.get_lamp_power_func.restype = ctypes.c_double

            return True

        except Exception as e:
            logging.error(f"Error setting up functions: {e}")
            return False

    def _validate_functions(self, functions) -> bool:
        """Validate the functions format"""
        if not isinstance(functions, list):
            logging.error("Functions must be a list")
            return False

        for func in functions:
            if not isinstance(func, dict):
                logging.error(f"Each function must be a dict, got {type(func)}")
                return False
            if 'name' not in func or 'parameters' not in func:
                logging.error("Function missing required fields 'name' or 'parameters'")
                return False

        return True

    def _calculate_red(self, system_type: str, flow: float, uvt: float,
                       uvt215: float = -1, d1_log: float = 18.0,
                       power_settings: dict = None,
                       efficiency_settings: dict = None) -> Optional[Dict]:
        """Calculate RED value with custom power and efficiency settings"""
        try:
            # Validate system type
            if system_type not in self.supported_systems:
                return {
                    "error": f"System type '{system_type}' not found. Available systems: {', '.join(self.supported_systems)}"
                }

            # Get number of lamps
            n_lamps = self.get_lamp_count_func(system_type.encode('utf-8'))
            if not n_lamps:
                return {
                    "error": f"Could not get lamp count for system {system_type}"
                }

            # Get lamp power
            lamp_power = self.get_lamp_power_func(system_type.encode('utf-8'))

            # Initialize default arrays
            power = [CONFIG.RED_CALCULATOR_DEFAULT_DRIVE] * n_lamps
            efficiency = [CONFIG.RED_CALCULATOR_DEFAULT_EFFICIENCY] * n_lamps

            # Process power settings
            if power_settings:
                # First apply all_lamps setting if present
                if 'all_lamps' in power_settings:
                    power = [float(power_settings['all_lamps'])] * n_lamps

                # Then apply specific lamp settings
                if 'specific_lamps' in power_settings:
                    for lamp_idx_str, value in power_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx_str) - 1  # Convert 1-based to 0-based indexing
                            if 0 <= idx < n_lamps:
                                power[idx] = float(value)
                            else:
                                return {
                                    "error": f"Invalid lamp index {lamp_idx_str}. System has {n_lamps} lamps"
                                }
                        except (ValueError, IndexError):
                            return {
                                "error": f"Invalid lamp power setting for lamp {lamp_idx_str}"
                            }

            # Process efficiency settings
            if efficiency_settings:
                # First apply all_lamps setting if present
                if 'all_lamps' in efficiency_settings:
                    efficiency = [float(efficiency_settings['all_lamps'])] * n_lamps

                # Then apply specific lamp settings
                if 'specific_lamps' in efficiency_settings:
                    for lamp_idx_str, value in efficiency_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx_str) - 1
                            if 0 <= idx < n_lamps:
                                efficiency[idx] = float(value)
                            else:
                                return {
                                    "error": f"Invalid lamp index {lamp_idx_str}. System has {n_lamps} lamps"
                                }
                        except (ValueError, IndexError):
                            return {
                                "error": f"Invalid lamp efficiency setting for lamp {lamp_idx_str}"
                            }

            # Convert lists to ctypes arrays
            power_array = (ctypes.c_double * n_lamps)(*power)
            efficiency_array = (ctypes.c_double * n_lamps)(*efficiency)

            # Get and call RED calculation function
            red_func = self.get_red_function(system_type.encode('utf-8'))
            if not red_func:
                return {
                    "error": f"Could not get RED calculation function for system {system_type}"
                }

            result = red_func(
                ctypes.c_double(flow),
                ctypes.c_double(uvt),
                ctypes.c_double(uvt215),
                power_array,
                efficiency_array,
                ctypes.c_double(d1_log),
                ctypes.c_uint32(n_lamps)
            )

            # Validate result
            if result <= 0:
                return {
                    "error": "Calculation resulted in invalid RED value"
                }

            # Prepare detailed output
            return {
                "result": round(result, 1),
                "details": {
                    "system_type": system_type,
                    "number_of_lamps": n_lamps,
                    "lamp_power_watts": round(lamp_power, 1),
                    "parameters": {
                        "flow": flow,
                        "uvt": uvt,
                        "uvt215": uvt215 if uvt215 > 0 else "N/A",
                        "d1_log": d1_log
                    },
                    "lamp_settings": {
                        "power": [round(p, 1) for p in power],
                        "efficiency": [round(e, 1) for e in efficiency]
                    }
                }
            }

        except Exception as e:
            return {
                "error": f"Calculation error: {str(e)}"
            }

    def detect_calculation_content(self, text: str, template: Dict) -> Dict:
        """Detect if text contains calculator-related content"""
        try:
            # Use OpenAI to detect calculator content based on the template
            messages = [
                {
                    "role": "system",
                    "content": template['system']
                }
            ]

            # Add examples if present in template
            if 'examples' in template:
                for example in template['examples']:
                    messages.extend([
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": json.dumps(example['output'], ensure_ascii=False)}
                    ])

            # Add the actual query
            messages.append({"role": "user", "content": text})

            response = openai_post_request(
                messages=messages,
                model_name=CONFIG.GPT_MODEL,
                temperature=0,  # Use 0 for consistent detection
                max_tokens=150,
                api_key=self.api_key
            )

            if 'choices' not in response:
                logging.error("No choices in OpenAI response")
                return {
                    "has_calculator_content": False,
                    "is_valid_system": False,
                    "parameters": {},
                    "error_message": "Failed to get response from OpenAI",
                    "extracted_text": ""
                }

            result_text = response['choices'][0]['message']['content']

            try:
                result = json.loads(result_text)

                # Validate against template output format
                if not isinstance(result.get('has_calculator_content'), bool):
                    logging.error("Invalid response format: missing or invalid has_calculator_content")
                    return {
                        "has_calculator_content": False,
                        "is_valid_system": False,
                        "parameters": {},
                        "error_message": "Invalid response format",
                        "extracted_text": ""
                    }

                # Validate system type if present
                if result.get('has_calculator_content') and 'parameters' in result:
                    if result['parameters'].get('system_type'):
                        if result['parameters']['system_type'] not in self.supported_systems:
                            logging.warning(f"Unsupported system type: {result['parameters']['system_type']}")
                            return {
                                "has_calculator_content": True,
                                "is_valid_system": False,
                                "parameters": result['parameters'],
                                "error_message": f"Unsupported system type: {result['parameters']['system_type']}. Please verify the system model number.",
                                "extracted_text": result.get('extracted_text', '')
                            }
                        else:
                            # Valid system type
                            result['is_valid_system'] = True
                            result['error_message'] = None

                return result

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {e}")
                return {
                    "has_calculator_content": False,
                    "is_valid_system": False,
                    "parameters": {},
                    "error_message": "Failed to parse response",
                    "extracted_text": ""
                }

        except Exception as e:
            logging.error(f"Error in calculator content detection: {e}")
            return {
                "has_calculator_content": False,
                "is_valid_system": False,
                "parameters": {},
                "error_message": str(e),
                "extracted_text": ""
            }

    def process_query(self, query: str) -> Union[Dict, None]:
        """Process a natural language query using OpenAI"""
        if self.lib is None:
            return {"error": "Library not initialized"}

        try:
            # Get prompts from prompt manager
            system_prompt = self.prompt_loader.get_template('red_calculator.system_prompt')
            functions = self.prompt_loader.get_template('red_calculator.functions')

            # Debug logging
            logging.debug(f"Functions type: {type(functions)}")
            logging.debug(f"Functions content: {functions}")

            if not isinstance(functions, list):
                logging.error(f"Functions is not a list: {functions}")
                return {"error": "Invalid functions format"}

            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            # Use LLM_utils for OpenAI request
            try:
                response = openai_post_request(
                    messages=messages,
                    model_name=CONFIG.RED_CALCULATOR_MODEL,
                    temperature=CONFIG.RED_CALCULATOR_TEMPERATURE,
                    max_tokens=CONFIG.RED_CALCULATOR_MAX_TOKENS,
                    functions=functions,
                    function_call={"name": "calculate_red"},  # Force it to use calculate_red
                    api_key=self.api_key
                )
            except Exception as e:
                logging.error(f"OpenAI API error: {str(e)}")
                return {"error": f"OpenAI API error: {str(e)}"}

            # Process response
            if not response or 'choices' not in response:
                return {"error": "Failed to get response from OpenAI"}

            message = response['choices'][0]['message']
            if 'function_call' not in message:
                return {"error": "No function call in response"}

            function_call = message['function_call']

            # Execute the requested function
            result = self._execute_function(function_call)
            if result is None:
                return {"error": "Failed to execute function"}

            return {
                "function": function_call['name'],
                "parameters": json.loads(function_call['arguments']),
                "result": result
            }

        except Exception as e:
            logging.error(f"Error in process_query: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}

    def _execute_function(self, function_call: Dict) -> Optional[Dict]:
        """Execute the requested function with provided arguments"""
        try:
            func_name = function_call['name']
            args = json.loads(function_call['arguments'])

            if func_name == "get_supported_systems":
                return self._get_supported_systems()
            elif func_name == "get_n_lamps":
                return self._get_n_lamps(**args)
            elif func_name == "calculate_red":
                return self._calculate_red(**args)
            else:
                return {"error": f"Unknown function: {func_name}"}

        except Exception as e:
            return {"error": f"Function execution error: {str(e)}"}


def main():
    try:
        # Load environment variables from the correct path
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OpenAI API key not found in environment variables")
            return

        calculator = REDLibrary(api_key=api_key)

        example_queries = [
            "Calculate RED for RZM-350-8 with flow 100, UVT 95%, lamp 1 at 90% power and lamp 2 at 80% power, all other lamps at 85% power",
            "Calculate RED for RZMW-350-11 with flow 200, UVT 92%, lamp 1 efficiency 85% and lamp 2 efficiency 75%, all lamps at 80% power",
            "Calculate RED for RZ-163-12 with flow 100, UVT 95%, lamp 1 efficiency 90%, lamp 2 efficiency 85%",
            "What day is it today?"
        ]

        for query in example_queries:
            logging.info(f"\nQuery: {query}")
            result = calculator.process_query(query)
            """
            print(json.dumps(result, indent=2))
            print("-" * 80)
            """
            # print the result:
            if 'result' in result:
                logging.info(f"Result: {result['result']['result']}")
            else:
                logging.info(f"Non-related query")

    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()