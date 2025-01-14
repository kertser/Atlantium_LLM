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
            self._json_lib = None  # Initialize JSON DLL reference
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

                    # Load JSON DLL first (dependency)
                    json_path = str(lib_path.parent / "libjson-c.dll")
                    json_handle = load_library_ex(json_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
                    if not json_handle:
                        logging.error("Could not load JSON DLL")
                        return None

                    # Keep a reference to the JSON DLL to prevent it from being unloaded
                    self._json_lib = ctypes.CDLL(json_path, handle=json_handle)

                    # Try loading RED API DLL
                    red_handle = load_library_ex(str(lib_path), None, LOAD_WITH_ALTERED_SEARCH_PATH)
                    if not red_handle:
                        logging.error("Could not load RED API DLL")
                        return None

                    # Create and return RED API DLL object
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

    def get_grouped_supported_systems(self) -> Dict[str, List[str]]:
        """Group supported systems by their series"""
        supported_systems = self._get_supported_systems()
        series_groups = {
            'RZ Series': [s for s in supported_systems if s.startswith('RZ-')],
            'RZM Series': [s for s in supported_systems if s.startswith('RZM-')],
            'RZMW Series': [s for s in supported_systems if s.startswith('RZMW-')],
            'Other Series': [s for s in supported_systems if not any(s.startswith(p) for p in ['RZ-', 'RZM-', 'RZMW-'])]
        }
        return {k: sorted(v) for k, v in series_groups.items() if v}  # Only include non-empty groups

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

    def _get_parameter_ranges(self, system_type: str) -> Optional[Dict]:
        """Get valid parameter ranges for a system from configuration"""
        try:
            specs = self._load_system_specifications()
            if not specs or 'supported_systems' not in specs:
                logging.error("No supported_systems found in specifications")
                return None

            systems = specs['supported_systems']
            if system_type not in systems:
                logging.error(f"No specifications found for system {system_type}")
                return None

            system_specs = systems[system_type]
            if 'operational_limits' not in system_specs:
                logging.error(f"No operational limits found for system {system_type}")
                return None

            limits = system_specs['operational_limits']

            return {
                "flow": {
                    "min": limits['flow']['min'],
                    "max": limits['flow']['max'],
                    "unit": limits['flow']['unit']
                },
                "uvt": {
                    "min": limits['uvt']['min'],
                    "max": limits['uvt']['max'],
                    "unit": limits['uvt']['unit']
                }
            }
        except Exception as e:
            logging.error(f"Error getting parameter ranges: {str(e)}")
            return None

    def _load_system_specifications(self) -> Dict:
        """Load system specifications from JSON file"""
        try:
            config_path = Path(__file__).parent / 'resources' / 'supported_systems.json'
            if not config_path.exists():
                logging.error(f"Configuration file not found at {config_path}")
                return {}

            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading system specifications: {e}")
            return {}

    def _calculate_red(self, system_type: str, flow: float, uvt: float,
                       uvt215: float = -1, d1_log: float = 18.0,
                       power_settings: dict = None,
                       efficiency_settings: dict = None) -> Optional[Dict]:
        """Calculate RED value with custom power and efficiency settings"""
        try:
            # Validate system type first
            if system_type not in self.supported_systems:
                return {
                    "status": "error",
                    "error": {
                        "type": "system",
                        "message": f"System type '{system_type}' not found"
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt
                    }
                }

            # Get parameter ranges first
            ranges = self._get_parameter_ranges(system_type)
            if ranges is None:
                return {
                    "status": "error",
                    "error": {
                        "type": "system",
                        "message": f"Could not get valid parameter ranges for system {system_type}"
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt
                    }
                }

            # Validate parameters against ranges
            validation_errors = {}
            if flow < ranges["flow"]["min"] or flow > ranges["flow"]["max"]:
                validation_errors["flow"] = {
                    "value": round(flow, 1),
                    "min": ranges["flow"]["min"],
                    "max": ranges["flow"]["max"],
                    "unit": ranges["flow"]["unit"]
                }

            if uvt < ranges["uvt"]["min"] or uvt > ranges["uvt"]["max"]:
                validation_errors["uvt"] = {
                    "value": round(uvt, 1),
                    "min": ranges["uvt"]["min"],
                    "max": ranges["uvt"]["max"],
                    "unit": ranges["uvt"]["unit"]
                }

            if validation_errors:
                return {
                    "status": "error",
                    "error": {
                        "type": "validation",
                        "errors": validation_errors
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt,
                        "power_settings": power_settings or {},
                        "efficiency_settings": efficiency_settings or {"all_lamps": 80.0}
                    }
                }

            # Get number of lamps
            n_lamps = self.get_lamp_count_func(system_type.encode('utf-8'))
            if not n_lamps:
                return {
                    "status": "error",
                    "error": {
                        "type": "system",
                        "message": f"Could not get lamp count for system {system_type}"
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt
                    }
                }

            # Get lamp power
            lamp_power = self.get_lamp_power_func(system_type.encode('utf-8'))

            # Initialize arrays with defaults
            power = [CONFIG.RED_CALCULATOR_DEFAULT_DRIVE] * n_lamps
            efficiency = [CONFIG.RED_CALCULATOR_DEFAULT_EFFICIENCY] * n_lamps

            # Process power settings
            if power_settings:
                if 'all_lamps' in power_settings:
                    power = [float(power_settings['all_lamps'])] * n_lamps

                if 'specific_lamps' in power_settings:
                    for lamp_idx_str, value in power_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx_str) - 1
                            if 0 <= idx < n_lamps:
                                power[idx] = float(value)
                            else:
                                return {
                                    "status": "error",
                                    "error": {
                                        "type": "validation",
                                        "message": f"Invalid lamp index {lamp_idx_str}. System has {n_lamps} lamps"
                                    },
                                    "parameters": {
                                        "system_type": system_type,
                                        "flow": flow,
                                        "uvt": uvt,
                                        "power_settings": power_settings
                                    }
                                }
                        except (ValueError, IndexError):
                            return {
                                "status": "error",
                                "error": {
                                    "type": "validation",
                                    "message": f"Invalid lamp power setting for lamp {lamp_idx_str}"
                                },
                                "parameters": {
                                    "system_type": system_type,
                                    "flow": flow,
                                    "uvt": uvt,
                                    "power_settings": power_settings
                                }
                            }

            # Process efficiency settings
            if efficiency_settings:
                if 'all_lamps' in efficiency_settings:
                    efficiency = [float(efficiency_settings['all_lamps'])] * n_lamps

                if 'specific_lamps' in efficiency_settings:
                    for lamp_idx_str, value in efficiency_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx_str) - 1
                            if 0 <= idx < n_lamps:
                                efficiency[idx] = float(value)
                            else:
                                return {
                                    "status": "error",
                                    "error": {
                                        "type": "validation",
                                        "message": f"Invalid lamp index {lamp_idx_str}. System has {n_lamps} lamps"
                                    },
                                    "parameters": {
                                        "system_type": system_type,
                                        "flow": flow,
                                        "uvt": uvt,
                                        "efficiency_settings": efficiency_settings
                                    }
                                }
                        except (ValueError, IndexError):
                            return {
                                "status": "error",
                                "error": {
                                    "type": "validation",
                                    "message": f"Invalid lamp efficiency setting for lamp {lamp_idx_str}"
                                },
                                "parameters": {
                                    "system_type": system_type,
                                    "flow": flow,
                                    "uvt": uvt,
                                    "efficiency_settings": efficiency_settings
                                }
                            }

            # Prepare arrays for calculation
            power_array = (ctypes.c_double * n_lamps)(*power)
            efficiency_array = (ctypes.c_double * n_lamps)(*efficiency)

            # Get RED calculation function
            red_func = self.get_red_function(system_type.encode('utf-8'))
            if not red_func:
                return {
                    "status": "error",
                    "error": {
                        "type": "system",
                        "message": f"Could not get RED calculation function for system {system_type}"
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt
                    }
                }

            # Calculate RED
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
                    "status": "error",
                    "error": {
                        "type": "calculation",
                        "message": "Calculation resulted in invalid RED value"
                    },
                    "parameters": {
                        "system_type": system_type,
                        "flow": flow,
                        "uvt": uvt,
                        "power_settings": power_settings or {},
                        "efficiency_settings": efficiency_settings or {"all_lamps": 80.0}
                    }
                }

            # Return successful result
            return {
                "status": "success",
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
                "status": "error",
                "error": {
                    "type": "system",
                    "message": f"Calculation error: {str(e)}"
                },
                "parameters": {
                    "system_type": system_type,
                    "flow": flow,
                    "uvt": uvt
                }
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
                            # Group and format supported systems
                            grouped_systems = self.get_grouped_supported_systems()
                            systems_list = []

                            # Add each series with proper formatting
                            for series, systems in grouped_systems.items():
                                if systems:
                                    series_systems = [f"  - {system}" for system in sorted(systems)]
                                    systems_list.extend([f"• {series}:"] + series_systems)

                            # Find similar system suggestion
                            invalid_system = result['parameters']['system_type']
                            similar_system = next((s for s in self.supported_systems
                                                   if s.replace('-', '').lower() ==
                                                   invalid_system.replace('-', '').lower()), None)

                            # Build error message
                            error_message = (
                                f"Unsupported system type: {invalid_system}. "
                                f"Please verify the system model number."
                                f"\n\nSupported System Types:\n\n"
                                f"{chr(10).join(systems_list)}"
                            )

                            if similar_system:
                                error_message += f"\n\nDid you mean: {similar_system}?"

                            logging.warning(f"Unsupported system type: {invalid_system}")
                            return {
                                "has_calculator_content": True,
                                "is_valid_system": False,
                                "parameters": result['parameters'],
                                "error_message": error_message,
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

            # Modify this part
            response_dict = {
                "function": function_call['name'],
                "parameters": json.loads(function_call['arguments'])
            }

            # Handle validation errors and successful calculations differently
            if isinstance(result, dict):
                if 'error' in result:
                    response_dict['error'] = result['error']
                    response_dict['original_parameters'] = result.get('original_parameters', {})
                else:
                    response_dict['result'] = result

            return response_dict

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