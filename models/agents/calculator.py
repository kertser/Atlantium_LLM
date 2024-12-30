from openai import OpenAI
import json
from typing import Dict, List, Optional, Union
import ctypes
import os
from dotenv import load_dotenv
from pathlib import Path
import platform
import traceback

load_dotenv("../../.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found")
client = OpenAI()

# Temporary constant for config:
MAX_FLOW = 1000   # Maximum flow rate in m³/h
MIN_FLOW = 0.1    # Minimum flow rate in m³/h
MAX_UVT = 99.9    # Maximum UV transmittance in %-1cm
MIN_UVT = 0.1     # Minimum UV transmittance in %-1cm
MAX_POWER = 100.0  # Maximum power setting in %
MIN_POWER = 40.0    # Minimum power setting in %
MAX_EFFICIENCY = 100.0  # Maximum efficiency setting in %
MIN_EFFICIENCY = 50.0    # Minimum efficiency setting in %

# Define the function schemas for OpenAI - will be transferred to templates later
FUNCTIONS = [
    {
        "name": "get_supported_systems",
        "description": "Get a list of all supported UV systems",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_n_lamps",
        "description": "Get the number of lamps for a specific UV system",
        "parameters": {
            "type": "object",
            "properties": {
                "system_type": {
                    "type": "string",
                    "description": "The type/model of the UV system"
                }
            },
            "required": ["system_type"]
        }
    },
    {
        "name": "calculate_red",
        "description": "Calculate RED (Reduction Equivalent Dose) for a UV system",
        "parameters": {
            "type": "object",
            "properties": {
                "system_type": {
                    "type": "string",
                    "description": "The type/model of the UV system"
                },
                "flow": {
                    "type": "number",
                    "description": "Flow rate in m³/h"
                },
                "uvt": {
                    "type": "number",
                    "description": "UV transmittance at 254nm in %-1cm"
                },
                "uvt215": {
                    "type": "number",
                    "description": "UV transmittance at 215nm in %-1cm, use -1 if not applicable",
                    "default": -1
                },
                "d1_log": {
                    "type": "number",
                    "description": "1-Log inactivation dose in mJ/cm²",
                    "default": 18.0
                },
                "power_settings": {
                    "type": "object",
                    "description": "Power settings for each lamp (in %), if not specified defaults to 100%",
                    "properties": {
                        "all_lamps": {"type": "number"},
                        "specific_lamps": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                },
                "efficiency_settings": {
                    "type": "object",
                    "description": "Efficiency settings for each lamp (in %), if not specified defaults to 80%",
                    "properties": {
                        "all_lamps": {"type": "number"},
                        "specific_lamps": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["system_type", "flow", "uvt"]
        }
    }
]


class REDLibrary:
    def __init__(self):
        """Initialize connection to the RED calculation library"""
        print("Initializing RED Library...")
        try:
            base_dir = Path(__file__).parent
            resources_dir = base_dir / 'resources'
            print(f"Resources directory: {resources_dir}")

            # List all files in resources directory
            print("\nFiles in resources directory:")
            for file in resources_dir.glob('*'):
                print(f"- {file.name}")

            self.lib = self._load_library()
            if self.lib is None:
                print("Failed to load library")
                return

            print("Library loaded successfully")
            self._setup_functions()

            if self.lib is None:
                print("Function setup failed")
                return

            self.supported_systems = self._get_supported_systems()
            print(f"Initialization complete. Found {len(self.supported_systems)} supported systems")

        except Exception as e:
            print(f"Initialization error: {e}")
            self.lib = None


    def _print_dll_exports(self, dll_path: str):
        """Print all exported functions from a DLL"""
        import subprocess
        try:
            # Use dumpbin to list exports (Windows only)
            result = subprocess.run(['dumpbin', '/EXPORTS', dll_path],
                                    capture_output=True,
                                    text=True)
            print("\nExported functions from DLL:")
            print(result.stdout)
        except Exception as e:
            print(f"Could not get DLL exports: {e}")

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

                get_last_error = kernel32.GetLastError
                format_message = kernel32.FormatMessageW
                format_message.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.POINTER(ctypes.c_wchar_p),
                                           ctypes.c_uint32, ctypes.c_void_p]
                format_message.restype = ctypes.c_uint32

                def get_error_message():
                    """Get detailed Windows error message"""
                    error_code = get_last_error()
                    message_buffer = ctypes.c_wchar_p()

                    format_message(
                        0x00001100,  # FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
                        None,
                        error_code,
                        0,  # Default language
                        ctypes.byref(message_buffer),
                        0,  # Size ignored due to FORMAT_MESSAGE_ALLOCATE_BUFFER
                        None
                    )

                    if message_buffer.value:
                        message = message_buffer.value
                        kernel32.LocalFree(message_buffer)
                        return f"Error {error_code}: {message}"
                    return f"Unknown error {error_code}"

                try:
                    # Load flags
                    LOAD_WITH_ALTERED_SEARCH_PATH = 0x8
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x100
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x1000

                    # Different loading strategies
                    loading_strategies = [
                        (LOAD_WITH_ALTERED_SEARCH_PATH, "LOAD_WITH_ALTERED_SEARCH_PATH"),
                        (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                         "LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS"),
                        (0, "Default loading")
                    ]

                    # Try loading JSON DLL with different strategies
                    json_path = str(lib_path.parent / "libjson-c.dll")
                    json_handle = None

                    print("\nTrying to load JSON DLL...")
                    for flags, strategy_name in loading_strategies:
                        print(f"\nAttempting with {strategy_name}")
                        json_handle = load_library_ex(json_path, None, flags)
                        if json_handle:
                            print(f"Successfully loaded JSON DLL with {strategy_name}")
                            break
                        else:
                            print(f"Failed: {get_error_message()}")

                    if not json_handle:
                        print("Could not load JSON DLL with any strategy")
                        return None

                    # Try loading RED API DLL with different strategies
                    red_handle = None
                    print("\nTrying to load RED API DLL...")
                    for flags, strategy_name in loading_strategies:
                        print(f"\nAttempting with {strategy_name}")
                        red_handle = load_library_ex(str(lib_path), None, flags)
                        if red_handle:
                            print(f"Successfully loaded RED API DLL with {strategy_name}")
                            break
                        else:
                            print(f"Failed: {get_error_message()}")

                    if not red_handle:
                        print("Could not load RED API DLL with any strategy")
                        return None

                    # Create CDLL objects
                    json_lib = ctypes.CDLL(json_path, handle=json_handle)
                    red_lib = ctypes.CDLL(str(lib_path), handle=red_handle)

                    # Print available functions
                    print("\nJSON DLL functions:")
                    for item in dir(json_lib):
                        if not item.startswith('_'):
                            print(f"- {item}")

                    print("\nRED API DLL functions:")
                    for item in dir(red_lib):
                        if not item.startswith('_'):
                            print(f"- {item}")

                    os.chdir(original_dir)
                    return red_lib

                except Exception as e:
                    print(f"Error during DLL loading: {e}")
                    traceback.print_exc()
                    os.chdir(original_dir)
                    return None
                finally:
                    os.chdir(original_dir)

            else:
                return ctypes.CDLL(str(lib_path))

        except Exception as e:
            print(f"Error in load_library: {e}")
            traceback.print_exc()
            return None

    def _init_system_config(self, config_path: str) -> bool:
        """Initialize system configuration"""
        try:
            print(f"Initializing system config with: {config_path}")
            if not hasattr(self, 'init_config_func'):
                print("Init config function not found")
                return False

            if not Path(config_path).exists():
                print(f"Config file not found: {config_path}")
                return False

            # Print config file contents for debugging
            try:
                with open(config_path, 'r') as f:
                    print("\nConfig file contents:")
                    print(f.read())
            except Exception as e:
                print(f"Error reading config file: {e}")

            self.init_config_func.argtypes = [ctypes.c_char_p]
            self.init_config_func.restype = ctypes.c_bool

            success = self.init_config_func(config_path.encode('utf-8'))
            if success:
                print("System configuration initialized successfully")
            else:
                print("Failed to initialize system configuration")
            return success
        except Exception as e:
            print(f"Error initializing system config: {e}")
            traceback.print_exc()
            return False

    def _get_library_path(self) -> Path:
        """Get the appropriate library path based on OS"""
        try:
            base_dir = Path(__file__).parent
            resources_dir = base_dir / 'resources'  # Remove platform.system().lower()

            lib_name = "red_api.dll" if platform.system() == 'Windows' else "libred_api.so"
            lib_path = resources_dir / lib_name

            if not lib_path.exists():
                print(f"Library not found at {lib_path}")
                return None

            # Check for JSON DLL too
            json_path = resources_dir / "libjson-c.dll"
            if not json_path.exists():
                print(f"JSON library not found at {json_path}")
                return None

            return lib_path
        except Exception as e:
            print(f"Error getting library path: {e}")
            return None

    def _init_system_config(self, config_path: str) -> bool:
        """Initialize system configuration"""
        try:
            print(f"Initializing system config with: {config_path}")
            # First make sure the file exists
            if not Path(config_path).exists():
                print(f"Config file not found: {config_path}")
                return False

            init_config = self.lib.init_system_config
            init_config.argtypes = [ctypes.c_char_p]
            init_config.restype = ctypes.c_bool
            success = init_config(config_path.encode('utf-8'))
            if success:
                print("System configuration initialized successfully")
            else:
                print("Failed to initialize system configuration")
            return success
        except Exception as e:
            print(f"Error initializing system config: {e}")
            traceback.print_exc()
            return False

    def _setup_functions(self) -> None:
        """Setup the C function interfaces"""
        if self.lib is None:
            print("Library is None, cannot setup functions")
            return False

        try:
            print("Setting up library functions...")

            # First initialize system configuration
            config_path = str(Path(__file__).parent / 'resources' / 'supported_systems.json')
            print(f"Loading configuration from: {config_path}")
            if not self._init_system_config(config_path):
                print("Failed to initialize system configuration")
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
            try:
                self.get_supported_systems_func = self.lib.get_supported_systems
                self.get_supported_systems_func.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
                self.get_supported_systems_func.restype = ctypes.POINTER(ctypes.c_char_p)
                print("Found get_supported_systems")
            except AttributeError as e:
                print(f"get_supported_systems not found: {e}")
                return False

            try:
                self.get_lamp_count_func = self.lib.get_lamp_count
                self.get_lamp_count_func.argtypes = [ctypes.c_char_p]
                self.get_lamp_count_func.restype = ctypes.c_uint32
                print("Found get_lamp_count")
            except AttributeError as e:
                print(f"get_lamp_count not found: {e}")
                return False

            try:
                self.get_red_function = self.lib.getREDFunction
                self.get_red_function.argtypes = [ctypes.c_char_p]
                self.get_red_function.restype = RED_FUNC
                print("Found getREDFunction")
            except AttributeError as e:
                print(f"getREDFunction not found: {e}")
                return False

            try:
                self.validate_parameters_func = self.lib.validate_parameters
                self.validate_parameters_func.argtypes = [
                    ctypes.c_char_p,  # systemType
                    ctypes.c_double,  # Flow
                    ctypes.c_double,  # UVT
                    ctypes.c_double,  # Power
                    ctypes.c_double  # Efficiency
                ]
                self.validate_parameters_func.restype = ctypes.c_bool
                print("Found validate_parameters")
            except AttributeError as e:
                print(f"validate_parameters not found: {e}")
                return False

            try:
                self.get_lamp_power_func = self.lib.get_lamp_power
                self.get_lamp_power_func.argtypes = [ctypes.c_char_p]
                self.get_lamp_power_func.restype = ctypes.c_double
                print("Found get_lamp_power")
            except AttributeError as e:
                print(f"get_lamp_power not found: {e}")
                return False

            print("Function setup complete")
            return True

        except Exception as e:
            print(f"Error setting up functions: {e}")
            traceback.print_exc()
            self.lib = None
            return False

    def process_query(self, query: str) -> Union[Dict, None]:
        """Process a natural language query using OpenAI"""
        if self.lib is None:
            return None

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a UV system calculation assistant. 
                    Convert user queries about RED calculations into appropriate function calls.
                    Return empty if the query cannot be answered with available functions.
                    Only use the provided functions."""
                }, {
                    "role": "user",
                    "content": query
                }],
                functions=FUNCTIONS,
                function_call="auto"
            )

            function_call = response.choices[0].message.function_call

            if not function_call:
                return None

            func_name = function_call.name
            args = json.loads(function_call.arguments)

            # Validate system type if present
            if "system_type" in args and args["system_type"] not in self.supported_systems:
                return None

            result = None
            if func_name == "get_supported_systems":
                result = self._get_supported_systems()
            elif func_name == "get_n_lamps":
                result = self._get_n_lamps(**args)
            elif func_name == "calculate_red":
                result = self._calculate_red(**args)

            if result is None:
                return None

            return {
                "function": func_name,
                "parameters": args,
                "result": result
            }

        except Exception:
            return None

    def _get_supported_systems(self) -> List[str]:
        """Get list of supported UV systems"""
        try:
            size = ctypes.c_size_t()
            systems_ptr = self.get_supported_systems_func(ctypes.byref(size))
            if not systems_ptr or size.value == 0:
                print("No systems found or empty pointer returned")
                return []

            print(f"Found {size.value} systems")
            systems = []
            for i in range(size.value):
                if systems_ptr[i]:
                    try:
                        system = systems_ptr[i].decode('utf-8')
                        systems.append(system)
                        print(f"Found system: {system}")
                    except Exception as e:
                        print(f"Error decoding system at index {i}: {e}")
            return systems
        except Exception as e:
            print(f"Error getting supported systems: {e}")
            traceback.print_exc()
            return []

    def _get_n_lamps(self, system_type: str) -> Optional[int]:
        """Get number of lamps for a system"""
        try:
            if system_type not in self.supported_systems:
                return None
            return self.get_lamp_count_func(system_type.encode('utf-8'))
        except Exception as e:
            print(f"Error getting lamp count: {e}")
            traceback.print_exc()
            return None

    def _calculate_red(self, system_type: str, flow: float, uvt: float,
                       uvt215: float = -1, d1_log: float = 18.0,
                       power_settings: dict = None,
                       efficiency_settings: dict = None) -> Optional[float]:
        """Calculate RED value with custom power and efficiency settings"""
        try:
            # Validate inputs
            if (system_type not in self.supported_systems or
                    not MIN_FLOW < flow < MAX_FLOW or
                    not MIN_UVT < uvt <= MAX_UVT):
                return None

            # Get number of lamps
            n_lamps = self.get_lamp_count_func(system_type.encode('utf-8'))
            if not n_lamps:
                return None

            # Initialize default arrays
            power = [100.0] * n_lamps  # Default 100% power
            efficiency = [80.0] * n_lamps  # Default 80% efficiency

            # Process power settings
            if power_settings:
                if 'all_lamps' in power_settings:
                    power = [float(power_settings['all_lamps'])] * n_lamps
                if 'specific_lamps' in power_settings:
                    for lamp_idx, value in power_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx) - 1  # Convert 1-based to 0-based indexing
                            if 0 <= idx < n_lamps:
                                power[idx] = float(value)
                        except (ValueError, IndexError):
                            continue

            # Process efficiency settings
            if efficiency_settings:
                if 'all_lamps' in efficiency_settings:
                    efficiency = [float(efficiency_settings['all_lamps'])] * n_lamps
                if 'specific_lamps' in efficiency_settings:
                    for lamp_idx, value in efficiency_settings['specific_lamps'].items():
                        try:
                            idx = int(lamp_idx) - 1  # Convert 1-based to 0-based indexing
                            if 0 <= idx < n_lamps:
                                efficiency[idx] = float(value)
                        except (ValueError, IndexError):
                            continue

            # Validate all values are within range
            if not all(MIN_POWER <= p <= MAX_POWER for p in power) or not all(MIN_EFFICIENCY <= e <= MAX_EFFICIENCY for e in efficiency):
                return None

            # Convert lists to ctypes arrays
            power_array = (ctypes.c_double * n_lamps)(*power)
            efficiency_array = (ctypes.c_double * n_lamps)(*efficiency)

            # Get and call RED calculation function
            red_func = self.get_red_function(system_type.encode('utf-8'))
            if not red_func:
                return None

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
                return None

            return result

        except Exception:
            return None

# Example usage:
def main():
    calculator = REDLibrary()

    # Example queries that include power and efficiency settings
    example_queries = [
        "Calculate RED for RZ-163-12 with flow 100, UVT 95%, all lamps at 80% power",  # This one works
        "Calculate RED for RZM-350-8 with flow 100, UVT 95%, all lamps at 85% power",  # Try with all lamps first
        "Calculate RED for RZMW-350-11 with flow 200, UVT 92%, all lamps at 80% power"  # Correct system name
    ]

    for query in example_queries:
        print(f"\nQuery: {query}")
        result = calculator.process_query(query)
        if result is None:
            print("Empty reply")
        else:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()