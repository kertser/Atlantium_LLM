from openai import OpenAI
import json
from typing import Dict, List, Optional, Union
import ctypes
import os
from dotenv import load_dotenv
from pathlib import Path
import platform

load_dotenv("../.env")

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
        try:
            self.lib = self._load_library()
            self._setup_functions()
            self.supported_systems = self._get_supported_systems()
        except Exception as e:
            print(f"Initialization error: {e}")
            self.lib = None

    def _get_library_path(self) -> Path:
        """Get the appropriate library path based on OS"""
        try:
            base_dir = Path(__file__).parent
            resources_dir = base_dir / 'resources'

            lib_name = "libred_api.dll" if platform.system() == 'Windows' else "libred_api.so"
            lib_path = resources_dir / lib_name

            if not lib_path.exists():
                raise FileNotFoundError(f"Library not found at {lib_path}")

            return lib_path
        except Exception:
            return None

    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Load the appropriate library based on the operating system"""
        try:
            lib_path = self._get_library_path()
            if lib_path is None:
                return None

            if platform.system() == 'Windows':
                os.environ['PATH'] = str(lib_path.parent) + os.pathsep + os.environ['PATH']
                return ctypes.WinDLL(str(lib_path))
            else:
                return ctypes.CDLL(str(lib_path))
        except Exception:
            return None

    def _setup_functions(self) -> None:
        """Setup the C function interfaces"""
        if self.lib is None:
            return

        try:
            self.lib.ListOfSupportedSystems.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
            self.lib.ListOfSupportedSystems.restype = ctypes.POINTER(ctypes.c_char_p)

            self.lib.getNLamps.argtypes = [ctypes.c_char_p]
            self.lib.getNLamps.restype = ctypes.c_uint32

            self.lib.getREDFunction.argtypes = [ctypes.c_char_p]
            self.lib.getREDFunction.restype = ctypes.CFUNCTYPE(
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_uint32
            )
        except Exception:
            self.lib = None

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
            systems_ptr = self.lib.ListOfSupportedSystems(ctypes.byref(size))
            return [systems_ptr[i].decode('utf-8') for i in range(size.value)]
        except Exception:
            return []

    def _get_n_lamps(self, system_type: str) -> Optional[int]:
        """Get number of lamps for a system"""
        try:
            if system_type not in self.supported_systems:
                return None
            return self.lib.getNLamps(system_type.encode('utf-8'))
        except Exception:
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
            n_lamps = self._get_n_lamps(system_type)
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
            red_func = self.lib.getREDFunction(system_type.encode('utf-8'))
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
        "Calculate RED for RZ-163-12 with flow 100, UVT 95%, all lamps at 80% power",
        "Calculate RED for RZM-350-8 with flow 100, UVT 95%, lamp 1 at 90% power and lamp 2 at 80% power",
        "Calculate RED for RZM-350-11 with flow 200, UVT 92%, lamp 1 efficiency 85% and lamp 2 efficiency 75%",
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