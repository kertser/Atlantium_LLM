# RED Calculation API Methods Documentation

## Overview

This document provides detailed information about the methods available in the `libred_api` for RED (Reduction Equivalent Dose) calculation. These methods enable interaction with various UV systems to perform precise dose calculations and validate operational parameters.

---

## Setup Procedures

### Windows Setup

1. **Install Dependencies:**
    - Install [CMake](https://cmake.org/download/).
    - Ensure a C compiler is installed (e.g., MinGW or Visual Studio).
2. **Build the Project:**
    ```bash
    mkdir build && cd build
    cmake .. -G "MinGW Makefiles"
    mingw32-make
    ```
3. **Library Path Configuration:**
    - Place `libred_api.dll` in the same directory as your executable or system's library path.
    - Place `supported_systems.json` and `libjson-c.dll` in the executable directory.

### Linux Setup

1. **Install Dependencies:**
    - Run: `sudo apt install cmake build-essential`
    - Ensure `json-c` development libraries are installed.
2. **Build the Project:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```
3. **Library Path Configuration:**
    - Ensure `libred_api.so.1` and `libred_api.so.1.0` are in `LD_LIBRARY_PATH` or `/usr/lib`.
    - Place `supported_systems.json` and `libjson-c.so.5` in the executable directory.

### Containerized Usage

1. **Build the Container:**
    ```bash
    docker build -t red-calculation-api .
    ```
2. **Run the Container:**
    ```bash
    docker run --rm -v $(pwd):/app -w /app/build red-calculation-api
    ```
3. **Environment Support:**
    - Includes all dependencies for Linux and Windows builds.
    - Handles library copying and symbolic linking.

---

## Typical Usage

### Python Example

```python
from calculator import REDLibrary

# Initialize the REDLibrary
api_key = "your_openai_api_key"
calculator = REDLibrary(api_key=api_key)

# Query example
query = "Calculate RED for RZM-350-8 with flow 100, UVT 95%, lamp 1 at 90% power and lamp 2 at 80% power"
result = calculator.process_query(query)

# Print results
if 'result' in result:
    print(f"Calculated RED: {result['result']['result']} mJ/cm²")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

---

## Core Methods

### `ListOfSupportedSystems`

Returns a list of supported UV systems.
```python
def _get_supported_systems(self) -> List[str]:
```

### `getREDFunction`

Selects the appropriate RED calculation function for a system type.
```python
def get_red_function(self, system_type: str) -> Callable:
```

### `getNLamps`

Returns the standard number of lamps for a given system type.
```python
def _get_n_lamps(self, system_type: str) -> Dict[str, Union[int, str]]:
```

### `validate_parameters`

Validates operational parameters against system configuration.
```python
def validate_parameters(self, system_type: str, flow: float, uvt: float, power: float, efficiency: float) -> bool:
```

---

## Supported Systems

Refer to `supported_systems.json` for detailed operational limits. Supported systems include:

- **Single Lamp Systems:**
  - RS-104

- **Multi-Lamp Systems:**
  - RZ-104-11, RZ-104-12
  - RZ-163-11, RZ-163-12, RZ-163-13, RZ-163-14
  - RZ-163HP/UHP series

- **Medium Pressure Systems:**
  - RZM-350-8, RZM-350-5
  - RZM-200 series
  - RZMW-350-11, RZMW-350-7

---

## Error Handling

### Common Scenarios

1. **Invalid System Type:** The system type does not exist in `supported_systems.json`.
2. **Out-of-Range Parameters:** Flow, UVT, power, or efficiency values exceed system limits.
3. **Calculation Errors:** Errors during dose computation.
4. **API Communication Issues:** Connectivity problems with OpenAI.
5. **JSON Parsing Errors:** Incorrect data format in configuration or input files.

### Standardized Result Format

Results follow this format:
```json
{
    "result": {
        "result": float,  // RED value in mJ/cm²
        "system": str,    // System type
        "parameters": {
            "flow": float,
            "uvt": float,
            "power": [float],
            "efficiency": [float]
        }
    }
}
```

---

## License

This project is proprietary to Atlantium. Unauthorized use, modification, or distribution is strictly prohibited.

---

## Contact

For inquiries or support, contact [mikek@atlantium.com](mailto:mikek@atlantium.com).

