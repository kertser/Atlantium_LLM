# RED Calculation API Methods Documentation

## Overview

This document provides information about the methods available in the `libred_api` for RED (Reduction Equivalent Dose) calculation. These methods allow users to interact with various UV systems to perform RED calculations.

Refer to explicit usage examples in `calculator.py` for practical implementation details.

---

## Setup Procedures

### Windows Setup
1. **Install Dependencies:**
    - Install [CMake](https://cmake.org/download/).
    - Ensure you have a C compiler like MinGW or Visual Studio.
2. **Build the Project:**
    - Open a terminal and navigate to the project directory.
    - Run the following commands:
      ```sh
      mkdir build && cd build
      cmake .. -G "MinGW Makefiles"
      mingw32-make
      ```
3. **Library Path Configuration:**
    - Ensure the `libred_api.dll` is in the same directory as your executable or included in the system's library path.
    - Ensure that `supported_systems.json` is in the same directory as your executable.
    - Ensure `libjson-c.dll` is in the same directory as your executable.

### Linux Setup
1. **Install Dependencies:**
    - Install CMake, GCC, and any other necessary build tools using your package manager (e.g., `sudo apt install cmake build-essential`).
    - Ensure `json-c` development libraries are available.
2. **Build the Project:**
    - Open a terminal and navigate to the project directory.
    - Run the following commands:
      ```sh
      mkdir build && cd build
      cmake ..
      make
      ```
3. **Library Path Configuration:**
    - Ensure `libred_api.so.1` and `libred_api.so.1.0` are in a directory included in your `LD_LIBRARY_PATH`, or copy it to `/usr/lib`.
    - Ensure that `supported_systems.json` is in the same directory as your executable.
    - Ensure `libjson-c.so.5` and `libjson-c.so.5.3.0` are in the same directory as your executable.

### Containerized Usage
For automated builds and containerized execution, a `Dockerfile` and `docker-compose.yml` are provided.

#### Steps:
1. Build the container using Docker:
   ```sh
   docker build -t red-calculation-api .
   ```
2. Run the container:
   ```sh
   docker run --rm -v $(pwd):/app -w /app/build red-calculation-api
   ```
3. The containerized environment includes all dependencies and supports both Linux and Windows builds.
    - For Windows builds, ensure Docker Desktop is configured to use Windows containers.
    - The build script in the `docker-compose.yml` handles library copying and symbolic linking.

---

## Typical Usage

```python
from calculator import REDLibrary

# Initialize the REDLibrary with your OpenAI API key
api_key = "your_openai_api_key"
calculator = REDLibrary(api_key=api_key)

# Example query
query = "Calculate RED for RZM-350-8 with flow 100, UVT 95%, lamp 1 at 90% power and lamp 2 at 80% power"

# Process the query
result = calculator.process_query(query)

# Print the result
if 'result' in result:
    print(f"Calculated RED: {result['result']['result']} mJ/cm²")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

---

## Use Cases

### Example for Python Code
```python
from calculator import REDLibrary

# Initialize the REDLibrary with your OpenAI API key
api_key = "your_openai_api_key"
calculator = REDLibrary(api_key=api_key)

# Example queries
queries = [
    "Calculate RED for RZM-350-8 with flow 100, UVT 95%, lamp 1 at 90% power and lamp 2 at 80% power",
    "Calculate RED for RZMW-350-11 with flow 200, UVT 92%, lamp 1 efficiency 85% and lamp 2 efficiency 75%, all lamps at 80% power",
    "Calculate RED for RZ-163-12 with flow 100, UVT 95%, lamp 1 efficiency 90%, lamp 2 efficiency 85%"
]

for query in queries:
    result = calculator.process_query(query)
    if 'result' in result:
        print(f"Query: {query}")
        print(f"Calculated RED: {result['result']['result']} mJ/cm²")
    else:
        print(f"Query: {query}")
        print(f"Error: {result.get('error', 'Unknown error')}")
```

---

## Methods

### `ListOfSupportedSystems`
Returns the list of supported UV systems.

**Prototype:**
```python
def _get_supported_systems(self) -> List[str]:
```

### `getREDFunction`
Selects the appropriate RED calculation function based on the system type.

**Prototype:**
```python
def get_red_function(self, system_type: str) -> Callable:
```

### `getNLamps`
Returns the standard number of lamps for a given system type.

**Prototype:**
```python
def _get_n_lamps(self, system_type: str) -> Dict[str, Union[int, str]]:
```

### `validate_parameters`
Validates operational parameters against system configuration.

**Prototype:**
```python
def validate_parameters(self, system_type: str, flow: float, uvt: float, power: float, efficiency: float) -> bool:
```

---

## Supported Systems

The following systems are supported (refer to `supported_systems.json` for operational limits):

- RZ-104-11
- RZ-104-12
- RZ-163-11
- RZ-163-12
- RZ-163-13
- RZ-163-14
- RZ-163HP-11
- RZ-163HP-12
- RZ-163HP-13
- RZ-163HP-14
- RZ-163UHP-11
- RZ-163UHP-12
- RZ-163UHP-13
- RZ-163UHP-14
- RZ-300-HDR
- RS-104
- RZB-300
- RZM-350-8
- RZM-350-5
- RZM-200-5
- RZM-200-3
- RZM-200-2
- RZMW-350-11
- RZMW-350-7

---

## License

This project is proprietary to Atlantium company. Unauthorized copying, modification, or distribution of this software is strictly prohibited.

---

## Contact

For any inquiries or support, please contact [mikek@atlantium.com](mailto:mikek@atlantium.com).