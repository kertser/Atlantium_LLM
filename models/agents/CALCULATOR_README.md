# RED Calculation API Methods Documentation

## Overview

This document provides information about the methods available in the `libred_api` for RED (Reduction Equivalent Dose) calculation. These methods allow users to interact with various UV systems to perform RED calculations.

Refer to explicit usage examples in `main.c` and `red_api.c` for practical implementation details.

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

```c
// Solve for RED, using previously declared parameters
REDFunction redFunction = getREDFunction(systemType);
double result = redFunction(Flow, UVT, UVT215, P, Eff, D1Log, NLamps);
```
Here we define the values for:
- Flow [m³/h]
- UVT254 and UVT215 [%-1cm]
- Power and Efficiency [%]
- 1-Log inactivation dose [mJ/cm²]
- Number of lamps.

Set `NLamps` to 0 if you wish to initialize the number of lamps automatically.

---

## Use Cases

### Example for C Code
```c
#include "system_config.h"

char systemType[] = "RZM-350-8";
uint32_t NLamps = getNLamps(systemType);
double Flow = 100;  // [m³/h]
double UVT = 95;    // [% -1cm]
double UVT215 = -1; // [% -1cm] or -1 if NaN
double P[8] = {100, 100, 100, 100, 100, 100, 100, 100};
double Eff[8] = {80, 80, 80, 80, 80, 80, 80, 80};
double D1Log = 18;  // [mJ/cm²]

REDFunction redFunction = getREDFunction(systemType);
double result = redFunction(Flow, UVT, UVT215, P, Eff, D1Log, NLamps);
printf("Calculated RED: %.2f mJ/cm²\n", result);
```

### Example for Java Integration
```java
public class RedApi {
    // Declare the native methods
    public native int getNLamps(String systemType);
    public native double calculateRed(
        String systemType, double flow, double uvt, double uvt215, 
        double[] power, double[] efficiency, double d1Log, int nLamps
    );

    static {
        // Load the native library
        System.loadLibrary("libred_api");
    }

    public static void main(String[] args) {
        RedApi api = new RedApi();

        String systemType = "RZM-350-8";
        double flow = 100.0; // [m³/h]
        double uvt = 95.0;   // [% -1cm]
        double uvt215 = -1.0; // [% -1cm] or -1 if NaN
        double d1Log = 18.0; // [mJ/cm²]

        // Retrieve the number of lamps for the system
        int nLamps = api.getNLamps(systemType);
        double[] power = new double[nLamps];
        double[] efficiency = new double[nLamps];

        for (int i = 0; i < nLamps; i++) {
            power[i] = 100.0; // [%]
            efficiency[i] = 80.0; // [%]
        }

        // Perform RED calculation
        double result = api.calculateRed(systemType, flow, uvt, uvt215, power, efficiency, d1Log, nLamps);
        System.out.println("Calculated RED for system " + systemType + " = " + result + " mJ/cm²");
    }
}
```

---

## Methods

### `ListOfSupportedSystems`
Returns the list of supported UV systems.

**Prototype:**
```c
const char** ListOfSupportedSystems(size_t* size);
```

### `getREDFunction`
Selects the appropriate RED calculation function based on the system type.

**Prototype:**
```c
REDFunction getREDFunction(char *systemType);
```

### `getNLamps`
Returns the standard number of lamps for a given system type.

**Prototype:**
```c
uint32_t getNLamps(char *systemType);
```

### `validate_parameters`
Validates operational parameters against system configuration.

**Prototype:**
```c
bool validate_parameters(const char* system_type, double flow, double uvt, double power, double efficiency);
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

