
# RED Calculation API Methods Documentation

## Overview

This document provides an information about the methods available in the `libred_api.dll` for RED (Reduction Equivalent Dose) calculation. 
<br>These methods allow users to interact with various UV systems to perform RED calculations.
<br>(see the explicit example of usage in main.c)

** Typical Usage **
```c
// Solve for RED, using previously declared parameters
REDFunction redFunction = getREDFunction(systemType);
double result = redFunction(Flow, UVT, UVT215, P, Eff, D1Log, NLamps);
```
Here we define the values for 
- Flow [m^3h] 
- UVT254 and UVT215 [%-1cm]
- Power and Efficiency [%]
- 1-Log inactivation dose [mJ/cm^2] 
- Number of lamps.
<br><sup>(One shall set the NLamps to 0 if he wishes to init the number of lamps automatically)</sup>

# Example for C code 
<sup>Make sure the libred_api.dll is in the same directory as your executable or in a directory included in your system's library path.</sup>:
```c
// declare system type
char systemType[] = "RZM-350-8"; // System Type

// Declare the parameters:
uint32_t NLamps = 0; // 0 is for automatic detection

// Get the number of lamps from the system type, unless manually set:
if (NLamps == 0) NLamps = getNLamps(systemType);

double Flow = 100; //[m^3/h]
double UVT = 95; //[%-1cm]
double UVT215 = -1; //[%-1cm] or -1 if NaN
double *P = (double *)malloc(NLamps * sizeof(double));
double *Eff = (double *)malloc(NLamps * sizeof(double));


// Initialize the P and Eff values to some set value
for (uint32_t i = 0; i < NLamps; i++) {
	P[i] = 100; // [%], defined for every lamp
	Eff[i] = 80; //[%], defined for every lamp
}

double D1Log = 18; // [mJ/cm^2]
```

# Example for Java (using the dll):
<sup>Make sure the libred_api.dll is in the same directory as your Java program or in a directory included in your system's library path.</sup>
```java
public class RedApi {
    // Declare the native methods
    public native int getNLamps(String systemType);
    public native double calculateRed(
        String systemType, double flow, double uvt, double uvt215, 
        double[] power, double[] efficiency, double d1Log, int nLamps
    );

    static {
        // Load the DLL
        System.loadLibrary("libred_api");
    }

    public static void main(String[] args) {
        RedApi api = new RedApi();
        
        String systemType = "RZM-350-8";
        double flow = 100; // [m^3/h]
        double uvt = 95; // [% - 1cm]
        double uvt215 = -1; // [% - 1cm] or -1 if NaN
        double power = 100; // [%], defined for every lamp
        double efficiency = 80; // [%], defined for every lamp
        double d1Log = 18; // [mJ/cm^2]

        int nLamps = api.getNLamps(systemType);
        double[] powerArray = new double[nLamps];
        double[] efficiencyArray = new double[nLamps];

        for (int i = 0; i < nLamps; i++) {
            powerArray[i] = power;
            efficiencyArray[i] = efficiency;
        }

        double result = api.calculateRed(systemType, flow, uvt, uvt215, powerArray, efficiencyArray, d1Log, nLamps);
        System.out.println("Calculated RED for " + systemType + " = " + result);
    }
}

```

## Methods

### `ListOfSupportedSystems`

Returns the list of supported UV systems.

**Prototype:**
```c
const char** ListOfSupportedSystems(size_t* size);
```

**Parameters:**
- `size`: Pointer to a `size_t` variable where the function will store the number of supported systems.

**Returns:**
- A pointer to an array of strings, each representing a supported system.

### `getREDFunction`

Selects the appropriate RED calculation function based on the system type.

**Prototype:**
```c
REDFunction getREDFunction(char *systemType);
```

**Parameters:**
- `systemType`: A string representing the system type identifier.

**Returns:**
- A function pointer to the selected RED calculation function.

### `getNLamps`

Returns the standard number of lamps for a given system type.

**Prototype:**
```c
uint32_t getNLamps(char *systemType);
```

**Parameters:**
- `systemType`: A string representing the system type identifier.

**Returns:**
- The standard number of lamps for the specified system type.

## Supported Systems

The `supported_systems.txt` file lists all supported UV systems:

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

For the full list, refer to the `src/supported_systems.txt` file.

## License

This project is proprietary to Atlantium company. 
<br>Unauthorized copying, modification, or distribution of this software is strictly prohibited.

## Contact

For any inquiries or support, please contact [mikek@atlantium.com](mailto:mikek@atlantium.com).
