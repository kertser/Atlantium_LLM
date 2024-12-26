
# RED Calculation API Methods Documentation

## Overview

This document provides an information about the methods available in the `libred_api.dll` or `libred_api.so` 
for <b>RED</b> (Reduction Equivalent Dose) calculation. 
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
<sup>Make sure the libred_api.dll or libred_api.so is in the same directory as your 
executable or in a directory included in your system's library path.</sup>:
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
