**Test Case Title:** TC - MKR - Setup Breadcrumb Step State - Nav Forwards and Backwards - Arm is Home at Start (1+ Pointer .json file)

**Test Case Objectives:** 
- Verify Breadcrumb Menu State while executing the sequence of panels falling under the Setup Breadcrumb Step during Marker Registration when navigating forwards and backwards assuming that the arm is in Home position at the start and that more than one .json specification file for the pointer is installed on the system.

**Requirements:**
- MKR - Setup Breadcrumb Step - Registration Breadcrumb Segment State
- REG - GEN – Validation Breadcrumb Segment State

**Environment:** Robot

**Pre-conditions:**
1. ROSA Brain GEN II software version being tested is installed on the ROSA Recon Platform.
2. More than 1 .json specification file for the Pointer is installed on the ROSA Recon Platform.
3. A patient case containing at least one image sets in the validated tree has been opened in the Case Manager
4. Define Virtual Marker Subpanel is displayed and the [Next] button is enabled

**Test Parameters:**
None.

**Test Steps:** 

Step: Ensure that all pre-conditions have been met.

Expected Result: N/A.

Step: Click on the [Next] button. Take a screenshot.

Expected Result: The "Select Pointer Serial Number Subpanel" is displayed containing:
- an enabled [Define markers] breadcrumb step
- an activated [Setup] breadcrumb step
- a disabled [Locate Markers] breadcrumb step
- a disabled [Results] breadcrumb step
- a disabled [Validation tool] breadcrumb step
- a disabled [Validation] breadcrumb step
A screenshot is recorded in the Actual Result cell.

Step: Select a [Serial Number].

Expected Result: N/A.

Step: Click on the [Next] button. Take a screenshot.

Expected Result: The "Install Pointer Subpanel" is displayed containing:
- an enabled [Define markers] breadcrumb step
- an activated [Setup] breadcrumb step
- a disabled [Locate Markers] breadcrumb step\
- a disabled [Results] breadcrumb step
- a disabled [Validation tool] breadcrumb step
- a disabled [Validation] breadcrumb step
A screenshot is recorded in the Actual Result cell.

Step: Click on the [Change serial number] button. Take a screenshot.

Expected Result: the "Select Pointer Serial Number Subpanel" is displayed containing:
- an enabled [Define markers] breadcrumb step
- an activated [Setup] breadcrumb step
- a disabled [Locate Markers] breadcrumb step\
- a disabled [Results] breadcrumb step
- a disabled [Validation tool] breadcrumb step
- a disabled [Validation] breadcrumb step
A screenshot is recorded in the Actual Result cell.

Step: Click on the [Next] button.

Expected Result: N/A

Step: Click on the [Verify] button. Take a screenshot

Expected Result: The "Move to Working Subpanel" is displayed containing:
- an enabled [Define markers] breadcrumb step
- an activated [Setup] breadcrumb step
- a disabled [Locate Markers] breadcrumb step\
- a disabled [Results] breadcrumb step
- a disabled [Validation tool] breadcrumb step
- a disabled [Validation] breadcrumb step
A screenshot is recorded in the Actual Result cell.