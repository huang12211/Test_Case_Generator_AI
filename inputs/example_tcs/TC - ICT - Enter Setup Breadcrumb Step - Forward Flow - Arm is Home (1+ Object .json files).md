**Test Case Title:** TC - ICT - Enter Setup Breadcrumb Step - Forward Flow - Arm is Home (1+ Object .json files)

**Test Case Objectives:** 
- Verify the landing panel when entering the Setup Breadcrumb Step is correct assuming that the arm is in Home position when more than one .json specification file for the registration object is installed on the system.
- Verify the sequence of panels during the Setup Breadcrumb Step are correct when navigating forwards when more than one .json specification file for the registration object is installed on the system and no errors occur.
- On each panel, verify the Breadcrumb Menu State & the panel content


**Requirements:** 
- ICT – Enter Setup Breadcrumb Step from Registration Workflow Selection Subpanel
- ICT – Setup Breadcrumb Step - Registration Breadcrumb Segment State
- COM - Enter Setup Breadcrumb Step - [Object/Pointer] - Arm Home - More than one SF
- COM – Select [Object/Pointer/Holder] SN Subpanel - Content
- COM - (Validation) Select [Object/Pointer/Holder] SN Subpanel - Transition to (Validation) Install [Object/Pointer/Holder] Subpanel
- COM - (Validation) Install [Object/Pointer/Holder] Subpanel - Content - more than one SF
- COM - (Validation) Install [Object/Pointer/Holder] Subpanel - Successful Verification

**Environment:** Robot

**Pre-conditions:**
1. ROSA Brain GEN II software version being tested is installed on the ROSA Recon Platform.
2. More than one 1 .json specification file for the Registration Object is installed on the ROSA Recon Platform.
3. A patient case containing at least one image sets in the validated tree has been opened in the Case Manager
4. Registration Workflow Selection Subpanel is displayed

**Test Parameters:** None.

**Test Steps:** 

Step: Ensure that all pre-conditions have beeen met.

Expected Result: N/A. 

Step: Click on the [Intra-op CT Registration] card. Take a screenshot.

Expected Result: "Select Object Serial Number Subpanel" is displayed.
- Stationary Robotic Mode is displayed
- [Setup] breadcrumb step is activated
- all other breadcrumb steps are disabled
- Select Object Serial Number Subpanel Instruction is displayed
- Serial Number List containing enabled Serial Numbers that are associated with the available SF files for the Object
- Object image is displayed
- [Cancel Registration] button is enabled
- [Next] button is disabled and highlighted
A screenshot is recorded in the Actual Result cell

Step: Select a [Serial Number] and write down the selected number

Expected Result: Selected Serial number is recorded in the Actual Result cell.

Step: Click on the [Next] button. Take a screenshot.

Expected Result: the "Install Object Subpanel" is displayed containing:
- Stationary Robotic Mode
- an activated [Setup] breadcrumb step
- all other breadcrumb steps are disabled
- Install Object Instruction
- Verify the serial number message
- the serial number that was selected in the screenshot recorded in step 5
- No force applied message
- Install Object Image
- [Change Serial Number] button
- [Verify] button
A screenshot is recorded in the Actual Result cell.

Step: 	Click on the [Verify] button

Expected Result: N/A.

Step: Wait for the Tool Verification Modal to disappear. Take a screenshot

Expected Result: the "Move to Working Subpanel" is displayed containing:
- Automatic Robotic Mode
- an activated [Setup] breadcrumb step
- all other breadcrumbs are disabled
- a Move to Working Image
- a Press Pedal Icon
- the Move to Working Message
A screenshot is recorded in the Actual Result cell.