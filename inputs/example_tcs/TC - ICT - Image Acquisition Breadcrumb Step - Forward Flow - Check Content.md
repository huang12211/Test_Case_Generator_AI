**Test Case Title:** TC - ICT - Image Acquisition Breadcrumb Step - Forward Flow - Check Content

**Test Case Objectives:** 
- Verify the forwards flow while in the Image Acquisition Breadcrumb Step
- On each panel, verify the Breadcrumb Menu State and the panel's content.

**Requirements:**
- ICT – Transition from Moving to Working Position Subpanel to Image Acquisition breadcrumb step
- ICT - Image Acquisition Breadcrumb Step - Registration Breadcrumb Segment State
- ICT – Place Object Subpanel - Content
- ICT – Transition from Place Object Subpanel to Image Acquisition Subpanel
- ICT – Image Acquisition Subpanel - Content

**Environment:** Robot

**Pre-conditions:**
1. ROSA Brain GEN II software version being tested is installed on the ROSA Recon Platform.
2. a patient case containing at least one image sets in the validated tree has been opened in the Case Manager
3. the intra-op CT registration method was selected in the Registration Workflow Selection Subpanel
4. the application is in the "Moving to Working Subpanel" of the Setup Breadcrumb step

**Test Parameters:**
None.

**Test Steps:** 

Step: Ensure that all pre-conditions have been met.

Expected Result: N/A.

Step: Press the vigilance device until the robot arm reaches the Working position. Take a screenshot.

Expected Result: "Place Object Subpanel" is displayed containing:
- Collaborative Robotic Mode
- an enabled [Setup] breadcrumb step
- an activated [Image Acquisition] breadcrumb step
- a disabled [Image Transfer] breadcrumb step
- a disabled [Registration] breadcrumb step
- Place object image
- A Pedal icon
- Press pedal message
- Place object message
- Adjust Speed Section: [Slow] and [Fast] options with the [Fast] option selected by default
- disabled and highlighted [Acquisition] button
A screenshot is recorded in the Actual Results cell.

Step: Press and release the vigilance device.

Expected Result: N/A.

Step: Click on the [Acquisition] button without applying any force on the registration object. Take a screenshot.

Expected Result: Image Acquisition Subpanel" is displayed containing:
- Stationary Robotic Mode
- an enabled [Setup] breadcrumb step
- an activated [Image Acquisition] breadcrumb step
- a disabled [Image Transfer] breadcrumb step
- a disabled [Registration] breadcrumb step
- Image Acquisition Image
- Image Acquisition Message
- an enabled [Reposition] button
- and enabled and highlighted [Image Acquired] button
A screenshot is recorded in the Actual Results cell.