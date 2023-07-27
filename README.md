# bci_2023_intern
my internship period works on bci


## RFC Folder (Record Filter Classifier)
 
* rfc_main.py : main application file, Run the application from here.

* rfc_genFunction : general function file.

* rfc_markerUI : Experiment UI file, the ui that will appear when you start recording the experiment.

*** Note that the "rfc_main.py" has to be run together with lsl streaming (or UI won't appear). In my application, i use Unicorn EEG lsl, streaming through OpenViBE Aquisition Server and Designer. ***

## Data Folder

* noisyecg (.mat/.csv) : 3 ecg signal with noise sample.

* FocusC(-G)_raw (.fif) : Saved raw eeg signal file from recorded from experiment done on rfc. *** (note that it is not an appropriate standardize experiment) ***

## Scratch Folder

Day and topics that are scratchly written to figure out how some of the code works, may be there is some example useful here
### Messaging Folder

Scratchly written python of sending command acquiring from eeg signal threshold triggering to unreal engine 5 socket

#### TCP to unreal Folder

Scratchly written python of connecting TCP socket of python to unreal engine 5 using Unreal Engine TCP Socket plugin.
