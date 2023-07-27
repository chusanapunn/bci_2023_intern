# bci_2023_intern
my internship period works on bci


# RFC Folder (Record Filter Classifier)
 
* rfc_main.py : main application file, Run the application from here.

* rfc_genFunction : general function file.

* rfc_markerUI : Experiment UI file, the ui that will appear when you start recording the experiment.

*** Note that the "rfc_main.py" has to be run together with lsl streaming (or UI won't appear). In my application, i use Unicorn EEG lsl, streaming through OpenViBE Aquisition Server and Designer. ***

## Data Folder

* noisyecg (.mat/.csv) : 3 ecg signal with noise sample.

* FocusC(-G)_raw (.fif) : Saved raw eeg signal file from recorded from experiment done on rfc. *** (note that it is not an appropriate standardize experiment) ***