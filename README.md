# bci_2023_intern
my internship period works on bci during 1/6/23 to 27/7/23 at Mahidol BCI lab

## RFC Folder
The Main Application created during the internship period, RFC is Recorder, Filterer to Classifier for EEG lsl experiment function.
 More details inside the code comments (ONLY RFC folder will have detailed comments)

* rfc_main.py : main application file, Run the application from here.

    -rfc UI divides into 2 main panels, plot panel for signal monitoring which the current version display signal in a channel(raw + filtered)+channel()+... vertically for plot_duration interval
    
    ### Record Panel
    -Start: start recording the signal.
    -Stop: stop recording the signal, *** also stops the plot and update function - mostly to support the calculation process after finish the record session ***
    -Reset: (kinda Deprecated) Should have reset the buffer and the variables value of the application to start new experiment. But for now, restart the application would be the most appropriate way.
    -PlotRaw: Plot the recorded signal with mne plot, you can toggle the Inlet Source between General and Unicorn to divides the unit of measurement.



    -InletSource: Deprecated. Using only General would be fine.


* rfc_genFunction : general function file.

* rfc_markerUI : Focus Experiment UI file, the ui that will appear when you start recording the experiment. The protocol of the marker is, randomly draw green circle on white fullscreen between the interval of 5-6, stay appearing for 3 seconds and disappear for another sequence of appearing. 
In which we will use for epoching that are offset from the initiation of the circle to the end at +- 0.5 seconds.

*** Note that the "rfc_main.py" has to be run together with lsl streaming (or UI won't appear). In my application, i use Unicorn EEG lsl, streaming through OpenViBE Aquisition Server and Designer. ***

## Data Folder

* noisyecg (.mat/.csv) : 3 ecg signal with noise sample.

* FocusC(-G)_raw (.fif) : Saved raw eeg signal file from recorded from Focus experiment done on rfc. 288 Epochs (144 Focus, 144 Non Focus, 0.25 s per epochs. at 250 sampling rate)
 *** (note that it is not an appropriate standardize experiment) ***

## Scratch Folder

Day and topics that are scratchly written to figure out how some of the code works, may be there is some example useful here
### Messaging Folder

Scratchly written python of sending command acquiring from eeg signal threshold triggering to unreal engine 5 socket

#### TCP to unreal Folder

Scratchly written python of connecting TCP socket of python to unreal engine 5 using Unreal Engine TCP Socket plugin.
