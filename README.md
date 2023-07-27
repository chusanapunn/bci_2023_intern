# bci_2023_intern
my internship period works on bci during 1/6/23 to 27/7/23 at Mahidol BCI lab
RFC application is created to support the research works on classifying attention state and non attention state. Which requires good signal measuring standard and good protocol to manipulate the data (shifting the epoch length, increase the epoch count, or rebalancing the epoch class amount).

## RFC Folder
The Main Application created during the internship period, RFC is Recorder, Filterer to Classifier for EEG lsl experiment function.
 More details inside the code comments (ONLY RFC folder will have detailed comments)

* rfc_main.py : main application file, Run the application from here.

    -rfc UI divides into 2 main panels, plot panel for signal monitoring which the current version display signal in a [channel(raw + filtered)+channel()+...+ StimulusChannel()] vertically for plot_duration interval
    
    ### Record Panel
    * Start: start recording the signal.
    * Stop: stop recording the signal, *** also stops the plot and update function - mostly to support the calculation process after finish the record session ***
    * Reset: (kinda Deprecated) Should have reset the buffer and the variables value of the application to start new experiment. But for now, restart the application would be the most appropriate way.
    * PlotRaw: Plot the recorded signal with mne plot, you can toggle the Inlet Source between General and Unicorn to divides the unit of measurement.
    * LogPanel: Display Overall Input datapoint amount, data array to be converted to raw data as raw data count, marker data count, buffer size, number of streaming channel, and fixation text of the events labels.
    * SaveRaw: Save the record signal into filename according to the savefile name textbox on the same line.
    * Browse: Browse saved record from your own devices and show the file to be loaded on the loadfile name textbox.
    * UseLoadedRaw: Use recorded signal from the browsed file name.

    ### Preprocessing Panel
    * PlotSpectral: Plot Power Spectral Density of all signals comparing between filtered and non-filtered signal
    * PlotEvents: Plot Signal with Events together.
    * PlotEpochs: Plot Spectrogram and Z Score of spectogram from 2 types of epoch for comparison. This Button will requires Raw signal plot with events to generate Epochs. Which is using to generate x_train for data training.
    * PlotPCA: Plot PCA Cluster and t-SNE cluster for all x_train data receive from PlotEpochs.

    ### Feature Extraction Panel
    * RadioButton, which have count = all streaming channel -1 (not using stimulus channel), = Currently selected channel to be used for feature extraction. *** note that feature extraction process is working on only one channel for each time, therefore, to feature extract from more than 1 channel, you have to modify the code furthermore. ***
    * Filterbank: Using Butterworth Filter to filter signal into significant frequency band (Delta/alpha/beta...), the code will auto assign alpha and beta as new x_train data.
    * DiscreteWavelets: Deprecate, as the dwt process downsampling the signal data, therefore, there is not enough data to use for training.
    * HilbertTransform: Using Hilbert Transform to decompose the signal into IMFS signal. Default coding will assign the IMF-1 and IMF-2 as new x_train data similarly to Filterbank 2 band wave.

    ### Classify Panel
    * Classify LogReg: Classify the data using x_train and y_train, either from PlotEpochs, FilterBank, or HilbertTransform, by LogRegression method. Output accuracy, confusion matrix with some evaluation matrix.
    * Classify SVM: Classify data using Support Vector Machine. Output accuracy, confusion matrix with some evaluation matrix.
    * Classify DNN: Classify data using simple perceptrons neural networks dense layers to classify roughly. Output accuracy, confusion matrix with some evaluation matrix.

    ### Inlet Source Panel
    * Inlet Source: Deprecated. Suppose to switch the units for alternative inlet source. Using only General would be fine.


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
