# Signal Editor

GUI program for editing and analyzing physiological signals.

## How to use (work in progress)

The following file types can be read by this program: 
- European Data Format (.edf)
- Text-based formats (.txt, .csv, .tsv)
- Arrow IPC / Feather (.ipc, .feather)

File Selection:

(.edf files are assumed to have 3 channels: temperature, hbr, ventilation)

1. Click Select File and choose a compatible data file to load
2. If you don't want to work with the entire dataset, you can specify a column along with an upper and lower value, and it will only load rows where the selected columns value is between those two values.
3. To work with the entire dataset, uncheck the Specify subset checkbox or just leave it at the default values (default lower / upper is the min / max value in the selected column).
4. Finally, click Load to load the selected range (or the entire file) into the program. The loaded data is shown in the window to the right, along with some summary statistics.

Go to the Plots tab to start working with the data.