# How rolling rate is calculated

## Input parameters

**Grouping column:** The column holding the indices of the detected peaks (needs to be a whole number / integer).

**Temperature column:** The column holding the corresponding temperature value for each peak.

**New window every:** The number of seconds between each new window.

**Window size:** The number of seconds each window covers.

**Start at:** The start time of the first window. Setting to 0 will start at the beginning of the signal.

**Sampling Rate:** The sampling rate at which the original signal was recorded. Needed to convert the seconds into indices.

**Start by:** `"window"` will set the start time at 0. `"datapoint"` will start at the first detected peak.

**Label:** `"left"` will use the lower boundary as the label of the window, `"right"` will use the upper boundary as the label of the window, `"datapoint"` will use the first detected peak in the window as the label.


## Example

We have a signal sampled at 400 Hz. The inputs dataframe looks like this:

|peak_index|temperature|
|---|---|
|64|16.200|
|361|16.100|
|652|16.100|
|946|16.000|
|1 231|15.900|
|1 525|16.000|
|1 817|16.000|
|2 105|16.100|
|2 382|16.200|
|2 668|15.900|
|2 946|15.900|
|...|...|


The algorithm now does the following:

1. Set the start of the first window to 0 or the first detected peak.
2. Set the end of the first window to the include all peak indices that are below window size, using the sampling rate to convert seconds into indices. (So 60 seconds @ 400 Hz = 24000 indices, the first window will start at index 0 and end at 24000, even if those values aren't actually in the column.)
3. Count the amount of rows in the window, the result is the amount of detected peaks over the course of one minute.
4. Repeat this by setting the next window to the value given by the `new_window_every` parameter.
5. The result will be an aggregated dataframe where each row represents the amount of peaks over the course of one minute, along with a mean temperature value.
6. Since the last couple of windows won't actually be full minutes, the current approach is to just remove them from the result. Optionally the values for those rows could be approximated by multiplying them with the corresponding value, i.e. the value of the 5th-last row would be multiplied by 1.2 to get to 60 seconds, the 4th row would be multiplied by 1.5, the 3rd by 2, etc, but this is not implemented yet.
7. After removing the last couple of rows, the data is then grouped again by the temperature values (rounded to 1 decimal place), and for each .1 degree temperature, the mean, std, min, max and median values of the amount of peaks are calculated.
8. The result of this is a dataframe with unique temperature values and the corresponding mean, std, min, max and median values of the amount of peaks over the course of one minute.
