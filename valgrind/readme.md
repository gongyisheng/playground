# Install
`sudo apt-get install valgrind`

# Profile
for python programs:`valgrind --leak-check=full --track-origins=yes python3 <your_script.py>`  
- `--leak-check=full`: This option tells `valgrind` to perform a detailed analysis of memory leaks and provide information about each leak discovered.
- `--track-origins=yes`: This option helps in tracking the origins of uninitialized values when they are used, which can help diagnose errors more clearly.

# Report
`definitely lost` indicates memory that your program has allocated and not freed, hence representing a memory leak. 