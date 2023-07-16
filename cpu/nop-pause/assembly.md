### NOP
No Operation, does nothing during a cpu circle  
Used to introduce a delay  
### PAUSE
PAUSE instruction is used to provide a hint to the processor that the code being executed is part of a spin-wait loop  
A spin-wait loop is a loop in which a thread repeatedly checks a condition until it becomes true. In a typical spin-wait loop, the thread repeatedly reads a shared variable, waiting for another thread to update the variable. If the shared variable is updated, the thread can then proceed.  
The PAUSE instruction is designed to improve the performance of spin-wait loops by indicating to the processor that the loop is a spin-wait loop. When the processor encounters a PAUSE instruction, it may choose to reduce power consumption or reduce the number of instructions executed in order to improve performance and reduce power consumption.  