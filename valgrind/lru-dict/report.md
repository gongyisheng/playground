# Case 1
```
(playground) yisheng@rpi5:/media/hdddisk/playground/valgrind/lru-dict$ valgrind --leak-check=full --track-origins=yes python3 case1.py
==442658== Memcheck, a memory error detector
==442658== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.
==442658== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
==442658== Command: python3 case1.py
==442658== 
==442658== 
==442658== HEAP SUMMARY:
==442658==     in use at exit: 398,449 bytes in 13 blocks
==442658==   total heap usage: 3,294 allocs, 3,281 frees, 1,772,302 bytes allocated
==442658== 
==442658== LEAK SUMMARY:
==442658==    definitely lost: 0 bytes in 0 blocks
==442658==    indirectly lost: 0 bytes in 0 blocks
==442658==      possibly lost: 0 bytes in 0 blocks
==442658==    still reachable: 398,449 bytes in 13 blocks
==442658==         suppressed: 0 bytes in 0 blocks
==442658== Reachable blocks (those to which a pointer was found) are not shown.
==442658== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==442658== 
==442658== For lists of detected and suppressed errors, rerun with: -s
==442658== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```