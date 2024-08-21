# Case 1
```
(playground) yisheng@rpi5:/media/hdddisk/playground/valgrind/lru-dict$ valgrind --leak-check=full --track-origins=yes python3 case1.py
==441580== Memcheck, a memory error detector
==441580== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.
==441580== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
==441580== Command: python3 lrudict_test.py
==441580== 
==441580== 
==441580== HEAP SUMMARY:
==441580==     in use at exit: 398,449 bytes in 13 blocks
==441580==   total heap usage: 3,281 allocs, 3,268 frees, 1,754,652 bytes allocated
==441580== 
==441580== LEAK SUMMARY:
==441580==    definitely lost: 0 bytes in 0 blocks
==441580==    indirectly lost: 0 bytes in 0 blocks
==441580==      possibly lost: 0 bytes in 0 blocks
==441580==    still reachable: 398,449 bytes in 13 blocks
==441580==         suppressed: 0 bytes in 0 blocks
==441580== Reachable blocks (those to which a pointer was found) are not shown.
==441580== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==441580== 
==441580== For lists of detected and suppressed errors, rerun with: -s
==441580== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```