# Case 1
```
(playground) yisheng@rpi5:/media/hdddisk/playground/valgrind/lru-dict$ valgrind --leak-check=full --track-origins=yes --show-reachable=yes python3 case1.py
==442956== Memcheck, a memory error detector
==442956== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.
==442956== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
==442956== Command: python3 case1.py
==442956== 
==442956== 
==442956== HEAP SUMMARY:
==442956==     in use at exit: 398,449 bytes in 13 blocks
==442956==   total heap usage: 3,294 allocs, 3,281 frees, 1,772,302 bytes allocated
==442956== 
==442956== 39 bytes in 1 blocks are still reachable in loss record 1 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x401C2C7: malloc (rtld-malloc.h:56)
==442956==    by 0x401C2C7: strdup (strdup.c:42)
==442956==    by 0x401209F: _dl_load_cache_lookup (dl-cache.c:515)
==442956==    by 0x4007FCB: _dl_map_object (dl-load.c:2135)
==442956==    by 0x400259B: openaux (dl-deps.c:64)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x4002AF3: _dl_map_object_deps (dl-deps.c:232)
==442956==    by 0x400B5AF: dl_open_worker_begin (dl-open.c:638)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956== 
==442956== 39 bytes in 1 blocks are still reachable in loss record 2 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x400AA13: malloc (rtld-malloc.h:56)
==442956==    by 0x400AA13: _dl_new_object (dl-object.c:199)
==442956==    by 0x4006957: _dl_map_object_from_fd (dl-load.c:1053)
==442956==    by 0x4007E2F: _dl_map_object (dl-load.c:2268)
==442956==    by 0x400259B: openaux (dl-deps.c:64)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x4002AF3: _dl_map_object_deps (dl-deps.c:232)
==442956==    by 0x400B5AF: dl_open_worker_begin (dl-open.c:638)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956== 
==442956== 104 bytes in 1 blocks are still reachable in loss record 3 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x401C2C7: malloc (rtld-malloc.h:56)
==442956==    by 0x401C2C7: strdup (strdup.c:42)
==442956==    by 0x4007DB3: _dl_map_object (dl-load.c:2201)
==442956==    by 0x400B55F: dl_open_worker_begin (dl-open.c:578)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956==    by 0x4A81593: dlopen_doit (dlopen.c:56)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400145B: _dl_catch_error (dl-catch.c:256)
==442956==    by 0x4A81013: _dlerror_run (dlerror.c:138)
==442956== 
==442956== 104 bytes in 1 blocks are still reachable in loss record 4 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x400AA13: malloc (rtld-malloc.h:56)
==442956==    by 0x400AA13: _dl_new_object (dl-object.c:199)
==442956==    by 0x4006957: _dl_map_object_from_fd (dl-load.c:1053)
==442956==    by 0x4007E2F: _dl_map_object (dl-load.c:2268)
==442956==    by 0x400B55F: dl_open_worker_begin (dl-open.c:578)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956==    by 0x4A81593: dlopen_doit (dlopen.c:56)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400145B: _dl_catch_error (dl-catch.c:256)
==442956== 
==442956== 264 bytes in 2 blocks are still reachable in loss record 5 of 12
==442956==    at 0x488C0AC: calloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x401153F: calloc (rtld-malloc.h:44)
==442956==    by 0x401153F: _dl_check_map_versions (dl-version.c:280)
==442956==    by 0x400B5E7: dl_open_worker_begin (dl-open.c:646)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956==    by 0x4A81593: dlopen_doit (dlopen.c:56)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400145B: _dl_catch_error (dl-catch.c:256)
==442956==    by 0x4A81013: _dlerror_run (dlerror.c:138)
==442956==    by 0x4A8166F: dlopen_implementation (dlopen.c:71)
==442956==    by 0x4A8166F: dlopen@@GLIBC_2.34 (dlopen.c:81)
==442956== 
==442956== 531 bytes in 1 blocks are still reachable in loss record 6 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x50A0DF: PyObject_Malloc (in /usr/bin/python3.12)
==442956==    by 0x52D0C7: ??? (in /usr/bin/python3.12)
==442956==    by 0x519057: ??? (in /usr/bin/python3.12)
==442956==    by 0x579293B: moduleinit (_lru.c:779)
==442956==    by 0x579293B: PyInit__lru (_lru.c:808)
==442956==    by 0x6708A3: ??? (in /usr/bin/python3.12)
==442956==    by 0x66FCEB: ??? (in /usr/bin/python3.12)
==442956==    by 0x503B9F: ??? (in /usr/bin/python3.12)
==442956==    by 0x567877: _PyEval_EvalFrameDefault (in /usr/bin/python3.12)
==442956==    by 0x4C3B83: ??? (in /usr/bin/python3.12)
==442956==    by 0x4C5767: PyObject_CallMethodObjArgs (in /usr/bin/python3.12)
==442956==    by 0x58E58F: PyImport_ImportModuleLevelObject (in /usr/bin/python3.12)
==442956== 
==442956== 768 bytes in 1 blocks are still reachable in loss record 7 of 12
==442956==    at 0x488C2D8: realloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x50A60B: ??? (in /usr/bin/python3.12)
==442956==    by 0x50A0BB: PyObject_Malloc (in /usr/bin/python3.12)
==442956==    by 0x5B2483: _PyObject_GC_New (in /usr/bin/python3.12)
==442956==    by 0x4F1BDF: PyDict_New (in /usr/bin/python3.12)
==442956==    by 0x62A0E3: ??? (in /usr/bin/python3.12)
==442956==    by 0x67A357: ??? (in /usr/bin/python3.12)
==442956==    by 0x679A63: ??? (in /usr/bin/python3.12)
==442956==    by 0x6797DB: Py_InitializeFromConfig (in /usr/bin/python3.12)
==442956==    by 0x68B493: ??? (in /usr/bin/python3.12)
==442956==    by 0x68B3C7: ??? (in /usr/bin/python3.12)
==442956==    by 0x68B3A7: Py_BytesMain (in /usr/bin/python3.12)
==442956== 
==442956== 768 bytes in 1 blocks are still reachable in loss record 8 of 12
==442956==    at 0x4885250: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x50A0DF: PyObject_Malloc (in /usr/bin/python3.12)
==442956==    by 0x4F34EF: ??? (in /usr/bin/python3.12)
==442956==    by 0x4F402F: PyDict_SetDefault (in /usr/bin/python3.12)
==442956==    by 0x5191E7: ??? (in /usr/bin/python3.12)
==442956==    by 0x579293B: moduleinit (_lru.c:779)
==442956==    by 0x579293B: PyInit__lru (_lru.c:808)
==442956==    by 0x6708A3: ??? (in /usr/bin/python3.12)
==442956==    by 0x66FCEB: ??? (in /usr/bin/python3.12)
==442956==    by 0x503B9F: ??? (in /usr/bin/python3.12)
==442956==    by 0x567877: _PyEval_EvalFrameDefault (in /usr/bin/python3.12)
==442956==    by 0x4C3B83: ??? (in /usr/bin/python3.12)
==442956==    by 0x4C5767: PyObject_CallMethodObjArgs (in /usr/bin/python3.12)
==442956== 
==442956== 1,264 bytes in 1 blocks are still reachable in loss record 9 of 12
==442956==    at 0x488C0AC: calloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x400A7B7: calloc (rtld-malloc.h:44)
==442956==    by 0x400A7B7: _dl_new_object (dl-object.c:92)
==442956==    by 0x4006957: _dl_map_object_from_fd (dl-load.c:1053)
==442956==    by 0x4007E2F: _dl_map_object (dl-load.c:2268)
==442956==    by 0x400259B: openaux (dl-deps.c:64)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x4002AF3: _dl_map_object_deps (dl-deps.c:232)
==442956==    by 0x400B5AF: dl_open_worker_begin (dl-open.c:638)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956== 
==442956== 1,352 bytes in 1 blocks are still reachable in loss record 10 of 12
==442956==    at 0x488C0AC: calloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x400A7B7: calloc (rtld-malloc.h:44)
==442956==    by 0x400A7B7: _dl_new_object (dl-object.c:92)
==442956==    by 0x4006957: _dl_map_object_from_fd (dl-load.c:1053)
==442956==    by 0x4007E2F: _dl_map_object (dl-load.c:2268)
==442956==    by 0x400B55F: dl_open_worker_begin (dl-open.c:578)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400ACB3: dl_open_worker (dl-open.c:803)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400B11B: _dl_open (dl-open.c:905)
==442956==    by 0x4A81593: dlopen_doit (dlopen.c:56)
==442956==    by 0x400133B: _dl_catch_exception (dl-catch.c:237)
==442956==    by 0x400145B: _dl_catch_error (dl-catch.c:256)
==442956== 
==442956== 131,072 bytes in 1 blocks are still reachable in loss record 11 of 12
==442956==    at 0x488C0AC: calloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x621EEF: ??? (in /usr/bin/python3.12)
==442956==    by 0x50A51B: ??? (in /usr/bin/python3.12)
==442956==    by 0x50A0BB: PyObject_Malloc (in /usr/bin/python3.12)
==442956==    by 0x5B2483: _PyObject_GC_New (in /usr/bin/python3.12)
==442956==    by 0x4F1BDF: PyDict_New (in /usr/bin/python3.12)
==442956==    by 0x62A0E3: ??? (in /usr/bin/python3.12)
==442956==    by 0x67A357: ??? (in /usr/bin/python3.12)
==442956==    by 0x679A63: ??? (in /usr/bin/python3.12)
==442956==    by 0x6797DB: Py_InitializeFromConfig (in /usr/bin/python3.12)
==442956==    by 0x68B493: ??? (in /usr/bin/python3.12)
==442956==    by 0x68B3C7: ??? (in /usr/bin/python3.12)
==442956== 
==442956== 262,144 bytes in 1 blocks are still reachable in loss record 12 of 12
==442956==    at 0x488C0AC: calloc (in /usr/libexec/valgrind/vgpreload_memcheck-arm64-linux.so)
==442956==    by 0x621EB3: ??? (in /usr/bin/python3.12)
==442956==    by 0x50A51B: ??? (in /usr/bin/python3.12)
==442956==    by 0x50A0BB: PyObject_Malloc (in /usr/bin/python3.12)
==442956==    by 0x5B2483: _PyObject_GC_New (in /usr/bin/python3.12)
==442956==    by 0x4F1BDF: PyDict_New (in /usr/bin/python3.12)
==442956==    by 0x62A0E3: ??? (in /usr/bin/python3.12)
==442956==    by 0x67A357: ??? (in /usr/bin/python3.12)
==442956==    by 0x679A63: ??? (in /usr/bin/python3.12)
==442956==    by 0x6797DB: Py_InitializeFromConfig (in /usr/bin/python3.12)
==442956==    by 0x68B493: ??? (in /usr/bin/python3.12)
==442956==    by 0x68B3C7: ??? (in /usr/bin/python3.12)
==442956== 
==442956== LEAK SUMMARY:
==442956==    definitely lost: 0 bytes in 0 blocks
==442956==    indirectly lost: 0 bytes in 0 blocks
==442956==      possibly lost: 0 bytes in 0 blocks
==442956==    still reachable: 398,449 bytes in 13 blocks
==442956==         suppressed: 0 bytes in 0 blocks
==442956== 
==442956== For lists of detected and suppressed errors, rerun with: -s
==442956== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```