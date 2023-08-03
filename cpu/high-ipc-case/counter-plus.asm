section .text
  global _start  ; must be declared for linker (ld)
_start:
  mov eax, 1     ; initialize i to 1
  .loop:         ; tells linker entry point
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    add eax, 1   ; add 1 to eax
    jmp .loop    ; loop back to .loop
  .exit:
    ret
