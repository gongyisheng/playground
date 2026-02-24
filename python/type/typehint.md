# Type Hint Notes

## Callable

`Callable[[ArgType1, ArgType2], ReturnType]` annotates functions and other callable objects — the first list is the parameter types, the second is the return type. Use `Callable[..., ReturnType]` when you don't care about the signature.
