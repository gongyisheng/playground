// In C++, any name that is not defined inside a class, function, 
//or a namespace is considered to be part of an implicitly defined 
//namespace called the global namespace (sometimes also called the global scope).

// When you use an identifier that is defined inside a namespace (such as the std namespace), 
// you have to tell the compiler that the identifier lives inside the namespace.

// Best practice: Use explicit namespace prefixes to access identifiers defined in a namespace.

// A using directive allows us to access the names in a namespace without using a namespace prefix.
// Best practice: Avoid using directives in header files. Avoid confilcts with libraries.