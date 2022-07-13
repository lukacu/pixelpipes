Extending PixelPipes
====================

Despite having a C++ core, the  


Writing macros
--------------

A macro is a combination of operations that are frequently used together. It is written in Python and combined normal Python language together with DAG generation. 
Macros can base generation on input type inferrence. Macros can also use other macros within them. During compilation 



Creating custom operations
--------------------------

Frequently used or complex operations can be included into the pipeline by crating and building a PixelPipes module. 
A module is a dynamic library written in C++ that contains operations. Operations are functions that are exposed in a special manner and can be integrated in operation pipeline.


