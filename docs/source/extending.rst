Extending
=========

PixelPipes contains a lot of operations used in data loading and augmentation in computer vision. Still, sometimes additional functionality is needed. Simple cases can be easily implemented by Writing
new macros, more complex cases require writing new C++ operations wrapped in a new custom module.   

Writing macros
--------------

A macro is a combination of operations that are frequently used together. It is written in Python and combined normal Python language together with DAG generation. 
Macros can change generated subgraph based on input type inferrence. Macros can also use other macros within them. During compilation all macros are reduced down to primitive operations. 
For this example lets write a macro that





Creating custom operations
--------------------------

Frequently used or complex operations can be included into the pipeline by crating and building a PixelPipes module. 
A module is a dynamic library written in C++ that contains operations. Operations are functions that are exposed in a special manner and can be integrated in operation pipeline.


