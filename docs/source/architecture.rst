Architecture and concepts
=========================

The pipeline is first described as a computational DAG in Python. 
It is then transformed into a sequenctial pipeline that is executed in C++, avoiding GIL and enabling the reusability of the generated stream in other languages.

The framework is divided into two parts:
  * a C++ core and modules, containing all low-level operations,
  * a Python wrapper that provides a high-level way of describing the computation graph and a compiler that transforms the graph into a pipeline.




Macros
------

A macro is a combination of operations that are frequently used together. It is written in Python and combined normal Python language together with DAG generation. 
Macros can base generation on input type inferrence. Macros can also use other macros within them. During compilation all macros are expanded to their inner operations
until only the basic operations remain.


Compilation
-----------


