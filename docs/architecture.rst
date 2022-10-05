Architecture and concepts
=========================

This document describes the concepts of the PixelPipes framework in extensive technical details. If you are more interested in practical examples, check out the collection of `tutorials <tutorials/index.html>`_.

Similar to some machine learning frameworks, most notably TensorFlow, a PixelPipes stream is formalized as a computational directed acyclic graph (DAG). This graph is constructed in Python.
The graph is then transformed into a sequence of operations that are executed in C++. These operations can be very primitive, e.g. summing up two numbers, or they can be very
specialized, e.g. a specific image operation. Most users will only work with Python frontend to describe a stream, but it can be beneficial to know what lies beneath.

The framework is therefore divided into two parts with a binding API bridge:

  * a C++ core containing all low-level operations together with some binding code,
  * a Python frontend that provides a high-level way of describing a computational graph.

TODO: image overview

Python
------

As already said, the Python part of the framework is just a frontent that makes assembling a pipeline easy. It is not involved in the execution in any way and once a pipeline is
assembled, it can be executed from C++ directly (or from any other language). The frontend makes graph description easier by introducing organizational concepts
like macros and resources and by allowing the user to leverage abundance of Python tools to import data and transform it into pipeline primitives.

Below is a list of Python-level concepts that are important to know when you start using the framework:

 * Node: represents an high-level operaton with zero or more inputs that produces a single output
 * Graph: a collection of connected nodes, a graph describes dependencies between operations, each operation, represented as a node in a graph can accept zero or more inputs and produces a single output. There are two types of nodes, operations and macros.
 * Operation: special nodes that map directly to individual operations in the final pipeline.
 * Macro: A macro is a combination of operations that are frequently used together and are represented as a single node. During compilation macros are expanded to their inner subgraph
   until only the basic operation nodes remain. Macros are written in Python and combined normal Python language together with DAG generation, they can therefore base generation 
   on input types, use loops and conditional statements.
 * Constant: Special type of operation node, used to introduce constant values into the pipeline. Constats can be scalars, but also tensors and lists.
 * Compilation: a process of iterativelly reducing macro nodes in a graph to basic operations, removing duplicates where possible and ordering operations according to dependencies.
 * Resource: Special type of macros that handle multi-field type abstractions, can be used to make the graph more organized, but get dissolved during compilation.

C++
---

The native part of the framework is written in C++, this part is less accessible when only using the framework, but it is important for  

 * Operation: A stateless algorithm that accepts zero or more inputs and produces a single output. Direct mapping of the operation node concept.
 * Token: Tokens are data units passed from operation to operation. 
 * Type: Describes type trait of a token.
 * Pipeline: A sequence of operations.
 * Module: Operations are organized into modules. Some modules are a part of the core framework and the separation is less noticeable. Other modules can be written as extensions to the framework, providing additional operations.




