Building your first graph
=========================

A graph can be constructed by leveraging Python concepts like context and operator overloading (it can also be created manually by adding every node to a graph, but 
this is a much less readable approach). Lets start by extending the initial example of sampling numbers from a list. We can insert a list of lists, sample one of them
then sample two elements from it and sum them together.




.. note::
   Be careful when using graph contexts, use only one context in normal situations. The context mechanism supports also hierarchical contexts however, nodes will only
   be added to the most recent context opened.

Utility decorators wrap even more boilerplate for frequently used cases, the example above can be slightly modified so that entire function is wrapped in a graph context.



To compile the pipeline use

