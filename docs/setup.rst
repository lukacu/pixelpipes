Installation and quick-start
============================

The source-code for the PixelPipes framwork can be downloaded from Github, but the most convenient way of testing it is by installing a prebuilt version. 
The package can be installed as a Python wheel package, currently from a testing PyPi compatible repository located `here <https://data.vicos.si/lukacu/pypi/>`_.

```
> pip install pixelpipes -i https://data.vicos.si/lukacu/pypi/
```

Simple example
--------------

To demonstrate a very simple example of using a PixelPipes pipeline (without any image operations), lets sample random numbers from a pre-defined list and 
display them.

.. literalinclude:: ../../examples/simple.py

Note that this sequence will be the same every time you run the script with the same sample indices. This example is very simple, but shows how a pipleine is built. More 
complex examples with image operations are presented `here <tutorials/index.html>`_.
