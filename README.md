## Do you like PyTorch? Do you like karpathy/micrograd? Do you like tinygrad?
Well, quantumgrad is none of those. Hell I don't even know what this is.

It's a deep learning framework based on CUDA made from ground up for the sole purpose of me developing a detailed understanding of every part of parallelized deep learning on a low level.

Starting from as low as efficient matrix multiplies in CUDA all the way to training LLMs (hopefully).

(It has nothing to do with quantum computation, yes it's misleading).

## How it works
There are CUDA files which get compiled to a shared library and there are Python wrappers that let you call those compiled CUDA functions from nice old convenient Python.
