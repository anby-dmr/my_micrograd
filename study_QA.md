# Theory
- Q: The necessity of each component in class "Value"?

- Q: Why use the class "Value" as the basic component instead of functions?

- Q: How does the "Value" build the computation graph?
By recording the operation and the inputs (also use the class "Value").

- Q: How does backprop work under the hood?
Treat the current "Value" as the root, and **recursively** traverse (use DFS) the graph to compute the topological order.
Then, compute the gradient of each "Value" in the topological order.

- Q: Engine only supports scalar values and how to extend it to support vector values?

# Coding style
- Use compose instead of inheritance to build "Module", "Layer", and "MLP".
- The use of "**kwargs" in "Layer".

# Mistakes I made
**Engine**
- Did not accumulate the gradient in the backward pass.
- Did not set the gradient of the root node to 1.0 before the backward pass.
- Did not set parameter "self" when define the class "Module".
- Class Layer should return a Value instead of a list [Value] when noutput is 1.