from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    a = [v - epsilon / 2 if i == arg else v for i, v in enumerate(vals)]
    b = [v + epsilon / 2 if i == arg else v for i, v in enumerate(vals)]
    grad = (f(*b) - f(*a)) / epsilon
    return grad


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    permanent_mark = set()
    temporary_mark = set()
    sorted_nodes = []

    def visit(node: Variable):
        if node.unique_id in permanent_mark:
            return
        if node.unique_id in temporary_mark:
            raise ValueError("Not a DAG")

        temporary_mark.add(node.unique_id)

        for parent in node.parents:
            visit(parent)

        temporary_mark.remove(node.unique_id)
        permanent_mark.add(node.unique_id)
        sorted_nodes.append(node)

    visit(variable)

    return sorted_nodes[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_nodes = topological_sort(variable)
    uid2deriv = {node.unique_id: 0.0 for node in sorted_nodes}
    uid2deriv[variable.unique_id] = deriv

    for node in sorted_nodes:
        if node.is_leaf():
            node.accumulate_derivative(uid2deriv[node.unique_id])
        else:
            results = node.chain_rule(uid2deriv[node.unique_id])
            for input, deriv in results:
                uid2deriv[input.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
