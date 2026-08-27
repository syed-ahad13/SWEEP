# SWEEP
Many loops look impossible to parallelize because step t needs the result of step t−1 (a running total, a running state, a filter, a parser). Escape is the classical parallel-prefix (scan) reformulation: if each step can be written as a small combinable object whose combine rule is associative, the loop's answers can all be computed in parallel.
