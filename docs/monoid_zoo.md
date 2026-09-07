# Monoid zoo

Every operator SWEEP scans over, with the proof that it is actually a monoid.
An operator earns its place here only once its associativity proof is written
down and `tests/monoid_laws.cu` checks identity and associativity on host and
device against the `exact` flag it declares.

Ordering contract, everywhere in this document and in the code:
`combine(earlier, later)` — the first argument is earlier in the array. For
transformation monoids that means composition later∘earlier, i.e. the matrix
product **B·A**, not A·B.

## AddU64

### Definition

### Associativity proof

### Identity

## MatModP

### Definition

### Associativity proof

### Identity

## Segmented&lt;M&gt;

### Definition

### Associativity proof

### Identity

## AffineF32

### Definition

### Associativity proof

### Identity

## MobiusF32

### Definition

### Associativity proof

### Identity

## FsmCsv

### Definition

### Associativity proof

### Identity

## MaxPlus&lt;K&gt;

### Definition

### Associativity proof

### Identity
