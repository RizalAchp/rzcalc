# Rust Simple Calculator
Simple calculator written in Rust. Supports many arithmetic functions, boolean logic, and builtin functions.

## Value types

The calculator supports four value types:

* Booleans (`true` and `false`)
* Numbers (examples: `1`, `3.2`, `-200`, `1.3333`, `1e-3`, `0xFFFFF`, `0b010011`, `0o777`) 
    - number support radix 2 (binary), 8 (octal), and 16 (hex)

## Arithmetic Options

Supports the following operations:

* Arithmetic
    * Addition: `a + b`
    * Subtraction: `a - b`
    * Multiplication: `a * b`
    * Division: `a / b`
    * Rem / Module: `a % b`
    * BitAnd: `a & b`
    * BitOr: `a | b`
    * BitXor: `a ^ b`
    * Shr: `a >> b`
    * Shl: `a << b`
* Relational
    * Equal: `a == b`, `a != b`
    * Compare: `a < b`, `a <= b`, `a > b`, `a >= b`
* Logic
    * Conjunction: `a and b`
    * Disjunction: `a or b`
    * Negation: `not a`


## TODO! 

### Predefined constants and functions.

- [] Supports the following constants.
    * `pi`
    * `tau`
    * `e`
    * `nan`
    * `inf`
    * `neginf`

- [] The following common mathematical functions are supported.
    * `sin(x)`, `cos(x)`, `tan(x)`
    * `asin(x)`, `acos(x)`, `atan(x)`
    * `ln(x)`, `log10(x)`, `log2(x)`, `log(x, base)`
    * `round(x)`, `floor(x)`, `ceil(x)`
    * `sqrt(x)`, `exp(x)`, `powf(x, e)`, `pow(x, e)`
    * `abs(x)`, `min_num(x, y)`, `max_num(x, y)`

- [] The following Python-like utility functions are included.
    * `min(...)`: minimum of arguments.
    * `max(...)`: maximum of arguments.
    * `rand()`, `rand(stop)`, `rand(start, stop)`: random float (default range is 0.0 to 1.0).

- [] The following common printing functions to stdout/stderr are supported.
    * `print(value)`: print to stdout
    * `println(value)`: print to stdout with new line at the end
    * `eprint(value)`: print to stderr
    * `eprintln(value)`: print to stderr with new line at the end
    * `debug(expr)`: print debug representation for expression

- [] The following common format number functions.
    * `bin(value, _bool)`: format number as string in binary format (`0b111`)
    * `oct(value, _bool)`: format number as string in octal format (`0o777`)
    * `hex(value, _bool)`: format number as string in hex format (`0xFFF`)

