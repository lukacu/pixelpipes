import unittest

from attributee import String, Map

from .graph import Macro, Input, Reference, Graph, Copy
from . import types, numbers

# Based on: https://gist.github.com/elias94/c03a170ae2e208d4a75f7f371a48be33

'''
-> CONTEXT-FREE GRAMMAR <-
expr     --> expr PLUS term      |  expr MINUS term    |  term
term     --> term TIMES factor   |  term DIVIDE factor |  factor
factor   --> exponent POW factor |  exponent
exponent --> MINUS exponent      |  final
final    --> CONSTANT            |  VARIABLE           | ( expr )        
'''

class _Expression(object):

    def expand(self, inputs):
        raise NotImplementedError()

    def __repr__(self):
        klass = self.__class__.__name__
        private = '_{0}__'.format(klass)
        args = []
        for name in self.__dict__:
            if name.startswith(private):
                value = self.__dict__[name]
                name = name[len(private):]
                args.append('{0}={1}'.format(name, repr(value)))
        return '{0}({1})'.format(klass, ', '.join(args))


class _Constant(_Expression):

    def __init__(self, value):
        self._value = value

    def expand(self, inputs):
        return self._value

class _Variable(_Expression):

    def __init__(self, value):
        self._value = value

    def expand(self, inputs):
        if not self._value in inputs:
            raise ValueError("Unknown variable '{}'".format(self._value))
        return inputs[self._value][0]

class _Operation(_Expression):

    def __init__(self, operation, left, right):
        super().__init__()
        self._op = operation
        self._left = left
        self._right = right

    def expand(self, inputs):

        x = self._left.expand(inputs)
        y = self._right.expand(inputs)

        if self._op == '+':
            return numbers.Add(a=x, b=y)
        if self._op == '-':
            return numbers.Subtract(a=x, b=y)
        if self._op == '*':
            return numbers.Multiply(a=x, b=y)
        if self._op == '/':
            return numbers.Divide(a=x, b=y)
        if self._op == '^':
            return numbers.Power(a=x, b=y)
        #if self._op == '%': TODO: also implement this in parser
        #    return numbers.Modulo(a=x, b=y)
        raise Exception('Unknown operator: ' + self._op)

class _Parser(object):

    def __init__(self):
        self._current = None
        self._tokens = None

    def parse(self, tokens):
        self._tokens = tokens
        self._next()
        return self.expr()

    def _next(self):
        if self._tokens:
            # remove first element from token list\
            self._current = self._tokens.pop(0)
        else:
            self._current = None

    def expr(self):
        result = self.term()
        while self._current in ('+', '-'):
            if self._current == '+':
                self._next()
                a = result
                b = self.term()
                result = _Operation('+', a, b)
            if self._current == '-':
                self._next()
                minuend = result
                subtrahend = self.term()
                result = _Operation('-', minuend, subtrahend)
        return result

    def term(self):
        result = self.factor()
        while self._current in ('*', '/'):
            if self._current == '*':
                self._next()
                factor = result
                multiplier = self.term()
                result = _Operation('*', factor, multiplier)
            if self._current == '/':
                self._next()
                dividend = result
                divisor = self.term()
                result = _Operation('/', dividend, divisor)
        return result

    def factor(self):
        result = self.exponent()
        while self._current == '^':
            self._next()
            base = result
            exp = self.factor()
            result = _Operation('^', base, exp)
        return result

    def exponent(self):
        if self._current == '-':
            self._next()
            positive = self.final()
            result = _Operation('-', _Constant(0), positive)
        else:
            result = self.final()
        return result

    def final(self):
        result = None
        if isinstance(self._current, float):
            result = _Constant(self._current)
            self._next()
        elif isinstance(self._current, str) and self._current[0].isalpha():
            result = _Variable(self._current)
            self._next()
        elif self._current == '(':
            self._next()
            result = self.expr()
            if self._current != ')':
                raise Exception('Expected )')
            self._next()
        else:
            raise Exception('Expected number or (expr)')
        return result

_SYMBOLS = ['+', '-', '*', '/', '^', '(', ')']

class Expression(Macro):

    """Numeric expression with variables

    Macro that expands into an arithmetic expression parsed from an input string.

    Inputs:
        - source: text representation of arithmetic expression
        - variables: a map of inputs that are inserted into the expression

    Category: arithmetic, macro
    """

    source = String()
    variables = Map(Input(types.Float()))

    def input_values(self):
        return [self.variables[name] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(k, types.Float()) for k in self.variables.keys()]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            assert k in config["variables"]
            config["variables"][k] = v
        return self.__class__(_origin=_origin, **config)

    def expand(self, **inputs):
        parser = _Parser()
        tokens = self._tokenize(list(self.source))
        if not tokens:
            # not parse if empty
            return None
        tree = parser.parse(tokens)
        return tree.expand(inputs)

    def _tokenize(self, chars):
        tokens = []
        pos = 0
        while pos < len(chars):
            char = chars[pos]

            if char == ' ':
                # do nothing...
                pass
 
            elif char in _SYMBOLS:
                tokens.append(char)
            elif char.isdigit():
                start = pos
                num = float(char)
                while pos + 1 < len(chars) and chars[pos + 1] not in _SYMBOLS:
                    try:
                        pos += 1
                        num = float("".join(chars[start:pos]))
                    except ValueError:
                        break
                tokens.append(num)
                
            elif char.isalpha():
                variable = char
                while pos + 1 < len(chars) and (chars[pos + 1].isalnum() or chars[pos + 1] in ["_"]):
                    pos += 1
                    variable += chars[pos]
                tokens.append(variable)
            else:
                raise Exception("Unexpected symbol at position {}: '{}'".format(pos, chars[pos]))

            pos += 1
        return tokens

from pixelpipes.tests import TestBase

class Tests(TestBase):    

    def test_expression(self):

        from .compiler import Compiler
        from .graph import Graph, Constant, Output

        with Graph() as graph:
            n1 = Constant(value=5)
            n2 = Constant(value=15)
            n3 = Expression(source="((x ^ 2 - y) * 2) / 5 + 2", variables=dict(x=n1, y=n2))
            Output(n3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 6)