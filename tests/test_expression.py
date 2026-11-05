from tests._base import TestBase


class TestExpression(TestBase):

    def test_expression(self):
        from pixelpipes.compiler import Compiler
        from pixelpipes.expression import Expression
        from pixelpipes.graph import Constant, Graph, Output

        with Graph() as graph:
            n1 = Constant(value=5)
            n2 = Constant(value=15)
            n3 = Expression(
                source="((x ^ 2 - y) * 2) / 5 + 2",
                variables=dict(x=n1, y=n2),
            )
            Output(n3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 6)
