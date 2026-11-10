"""Bounded symbolic checks for explicit affine claims; unsupported syntax is unchecked."""

import ast
import re
from typing import Any


def affine_expression(text: str) -> Any:
    import sympy as sp

    if len(text) > 256:
        raise ValueError("Expression too large")
    value = text.replace("−", "-")
    value = re.sub(r"\\(?:left|right|displaystyle)\b", "", value)
    value = re.sub(r"\\(?:cdot|times)\b", "*", value)
    value = re.sub(r"\\frac\s*(\d)\s*(\d)", r"(\1)/(\2)", value)
    for _ in range(8):
        before = value
        value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", value)
        if value == before:
            break
    value = value.replace("{", "(").replace("}", ")").replace("^", "**")
    value = re.sub(r"(\d|\))\s*(?=[xy(])", r"\1*", value)
    if not re.fullmatch(r"[0-9xy+*/().\s-]+", value):
        raise ValueError("Unsupported expression syntax")
    try:
        tree = ast.parse(value.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Malformed expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression too complex")

    def build(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 32:
                raise ValueError("Numeric literal too large")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in {"x", "y"}:
            return sp.Symbol(node.id, real=True)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a = build(node.operand)
            return -a if isinstance(node.op, ast.USub) else a
        if isinstance(node, ast.BinOp):
            a, b = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow) and b.is_Integer and abs(b) <= 2:
                return a**b
        raise ValueError("Unsupported expression node")

    return build(tree.body)


def affine_coefficients(equation: str) -> tuple[Any, Any, Any]:
    import sympy as sp

    if equation.count("=") != 1:
        raise ValueError("Expected one equality")
    left, right = equation.split("=")
    x, y = sp.symbols("x y", real=True)
    expression = sp.cancel(affine_expression(left) - affine_expression(right))
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Nonfinite expression")
    try:
        polynomial = sp.Poly(expression, x, y)
    except sp.PolynomialError as exc:
        raise ValueError("Not affine") from exc
    if polynomial.total_degree() != 1:
        raise ValueError("Not affine")
    return polynomial.coeff_monomial(x), polynomial.coeff_monomial(y), polynomial.coeff_monomial(1)


def affine_findings(text: str) -> list[tuple[str, str]]:
    result = []
    if re.search(r"(?i)\b(?:slope|intercept|rearrang|equivalent)\b", text):
        pairs = re.finditer(
            r"\$([^$\n]{3,180})\$\s*,?\s+(?:or|equivalently)\s+\$([^$\n]{3,180})\$", text
        )
        for match in pairs:
            context = text[max(0, match.start() - 200) : match.start()]
            discussion = text[max(0, match.start() - 200) : match.end() + 200]
            if re.search(r"(?i)\b(?:incorrect|mistake|wrong|erroneous|correction)\b", discussion):
                continue
            if not re.search(
                r"(?i)(?:to get|we get|we obtain|rewrite|rearrang|equivalent)", context
            ):
                continue
            try:
                a = affine_coefficients(match[1])
                b = affine_coefficients(match[2])
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            # Equations of the same line have proportional coefficient vectors.
            if any(a[i] * b[j] != a[j] * b[i] for i in range(3) for j in range(i + 1, 3)):
                result.append(("inequivalent_affine_rewrite", match[0]))
    heading = text.split("\n", 1)[0]
    point = re.search(r"(?i)passes through\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", heading)
    if point is not None and re.search(r"(?i)\bline\b", heading):
        body = text[len(heading) :]
        equations = list(re.finditer(r"\$([^$\n]{3,180})\$", body))
        # With several equations, a premise or intermediate step is not an
        # answer. Check only one explicitly introduced as the answer.
        answer = (
            equations[0]
            if len(equations) == 1
            else next(
                (
                    equation
                    for equation in equations
                    if re.search(
                        r"(?i)\b(?:answer|solution)\s*(?:is|:)\s*$",
                        body[max(0, equation.start() - 40) : equation.start()],
                    )
                ),
                None,
            )
        )
        if answer is not None:
            try:
                a, b, c = affine_coefficients(answer[1])
                if a * int(point[1]) + b * int(point[2]) + c != 0:
                    if re.search(r"(?i)\b(?:incorrect|wrong|correction)\b", text[len(heading) :]):
                        return result
                    result.append(("line_answer_misses_given_point", point[0] + "; " + answer[0]))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
    return result
