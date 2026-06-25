"""Bundled organ: calc_eval — safe arithmetic evaluator.

Handles user math like "calculate 1234 * 56", "what is 2 + 2",
"what is the square root of 144", "5 plus 3 times 2". Without this
organ those queries fall through to wikipedia_lookup (which
returned the Wikipedia article on "New Jerusalem" for square-root
questions) or trigger expensive organ synthesis.

Evaluation is AST-only — never eval() — with a whitelist of node
types (BinOp, UnaryOp, Constant) and operators (+ - * / // % **).
No function calls, no names, no attribute access.
"""
import ast
import math
import re

ORGAN_META = {
    "intent":      "calc_eval",
    "description": "evaluate arithmetic expressions safely (no eval)",
    "version":     "1.0",
    "capabilities": [],
    "inputs":  {},
    "outputs": {"value": "float", "expression": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


_ALLOWED_BIN = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.Pow, ast.FloorDiv,
)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)

# Word-form operators users type when they don't reach for symbols.
_WORD_OPS = [
    (re.compile(r"\bplus\b",          re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b",         re.IGNORECASE), "-"),
    (re.compile(r"\btimes\b",         re.IGNORECASE), "*"),
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided\s+by\b",  re.IGNORECASE), "/"),
    (re.compile(r"\bover\b",          re.IGNORECASE), "/"),
    (re.compile(r"\bmodulo\b",        re.IGNORECASE), "%"),
    (re.compile(r"\bmod\b",           re.IGNORECASE), "%"),
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), "**"),
    (re.compile(r"[×]", re.IGNORECASE), "*"),
    (re.compile(r"[÷]", re.IGNORECASE), "/"),
    (re.compile(r"\^",  re.IGNORECASE), "**"),
]

# "square root of N" / "sqrt of N" / "sqrt(N)" → "(N)**0.5"
_SQRT_RE = re.compile(
    r"(?:square\s+root\s+of|sqrt\s+of|sqrt)\s*\(?\s*([0-9.+\-*/() ]+?)\s*\)?(?=$|\?|[^0-9.+\-*/() ])",
    re.IGNORECASE,
)

# "N squared" → "(N)**2", "N cubed" → "(N)**3"
_SQUARED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*squared\b", re.IGNORECASE)
_CUBED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*cubed\b", re.IGNORECASE)

# "X% of Y" or "X percent of Y" → "(X/100)*Y". Stays infix so the AST
# walker only sees plain arithmetic.
_PERCENT_OF_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Strip the conversational lead-in so we get a clean expression.
_LEAD_RE = re.compile(
    r"^\s*(?:calc(?:ulate)?|compute|evaluate|solve|how\s+much\s+is|"
    r"what(?:'s| is)?(?:\s+the)?|tell\s+me)\b\s*[:\s]?",
    re.IGNORECASE,
)


def _extract_expression(message: str) -> str:
    text = message or ""
    text = text.strip().rstrip("?.")
    text = _LEAD_RE.sub("", text).strip()
    # square root → ()**0.5
    text = _SQRT_RE.sub(lambda m: f"(({m.group(1).strip()})**0.5)", text)
    # "N squared" / "N cubed" → power form
    text = _SQUARED_RE.sub(lambda m: f"(({m.group(1)})**2)", text)
    text = _CUBED_RE.sub(lambda m: f"(({m.group(1)})**3)", text)
    # "X% of Y" / "X percent of Y" → (X/100)*Y
    text = _PERCENT_OF_RE.sub(
        lambda m: f"(({m.group(1)}/100)*{m.group(2)})", text,
    )
    # word operators → symbols
    for pat, sym in _WORD_OPS:
        text = pat.sub(sym, text)
    # Strip stray trailing words like "please".
    text = re.sub(r"\b(?:please|equals?|equal to|result)\b", "", text, flags=re.IGNORECASE)
    # Collapse spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN):
            left = walk(node.left)
            right = walk(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                # Reject astronomical exponents so we don't lock up on
                # 2**99999999. Cap is generous but well below DoS.
                if isinstance(right, (int, float)) and abs(right) > 1000:
                    raise ValueError("exponent too large")
                return left ** right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return walk(tree)


def _format(value):
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        # Trim trailing zeros but keep precision.
        return f"{value:.10g}"
    return str(value)


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    expr = _extract_expression(message)
    if not expr or not re.search(r"\d", expr):
        return text_card(
            "I couldn't find an expression to evaluate. Try "
            "'calculate 12 * 7' or 'what is the square root of 144'.",
            "Calculator",
        )
    try:
        value = _safe_eval(expr)
    except ZeroDivisionError:
        return text_card("Division by zero.", "Calculator")
    except Exception as exc:
        return text_card(
            f"I couldn't evaluate that as arithmetic: {exc}. "
            "Supported: + - * / // % ** and square root.",
            "Calculator",
        )

    body = f"{expr} = {_format(value)}"
    card = text_card(body, "Calculator")
    card.card_data.update({"value": value, "expression": expr})
    return card
