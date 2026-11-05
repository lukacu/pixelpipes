"""Discover public PixelPipes nodes without importing the package.

The documentation build must work before the native extension is compiled, so
node discovery is deliberately based on Python syntax and imports only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ROOT_PACKAGE = "pixelpipes"
NODE_BASE = "pixelpipes.graph.Node"


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    type_name: str
    default: str | None
    description: str


@dataclass(frozen=True)
class ClassInfo:
    name: str
    bases: tuple[str, ...]
    hidden: bool
    parameters: tuple[ParameterInfo, ...]


def _module_name(root: Path, source: Path) -> str:
    relative = source.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((ROOT_PACKAGE, *parts))


def _package_name(module: str) -> str:
    return module.rpartition(".")[0]


def _relative_module(module: str, imported: ast.ImportFrom) -> str:
    if imported.level == 0:
        return imported.module or ""

    package = _package_name(module).split(".") if _package_name(module) else []
    package = package[: len(package) - imported.level + 1]
    if imported.module:
        package.extend(imported.module.split("."))
    return ".".join(package)


def _expression_name(expression: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        parent = _expression_name(expression.value, aliases)
        return f"{parent}.{expression.attr}"
    return ""


def _decorator_name(decorator: ast.AST) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return ""


def _call_name(expression: ast.AST) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return ""


def _source(expression: ast.AST) -> str:
    return ast.unparse(expression)


def _literal_text(expression: ast.AST | None) -> str | None:
    if expression is None:
        return None
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError):
        return _source(expression)
    if value is None:
        return None
    return repr(value)


def _description_text(expression: ast.AST | None) -> str:
    if expression is None:
        return ""
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError):
        return _source(expression)
    return value if isinstance(value, str) else str(value)


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _type_name(call: ast.Call) -> str:
    factory = _call_name(call.func)

    if factory == "SeedInput":
        return "Integer"

    if factory in {"Input", "EnumerationInput", "Enumeration"} and call.args:
        return _source(call.args[0])

    if factory in {"List", "Map", "InputCollector"} and call.args:
        nested = call.args[0]
        if isinstance(nested, ast.Call):
            nested_type = _call_name(nested.func)
            if nested_type in {"Input", "EnumerationInput", "Enumeration"} and nested.args:
                return f"{factory.lower()}[{_source(nested.args[0])}]"
            return f"{factory.lower()}[{_source(nested.func)}]"
        return f"{factory.lower()}[{_source(nested)}]"

    return factory or _source(call.func)


_ATTRIBUTE_FACTORIES = {
    "Any",
    "Boolean",
    "Callable",
    "Enumeration",
    "EnumerationInput",
    "Float",
    "Input",
    "InputCollector",
    "Integer",
    "IntegerInput",
    "List",
    "Map",
    "Number",
    "Primitive",
    "SeedInput",
    "String",
}


def _parameter_from_assignment(statement: ast.Assign) -> ParameterInfo | None:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        return None
    if not isinstance(statement.value, ast.Call):
        return None

    call = statement.value
    factory = _call_name(call.func)
    if factory not in _ATTRIBUTE_FACTORIES:
        return None

    default = _keyword(call, "default")
    if default is None and factory in {"Input", "EnumerationInput", "Enumeration"}:
        default = call.args[1] if len(call.args) > 1 else None

    description = _keyword(call, "description")
    if description is None and factory == "SeedInput" and call.args:
        description = call.args[0]

    description_text = _description_text(description)
    return ParameterInfo(
        name=statement.targets[0].id,
        type_name=_type_name(call),
        default=_literal_text(default),
        description=description_text,
    )


def _classes_in(source: Path, root: Path) -> dict[str, ClassInfo]:
    module = _module_name(root, source)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    aliases: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            imported_module = _relative_module(module, statement)
            for imported in statement.names:
                if imported.name == "*":
                    continue
                local_name = imported.asname or imported.name
                aliases[local_name] = f"{imported_module}.{imported.name}"
        elif isinstance(statement, ast.Import):
            for imported in statement.names:
                local_name = imported.asname or imported.name.split(".")[0]
                aliases[local_name] = imported.name

    local_classes = {
        statement.name: f"{module}.{statement.name}"
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
    }
    aliases.update(local_classes)

    result: dict[str, ClassInfo] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        full_name = local_classes[statement.name]
        bases = tuple(_expression_name(base, aliases) for base in statement.bases)
        hidden = any(_decorator_name(decorator) == "hidden" for decorator in statement.decorator_list)
        parameters = tuple(
            parameter
            for body_statement in statement.body
            if isinstance(body_statement, ast.Assign)
            for parameter in [_parameter_from_assignment(body_statement)]
            if parameter is not None
        )
        result[full_name] = ClassInfo(full_name, bases, hidden, parameters)
    return result


def discover_classes(root: Path) -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for source in sorted(root.rglob("*.py")):
        classes.update(_classes_in(source, root))
    return classes


def _is_node(name: str, classes: dict[str, ClassInfo], visiting: set[str] | None = None) -> bool:
    if name == NODE_BASE:
        return True
    if name not in classes:
        return False

    visiting = set() if visiting is None else visiting
    if name in visiting:
        return False
    visiting.add(name)
    result = any(_is_node(base, classes, visiting) for base in classes[name].bases)
    visiting.remove(name)
    return result


def _parameters_for(name: str, classes: Mapping[str, ClassInfo], visiting: set[str] | None = None) -> tuple[ParameterInfo, ...]:
    if name not in classes:
        return ()

    visiting = set() if visiting is None else visiting
    if name in visiting:
        return ()
    visiting.add(name)

    parameters: dict[str, ParameterInfo] = {}
    for base in classes[name].bases:
        for parameter in _parameters_for(base, classes, visiting):
            parameters[parameter.name] = parameter
    for parameter in classes[name].parameters:
        parameters[parameter.name] = parameter

    visiting.remove(name)
    return tuple(parameters.values())


def discover_node_info(root: Path) -> dict[str, tuple[ParameterInfo, ...]]:
    classes = discover_classes(root)
    return {
        name: _parameters_for(name, classes)
        for name, info in classes.items()
        if not info.hidden
        and not name.rsplit(".", 1)[-1].startswith("_")
        and _is_node(name, classes)
    }


def discover_nodes(root: Path) -> list[str]:
    return sorted(discover_node_info(root))


def _format_signature(parameters: Iterable[ParameterInfo]) -> str:
    values = []
    for parameter in parameters:
        value = parameter.name
        if parameter.default is not None:
            value += f"={parameter.default}"
        values.append(value)
    return "(" + ", ".join(values) + ")"


def _append_inputs(lines: list[str], parameters: Iterable[ParameterInfo]) -> None:
    parameters = tuple(parameters)
    if not parameters:
        return

    lines.extend(("", "Inputs", "------", ""))
    for parameter in parameters:
        label = f"**{parameter.name}** (``{parameter.type_name}``)"
        if parameter.default is not None:
            label += f", default ``{parameter.default}``"
        lines.extend((label + ":", ""))
        if parameter.description:
            lines.extend((f"    {parameter.description}", ""))


def _node_parameters(app, name: str) -> tuple[ParameterInfo, ...] | None:
    node_info = getattr(app, "_pixelpipes_node_info", {})
    parameters = node_info.get(name)
    if parameters is not None:
        return parameters

    matches = [
        parameters
        for full_name, parameters in node_info.items()
        if full_name.rsplit(".", 1)[-1] == name
    ]
    return matches[0] if len(matches) == 1 else None


def process_node_signature(app, what, name, obj, options, signature, return_annotation):
    if what not in {"class", "apiclass"}:
        return None
    parameters = _node_parameters(app, name)
    if parameters is None:
        return None
    return _format_signature(parameters), return_annotation


def process_node_docstring(app, what, name, obj, options, lines):
    if what not in {"class", "apiclass"}:
        return
    parameters = _node_parameters(app, name)
    if parameters is not None:
        _append_inputs(lines, parameters)


def render_nodes(nodes: Iterable[str]) -> str:
    lines = ["Node classes", "------------", ""]
    current_module = None
    for name in nodes:
        module = name.rpartition(".")[0]
        if module != current_module:
            if current_module is not None:
                lines.append("")
            lines.extend((module, "~" * len(module), ""))
            current_module = module
        lines.extend((
            f".. _node-{name}:",
            "",
            f".. autoapiclass:: {name}",
            "   :members:",
            "   :show-inheritance:",
            "   :no-index:",
            "",
        ))
    return "\n".join(lines)


def write_nodes_page(root: Path, output: Path) -> list[str]:
    nodes = discover_nodes(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_nodes(nodes), encoding="utf-8")
    return nodes


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    output_path = Path(__file__).resolve().parent / "_generated" / "nodes.rst"
    write_nodes_page(repository_root / ROOT_PACKAGE, output_path)
