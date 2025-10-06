#!/usr/bin/env python
"""
LibCST-based variable renamer for Python files.
Default: dry-run. Use --apply to write changes.

Scope-safe strategy:
- Rename function parameters and local variables within each function/method scope.
- Skip renaming attributes, import names, class names, and module-level globals to avoid cross-file breakage.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Set, Tuple, List

import libcst as cst
from libcst import RemovalSentinel
from libcst.metadata import ParentNodeProvider


DICTIONARY = [
    "alias", "token", "entry", "record", "datum", "value", "item", "node", "unit", "ref",
    "handle", "slot", "field", "flag", "marker", "facet", "aspect", "figure", "gauge", "meter",
]


def generate_alternate_name(original: str) -> str:
    base = random.choice(DICTIONARY)
    # Detect styles
    is_upper_snake = bool(re.fullmatch(r"[A-Z0-9]+(_[A-Z0-9]+)+", original))
    is_lower_snake = bool(re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)+", original))
    is_pascal = bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*", original))
    is_camel = bool(re.fullmatch(r"[a-z][A-Za-z0-9]*", original)) and any(c.isupper() for c in original)

    def to_pascal(s: str) -> str:
        return re.sub(r"(^|[_-])(\w)", lambda m: m.group(2).upper(), s)

    def to_camel(s: str) -> str:
        p = to_pascal(s)
        return p[0].lower() + p[1:] if p else p

    def to_upper_snake(s: str) -> str:
        return re.sub(r"[-\s]", "_", re.sub(r"([a-z])([A-Z])", r"\1_\2", s)).upper()

    def to_lower_snake(s: str) -> str:
        return re.sub(r"[-\s]", "_", re.sub(r"([a-z])([A-Z])", r"\1_\2", s)).lower()

    if is_pascal:
        return to_pascal(base)
    if is_camel:
        return to_camel(base)
    if is_upper_snake:
        return to_upper_snake(base)
    if is_lower_snake:
        return to_lower_snake(base)
    return base


class FunctionLocalRenamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.function_stack: List[Dict[str, str]] = []
        self.class_stack: List[Dict[str, str]] = []  # class-level attribute mapping
        self.module_map: Dict[str, str] = {}  # module-level variable mapping
        self.used_names: Set[str] = set()
        self.changes: List[Tuple[str, str, int]] = []  # (from, to, line)

    # --- Helpers ---
    def _reserve(self, name: str) -> None:
        self.used_names.add(name)

    def _unique(self, base: str) -> str:
        candidate = base
        i = 1
        while candidate in self.used_names:
            i += 1
            candidate = f"{base}_{i}"
        self._reserve(candidate)
        return candidate

    # --- Function scope handling ---
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Start a new scope mapping
        self.function_stack.append({})
        # Reserve existing parameter names to avoid collisions
        for param in node.params.params + node.params.posonly_params + node.params.kwonly_params:
            self._reserve(param.name.value)
        if node.params.vararg:
            self._reserve(node.params.vararg.name.value)
        if node.params.kwarg:
            self._reserve(node.params.kwarg.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Apply parameter renames first based on mapping
        mapping = self.function_stack.pop()

        def rename_param(p: cst.Param) -> cst.Param:
            old = p.name.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, p.name.start.position[0] if p.name.start else -1))
            return p.with_changes(name=p.name.with_changes(value=mapping[old]))

        params = list(updated_node.params.params)
        posonly = list(updated_node.params.posonly_params)
        kwonly = list(updated_node.params.kwonly_params)
        params = [rename_param(p) for p in params]
        posonly = [rename_param(p) for p in posonly]
        kwonly = [rename_param(p) for p in kwonly]

        vararg = updated_node.params.vararg
        if vararg:
            old = vararg.name.value
            new_name = mapping.get(old) or self._unique(generate_alternate_name(old))
            mapping[old] = new_name
            self.changes.append((old, new_name, vararg.name.start.position[0] if vararg.name.start else -1))
            vararg = vararg.with_changes(name=vararg.name.with_changes(value=new_name))

        kwarg = updated_node.params.kwarg
        if kwarg:
            old = kwarg.name.value
            new_name = mapping.get(old) or self._unique(generate_alternate_name(old))
            mapping[old] = new_name
            self.changes.append((old, new_name, kwarg.name.start.position[0] if kwarg.name.start else -1))
            kwarg = kwarg.with_changes(name=kwarg.name.with_changes(value=new_name))

        new_params = updated_node.params.with_changes(
            params=params,
            posonly_params=posonly,
            kwonly_params=kwonly,
            vararg=vararg,
            kwarg=kwarg,
        )
        return updated_node.with_changes(params=new_params)

    # --- Class scope handling ---
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Push a new class attribute mapping
        self.class_stack.append({})

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # Pop class attribute mapping
        if self.class_stack:
            self.class_stack.pop()
        return updated_node

    # Rename local variable targets (simple names) on assignment
    def leave_AssignTarget(self, original_node: cst.AssignTarget, updated_node: cst.AssignTarget) -> cst.AssignTarget:
        # Helper to rename a Name by mapping
        def rename_name(name_node: cst.Name, mapping: Dict[str, str]) -> cst.Name:
            old = name_node.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, name_node.start.position[0] if name_node.start else -1))
            return name_node.with_changes(value=mapping[old])

        # Rename module-level and function-level assignment targets
        target = updated_node.target
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        if isinstance(target, cst.Name):
            return updated_node.with_changes(target=rename_name(target, mapping))
        # Rename tuple/list destructuring simple names
        if isinstance(target, cst.Tuple):
            elts = list(target.elements)
            new_elts = []
            for e in elts:
                v = e.value
                if isinstance(v, cst.Name):
                    v = rename_name(v, mapping)
                new_elts.append(e.with_changes(value=v))
            return updated_node.with_changes(target=target.with_changes(elements=new_elts))
        if isinstance(target, cst.List):
            elts = list(target.elements)
            new_elts = []
            for e in elts:
                v = e.value
                if isinstance(v, cst.Name):
                    v = rename_name(v, mapping)
                new_elts.append(e.with_changes(value=v))
            return updated_node.with_changes(target=target.with_changes(elements=new_elts))
        # Rename self/cls attribute targets within classes
        if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name) and target.attr and isinstance(target.attr, cst.Name):
            base_obj = target.value.value
            if base_obj in {"self", "cls"} and self.class_stack:
                class_map = self.class_stack[-1]
                old_attr = target.attr.value
                if old_attr not in class_map:
                    new_attr = self._unique(generate_alternate_name(old_attr))
                    class_map[old_attr] = new_attr
                    self.changes.append((old_attr, new_attr, target.attr.start.position[0] if target.attr.start else -1))
                new_target = target.with_changes(attr=target.attr.with_changes(value=class_map[old_attr]))
                return updated_node.with_changes(target=new_target)
        return updated_node

    # Rewrite references inside functions based on current function mapping
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        # Decide scope mapping: function or module
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        parent = self.get_metadata(ParentNodeProvider, original_node, None)
        # Skip attributes (obj.name), named arguments, and imports
        if isinstance(parent, cst.Attribute) and parent.attr is original_node:
            return updated_node
        if isinstance(parent, (cst.ImportAlias, cst.ImportFrom, cst.Param)):
            return updated_node
        if isinstance(parent, cst.Arg) and parent.keyword is original_node:
            return updated_node
        new_name = mapping.get(updated_node.value)
        if new_name and new_name != updated_node.value:
            return updated_node.with_changes(value=new_name)
        return updated_node

    # Rename uses of self/cls attribute access based on class mapping; avoid method calls
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.Attribute:
        if not self.class_stack:
            return updated_node
        if not (isinstance(updated_node.value, cst.Name) and isinstance(updated_node.attr, cst.Name)):
            return updated_node
        base_obj = updated_node.value.value
        if base_obj not in {"self", "cls"}:
            return updated_node
        # Skip if this attribute is being called as a method
        parent = self.get_metadata(ParentNodeProvider, original_node, None)
        if isinstance(parent, cst.Call) and parent.func is original_node:
            return updated_node
        class_map = self.class_stack[-1]
        old_attr = updated_node.attr.value
        new_attr = class_map.get(old_attr)
        if new_attr and new_attr != old_attr:
            return updated_node.with_changes(attr=updated_node.attr.with_changes(value=new_attr))
        return updated_node

    # Rename loop targets (for x in y)
    def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        target = updated_node.target
        if isinstance(target, cst.Name):
            old = target.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, target.start.position[0] if target.start else -1))
            target = target.with_changes(value=mapping[old])
            return updated_node.with_changes(target=target)
        return updated_node

    # Rename annotated assignments (x: int = ...)
    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        target = updated_node.target
        if isinstance(target, cst.Name):
            old = target.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, target.start.position[0] if target.start else -1))
            return updated_node.with_changes(target=target.with_changes(value=mapping[old]))
        return updated_node

    # Rename augmented assignments (x += 1)
    def leave_AugAssign(self, original_node: cst.AugAssign, updated_node: cst.AugAssign) -> cst.AugAssign:
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        target = updated_node.target
        if isinstance(target, cst.Name):
            old = target.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, target.start.position[0] if target.start else -1))
            return updated_node.with_changes(target=target.with_changes(value=mapping[old]))
        return updated_node

    # Rename with-as targets (with open() as f)
    def leave_WithItem(self, original_node: cst.WithItem, updated_node: cst.WithItem) -> cst.WithItem:
        if updated_node.asname and isinstance(updated_node.asname.name, cst.Name):
            mapping = self.function_stack[-1] if self.function_stack else self.module_map
            name_node = updated_node.asname.name
            old = name_node.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, name_node.start.position[0] if name_node.start else -1))
            new_as = updated_node.asname.with_changes(name=name_node.with_changes(value=mapping[old]))
            return updated_node.with_changes(asname=new_as)
        return updated_node

    # Rename comprehension targets ([x for x in ...])
    def leave_CompFor(self, original_node: cst.CompFor, updated_node: cst.CompFor) -> cst.CompFor:
        mapping = self.function_stack[-1] if self.function_stack else self.module_map
        target = updated_node.target
        if isinstance(target, cst.Name):
            old = target.value
            if old not in mapping:
                new_name = self._unique(generate_alternate_name(old))
                mapping[old] = new_name
                self.changes.append((old, new_name, target.start.position[0] if target.start else -1))
            return updated_node.with_changes(target=target.with_changes(value=mapping[old]))
        return updated_node


def process_file(py_path: Path, apply: bool) -> Dict[str, Dict[str, int]]:
    src = py_path.read_text(encoding="utf-8")
    try:
        mod = cst.parse_module(src)
    except Exception:
        return {}
    wrapper = cst.MetadataWrapper(mod)
    transformer = FunctionLocalRenamer()
    new_mod = wrapper.visit(transformer)
    summary: Dict[str, int] = {}
    for old, new, _ in transformer.changes:
        key = f"{old}=>{new}"
        summary[key] = summary.get(key, 0) + 1
    if apply and new_mod.code != src:
        py_path.write_text(new_mod.code, encoding="utf-8")
    return {str(py_path): summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    py_dir = root / "python_code"
    results: Dict[str, Dict[str, int]] = {}

    for path in py_dir.rglob("*.py"):
        if path.name == os.path.basename(__file__):
            continue
        res = process_file(path, args.apply)
        results.update(res)

    print(json.dumps({"apply": args.apply, "files": results}, indent=2))


if __name__ == "__main__":
    main()


