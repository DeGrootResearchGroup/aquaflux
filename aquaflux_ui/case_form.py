"""A case file as a form: rows built from the case-file schema, and the edits those rows make.

The schema is the description ``aquaflux schema`` prints -- every kind of value a case file may hold,
each field, what it accepts and its default -- read off the same annotations aquaflux checks a file
against. Nothing here lists a kind, a field or a choice of its own: every dropdown a row offers comes
from the schema, so the form offers exactly what the installed solver reads, and a kind added to the
case file appears in it with no change here.

A document is the case file's content as plain data -- mappings, lists, strings, numbers -- and every
edit returns a new document rather than changing one. :func:`form_sections` lays a document out as
rows, one per setting, each carrying what the page needs to draw it and the path that addresses it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from collections.abc import Mapping, Sequence

__all__ = [
    "KIND",
    "CaseSchema",
    "Row",
    "add_entry",
    "add_item",
    "entry_kinds",
    "field_at",
    "form_sections",
    "parse_input",
    "plain_doc",
    "remove_at",
    "set_kind",
    "set_value",
    "unset",
    "without_unread",
]

#: The key naming a value's kind in its mapping.
KIND = "kind"

#: A path into a document: mapping keys and list indices, from the top.
Path = tuple[str | int, ...]

#: A reStructuredText role in a docstring -- ``:class:`~aquaflux.flow.PinnedPoint``` -- whose text is
#: what a reader of a form wants: the last name, without the role or the module.
_ROLE = re.compile(r":[a-z]+:`~?(?:[\w.]*\.)?([^`]+)`")
_LITERAL = re.compile(r"``([^`]+)``")


def plain_doc(text: str) -> str:
    """A docstring fragment as plain text: roles reduced to their names, literals unquoted."""
    return _LITERAL.sub(r"\1", _ROLE.sub(r"\1", text))


@dataclasses.dataclass(frozen=True)
class CaseSchema:
    """The case-file schema, with the lookups a form needs.

    Parameters
    ----------
    data : mapping
        What ``aquaflux schema`` prints: ``root``, ``one_form_sections``, ``scopes_from`` and
        ``kinds``.
    reads : frozenset of str, optional
        The scopes of setting a case reads (see :meth:`reading`); unset, every setting is offered.
    """

    data: Mapping
    reads: frozenset[str] | None = None

    def reading(self, document: Mapping) -> CaseSchema:
        """This schema offering only the settings ``document``'s physics reads.

        A setting the schema gives a ``scope`` is offered only when the kind of the document's
        ``scopes_from`` section (its physics) lists that scope in its ``reads``: a laminar case is
        not offered the turbulence advection scheme, nor a wall's ``k`` condition. While that kind is
        not chosen, or reads nothing in particular, everything is offered.
        """
        section = self.data.get("scopes_from")
        chosen = document.get(section) if section else None
        kind = chosen.get(KIND) if isinstance(chosen, Mapping) else None
        if not self.has(kind) or "reads" not in self.data["kinds"][kind]:
            return dataclasses.replace(self, reads=None)
        return dataclasses.replace(self, reads=frozenset(self.data["kinds"][kind]["reads"]))

    @property
    def root(self) -> str:
        """The kind whose fields are the file's top-level sections."""
        return self.data["root"]

    def fields(self, kind: str) -> list[dict]:
        """A kind's fields, in declaration order: those of a scope the case reads, if :attr:`reads`."""
        fields = self.data["kinds"][kind]["fields"]
        if self.reads is None:
            return fields
        return [field for field in fields if field.get("scope", None) in (None, *self.reads)]

    def field(self, kind: str, name: str) -> dict | None:
        """One field of a kind, or ``None`` if it has none of that name."""
        return next((field for field in self.fields(kind) if field["name"] == name), None)

    def summary(self, kind: str) -> str:
        """A kind's one-line description."""
        return plain_doc(self.data["kinds"][kind]["summary"])

    def has(self, kind: object) -> bool:
        """Whether ``kind`` is a kind the schema knows."""
        return isinstance(kind, str) and kind in self.data["kinds"]


@dataclasses.dataclass(frozen=True)
class Row:
    """One line of the form.

    Attributes
    ----------
    path : tuple
        The setting it edits, from the top of the document.
    widget : str
        How it is drawn and edited: ``kind`` (a choice of nested kind), ``choice``, ``choices`` (a
        list of fixed choices, picked several at once), ``integer``,
        ``number``, ``string``, ``boolean``, ``numbers`` (a list of numbers in one box),
        ``strings`` (a list of strings in one box), ``list`` (a heading for a list of values, with a
        choice of kind to add), ``table`` (a heading for named entries, with a name to add), ``entry``
        (one table entry's heading), ``item`` (one list entry's heading), or ``raw`` (anything else,
        edited as JSON text).
    label : str
        Its name.
    depth : int
        How deeply it is nested, for indentation.
    value : object
        What it shows: the setting, or for a ``kind`` row the kind's name (empty when unset).
    placeholder : str
        The default it falls back to, shown when it is unset.
    items : list
        A choice row's options: ``{"title", "value"}`` each.
    is_set : bool
        Whether the document states it, rather than leaving it to its default.
    removable : bool
        Whether it can be removed: an optional setting (back to its default), a list item, a table
        entry, or a key the schema does not know.
    doc : str
        Its help text.
    unknown : bool
        Whether the document holds it but the schema has no such field: shown so it can be removed.
    kind_summary : str
        For a group heading (a ``kind``, ``item`` or ``entry`` row), the one-line description of the
        kind it holds; empty otherwise.
    contents : str
        For a group heading, what is set inside it, to show while it is folded: each setting stated
        directly in it, by name and value, separated by ``·``. Empty when nothing is.
    set_inside : int
        For a group heading, how many settings anywhere inside it the document states.
    has_inside : bool
        For a group heading, whether it holds any rows at all, and so has anything to fold.
    """

    path: Path
    widget: str
    label: str
    depth: int
    value: object = None
    placeholder: str = ""
    items: tuple[dict, ...] = ()
    is_set: bool = False
    removable: bool = False
    doc: str = ""
    unknown: bool = False
    kind_summary: str = ""
    contents: str = ""
    set_inside: int = 0
    has_inside: bool = False

    def to_state(self) -> dict:
        """The row as the page holds it; its path is a JSON string, to travel back with an edit."""
        state = dataclasses.asdict(self)
        state["path"] = json.dumps(list(self.path))
        state["items"] = list(self.items)
        state["id"] = state["path"]
        return state


# --- reading a value's form off the schema ----------------------------------------------------


def _types(accepts: Sequence[dict]) -> set[str]:
    return {atom["type"] for atom in accepts}


def _nested_kinds(accepts: Sequence[dict]) -> list[str]:
    return [kind for atom in accepts if atom["type"] == "nested" for kind in atom["kinds"]]


def _of(accepts: Sequence[dict], type_: str) -> list[dict]:
    return next((atom["of"] for atom in accepts if atom["type"] == type_), [])


def _kind_of(value: object, accepts: Sequence[dict]) -> str | None:
    """The kind a mapping in a nested position is: its own ``kind``, or the position's only one."""
    if not isinstance(value, Mapping):
        return None
    if KIND in value:
        return value[KIND]
    kinds = _nested_kinds(accepts)
    return kinds[0] if len(kinds) == 1 else None


#: What an unset setting shows when the schema states no value for it: the solver decides from the rest
#: of the case (a wall's ``k`` condition depends on whether the case is turbulent), as its help says.
NOT_SET = "Not set"

#: What an unset setting shows when the schema says leaving it unset turns the feature off.
OFF = "Off"

#: Marks the option, or the value, a setting takes when it is left unset.
DEFAULT_MARK = " (default)"


def _default(field: dict) -> object:
    """The value an unset setting takes: its own default, or the one the schema resolved it to."""
    default = field.get("default")
    return field.get("resolved_default") if default is None else default


def _default_text(field: dict) -> str:
    """What an unset setting shows: its default, marked as one; "required"; :data:`OFF`; or :data:`NOT_SET`."""
    if "default" not in field or field.get("required_in_scope"):
        return "required"  # offered only where its scope is read, and there it is required
    if field.get("off"):
        return OFF
    default = _default(field)
    return NOT_SET if default is None else _show(default) + DEFAULT_MARK


def _options(values: Sequence[object], default: object) -> tuple[dict, ...]:
    """Dropdown items for ``values``, the one an unset setting takes marked as the default."""
    return tuple(
        {
            "title": _show(v) + (DEFAULT_MARK if v == default and default is not None else ""),
            "value": v,
        }
        for v in values
    )


def _show(value: object) -> str:
    """A value as the text a box or an option shows."""
    if value is None:
        return NOT_SET
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"  # a default such as 5/9 is shown, not spelt out to 16 digits
    if isinstance(value, list):
        return ", ".join(_show(item) for item in value) or "none"
    if isinstance(value, Mapping):
        return str(value.get(KIND, "…"))
    return str(value)


def _scalar_widget(accepts: Sequence[dict]) -> str | None:
    """The widget for a position holding one plain value, or ``None`` if it holds something else."""
    types = _types(accepts) - {"null"}
    if types == {"choice"}:
        return "choice"
    if types and types <= {"integer", "number", "string", "boolean"} and len(types) == 1:
        return types.pop()
    if types == {"list"}:
        of = _types(_of(accepts, "list"))
        if of <= {"number", "integer"} and of:
            return "numbers"
        if of == {"string"}:
            return "strings"
        if of == {"choice"}:
            return "choices"
    return None


# --- laying a document out as rows ------------------------------------------------------------


def form_sections(schema: CaseSchema, document: Mapping) -> list[dict]:
    """The form for ``document``: one section per top-level field of the case, each a list of rows.

    Parameters
    ----------
    schema : CaseSchema
        The case-file schema.
    document : mapping
        The case file's content.

    Returns
    -------
    list of dict
        ``{"key", "title", "doc", "is_set", "rows"}`` per section, in the order the case declares
        them, then a section of keys the schema does not know, if the document holds any. ``rows``
        are :class:`Row` objects.
    """
    schema = schema.reading(document)  # offer only what the case's physics reads
    sections = []
    known = set()
    for field in schema.fields(schema.root):
        name = field["name"]
        known.add(name)
        rows: list[Row] = []
        _field_rows(schema, field, document, (name,), 0, rows, top=True)
        sections.append(
            {
                "key": name,
                "title": name.replace("_", " ").capitalize(),
                "doc": plain_doc(field["doc"]),
                "is_set": name in document,
                "rows": rows,
            }
        )
    stray = [key for key in document if key not in known]
    if stray:
        sections.append(
            {
                "key": "_unknown",
                "title": "Not in the case file's specification",
                "doc": "Sections aquaflux does not read. Remove them, or the file is refused.",
                "is_set": True,
                "rows": [_unknown_row((key,), key, document[key], 0) for key in stray],
            }
        )
    for section in sections:
        section["rows"] = _with_group_contents(schema, section["rows"])
    return sections


#: The rows that head a group of settings, which can be folded.
GROUP_WIDGETS = ("kind", "item", "entry")

#: The rows that are a heading or a container rather than one setting, and so are not counted as set.
_HEADINGS = (*GROUP_WIDGETS, "list", "table")


def _with_group_contents(schema: CaseSchema, rows: list[Row]) -> list[Row]:
    """``rows`` with each group heading's kind summary and what is set inside it filled in."""
    filled = []
    for index, row in enumerate(rows):
        if row.widget not in GROUP_WIDGETS:
            filled.append(row)
            continue
        inside = []
        for later in rows[index + 1 :]:
            if later.path[: len(row.path)] != row.path or len(later.path) <= len(row.path):
                break
            inside.append(later)
        direct = [
            r for r in inside if len(r.path) == len(row.path) + 1 and r.is_set and not r.unknown
        ]
        filled.append(
            dataclasses.replace(
                row,
                kind_summary=schema.summary(row.value) if schema.has(row.value) else "",
                contents=" · ".join(f"{r.label} {_shown(r)}".strip() for r in direct),
                set_inside=sum(1 for r in inside if r.is_set and r.widget not in _HEADINGS),
                has_inside=bool(inside),
            )
        )
    return filled


def _shown(row: Row) -> str:
    """A set row's value as a folded group's summary names it."""
    if row.widget in ("list", "table"):
        return ""
    if isinstance(row.value, list):
        return ", ".join(_show(item) for item in row.value)
    return "" if row.value is None else _show(row.value)


def _unknown_row(path: Path, label: str, value: object, depth: int) -> Row:
    return Row(
        path, "raw", label, depth, json.dumps(value), is_set=True, removable=True, unknown=True
    )


def _field_rows(
    schema: CaseSchema,
    field: dict,
    parent: Mapping,
    path: Path,
    depth: int,
    rows: list[Row],
    top: bool = False,
) -> None:
    """The rows for one field of a mapping: its own, then its value's if that is nested."""
    name, accepts = field["name"], field["accepts"]
    is_set = name in parent
    value = parent[name] if is_set else _default(field)
    label = name.replace("_", " ")
    doc = plain_doc(field["doc"])
    removable = is_set and not field["required"]
    types = _types(accepts)
    widget = _scalar_widget(accepts)

    if widget is not None:
        shown = value if is_set else None
        if widget in ("numbers", "strings") and shown is not None:
            shown = ", ".join(str(item) for item in shown)
        items = ()
        if widget == "choice":
            values = [v for atom in accepts if atom["type"] == "choice" for v in atom["values"]]
            items = _options(values, _default(field))
        elif widget == "choices":
            choices = _of(accepts, "list")
            values = [v for atom in choices if atom["type"] == "choice" for v in atom["values"]]
            items = _options(values, None)  # the default is a whole list, shown as the placeholder
            shown = list(value) if is_set else []
        elif widget == "boolean":
            items = _options([True, False], _default(field))
        rows.append(
            Row(
                path, widget, label, depth, shown, placeholder=_default_text(field),
                items=items, is_set=is_set, removable=removable, doc=doc,
            )
        )  # fmt: skip
        return

    if "nested" in types and not types - {"nested", "null"}:
        kinds = _nested_kinds(accepts)
        kind = _kind_of(value, accepts) if value is not None else None
        if kind is None and value is None and field["required"] and len(kinds) == 1:
            # A required section of one form, not yet written: its fields are still the form's.
            kind, value = kinds[0], {}
        optional = "null" in types or not field["required"]
        # A kind row heads the value's own rows, which then sit one level in; a required section of one
        # form shows no kind row, and its fields are the section's own top level.
        heading = len(kinds) > 1 or optional or not top
        if heading:
            default = _default(field)
            items = _options(kinds, _kind_of(default, accepts) if default is not None else None)
            rows.append(
                Row(
                    # Unset is None, not "": a dropdown takes "" for a chosen (blank) option and so
                    # would not show its placeholder.
                    path, "kind", label, depth, kind,
                    placeholder=_default_text(field), items=items, is_set=is_set,
                    removable=removable, doc=doc or (schema.summary(kind) if schema.has(kind) else ""),
                )
            )  # fmt: skip
        if schema.has(kind) and isinstance(value, Mapping):
            _value_rows(
                schema, kind, value if is_set else {}, path, depth + (1 if heading else 0), rows
            )
        return

    if types - {"null"} == {"list"}:
        of = _of(accepts, "list")
        kinds = _nested_kinds(of)
        items = tuple({"title": k, "value": k} for k in kinds)
        rows.append(Row(path, "list", label, depth, None, items=items, is_set=is_set, doc=doc))
        for index, item in enumerate(value or []):
            item_kind = _kind_of(item, of)
            rows.append(
                Row(
                    (*path, index), "item", f"{index + 1}. {item_kind or 'item'}", depth + 1,
                    item_kind, is_set=True, removable=True,
                    doc=schema.summary(item_kind) if schema.has(item_kind) else "",
                )
            )  # fmt: skip
            if schema.has(item_kind):
                _value_rows(schema, item_kind, item, (*path, index), depth + 2, rows)
        return

    if types - {"null"} == {"table"}:
        of = _of(accepts, "table")
        kinds = _nested_kinds(of)
        rows.append(Row(path, "table", label, depth, None, is_set=is_set, doc=doc))
        for entry, item in (value or {}).items():
            entry_kind = _kind_of(item, of)
            rows.append(
                Row(
                    (*path, entry), "entry", entry, depth + 1, entry_kind,
                    items=tuple({"title": k, "value": k} for k in kinds),
                    is_set=True, removable=True,
                    doc=schema.summary(entry_kind) if schema.has(entry_kind) else "",
                )
            )  # fmt: skip
            if schema.has(entry_kind):
                _value_rows(schema, entry_kind, item, (*path, entry), depth + 2, rows)
        return

    rows.append(
        Row(
            path, "raw", label, depth, json.dumps(value) if is_set else "",
            placeholder=_default_text(field), is_set=is_set, removable=removable, doc=doc,
        )
    )  # fmt: skip


def _value_rows(
    schema: CaseSchema, kind: str, value: Mapping, path: Path, depth: int, rows: list[Row]
) -> None:
    """The rows for every field of one nested value, then any key its kind does not have."""
    for field in schema.fields(kind):
        _field_rows(schema, field, value, (*path, field["name"]), depth, rows)
    names = {field["name"] for field in schema.fields(kind)}
    for key in value:
        if key != KIND and key not in names:
            rows.append(_unknown_row((*path, key), key, value[key], depth))


# --- editing a document ----------------------------------------------------------------------


def _container(document: dict, path: Path, schema: CaseSchema | None = None) -> object:
    """The mapping or list holding ``path``'s last step, created on the way where it is absent.

    A mapping created for a nested value that was left to its default takes that value's kind -- the
    only one its position allows -- so the document stays one aquaflux reads.
    """
    node = document
    for depth, step in enumerate(path[:-1]):
        child = node[step] if isinstance(node, list) else node.get(step)
        if child is None:
            child = {}
            if schema is not None:
                kind = _default_kind(schema, document, path[: depth + 1])
                if kind is not None:
                    child[KIND] = kind
            node[step] = child
        node = child
    return node


def field_at(schema: CaseSchema, document: Mapping, path: Sequence) -> dict | None:
    """The schema's field for the setting at ``path``, following the kinds the document states.

    Each step names a field of the value above it, whose kind is the one the document gives it -- or,
    where the document leaves it to its default, the default's kind or the position's only one. A
    list index or a table entry name steps into one value held by the list or table.

    Returns
    -------
    dict or None
        The field the last step names, or ``None`` if ``path`` leads somewhere the schema does not
        describe or ends inside a list or table.
    """
    kind: str | None = schema.root
    node: object = document
    field: dict | None = None
    for step in path:
        accepts = field["accepts"] if field is not None else None
        if accepts is not None and _types(accepts) - {"null"} in ({"list"}, {"table"}):
            # One item of the list, or one entry of the table, above it.
            of = _of(accepts, "list") or _of(accepts, "table")
            node = node[step] if isinstance(node, list) else (node or {}).get(step)
            kind, field = _resolved_kind(node, of, None), None
            continue
        if kind is None:
            return None
        field = schema.field(kind, step)
        if field is None:
            return None
        node = node.get(step) if isinstance(node, Mapping) else None
        kind = _resolved_kind(node, field["accepts"], _default(field))
    return field


def _resolved_kind(node: object, accepts: Sequence[dict], default: object) -> str | None:
    """The kind of a value in a position: its own, its default's, or the position's only one."""
    kind = _kind_of(node, accepts) if node is not None else None
    if kind is None and isinstance(default, Mapping):
        kind = _kind_of(default, accepts)
    if kind is None:
        kinds = _nested_kinds(accepts)
        kind = kinds[0] if len(kinds) == 1 else None
    return kind


def entry_kinds(schema: CaseSchema, document: Mapping, path: Sequence) -> list[str]:
    """The kinds an entry of the table at ``path`` may be, in the schema's order."""
    field = field_at(schema, document, path)
    return _nested_kinds(_of(field["accepts"], "table")) if field else []


def _default_kind(schema: CaseSchema, document: Mapping, path: Path) -> str | None:
    """The kind a value at ``path`` takes when it is created: its default's, or its position's only."""
    field = field_at(schema, document, path)
    return _resolved_kind(None, field["accepts"], _default(field)) if field else None


def set_value(
    document: Mapping, path: Sequence, value: object, schema: CaseSchema | None = None
) -> dict:
    """``document`` with the setting at ``path`` set to ``value``.

    Parameters
    ----------
    document : mapping
        The case file's content.
    path : sequence
        Where, from the top: mapping keys and list indices.
    value : object
        Plain data.
    schema : CaseSchema, optional
        Given, a nested value created on the way -- one that was left to its default -- takes its kind.

    Returns
    -------
    dict
        A new document; ``document`` is not changed.
    """
    edited = copy.deepcopy(dict(document))
    path = tuple(path)
    _container(edited, path, schema)[path[-1]] = copy.deepcopy(value)
    return edited


def unset(document: Mapping, path: Sequence) -> dict:
    """``document`` without the setting at ``path``: an optional setting goes back to its default.

    Returns
    -------
    dict
        A new document; unchanged if nothing is there.
    """
    edited = copy.deepcopy(dict(document))
    path = tuple(path)
    node = edited
    for step in path[:-1]:
        node = node[step] if isinstance(node, list) else node.get(step)
        if node is None:
            return edited
    if isinstance(node, list):
        del node[path[-1]]
    else:
        node.pop(path[-1], None)
    return edited


#: The edit that removes a list item or a table entry is the same one: remove what ``path`` addresses.
remove_at = unset


def set_kind(document: Mapping, path: Sequence, kind: str, schema: CaseSchema) -> dict:
    """``document`` with the value at ``path`` made a ``kind``, keeping the settings it shares with it.

    An empty ``kind`` removes the value, leaving an optional one to its default.

    Returns
    -------
    dict
        A new document.
    """
    if not kind:
        return unset(document, path)
    node: object = document
    for step in path:
        node = node[step] if isinstance(node, list) else (node or {}).get(step)
    kept = {}
    if isinstance(node, Mapping):
        names = {field["name"] for field in schema.fields(kind)}
        kept = {key: item for key, item in node.items() if key in names}
    changed = set_value(document, path, {KIND: kind, **kept}, schema)
    if list(path) == [schema.data.get("scopes_from")]:
        # A new physics: drop what it does not read, which it would refuse.
        changed = without_unread(changed, schema)
    return changed


def without_unread(document: Mapping, schema: CaseSchema) -> dict:
    """``document`` without the settings its physics does not read (see :meth:`CaseSchema.reading`).

    Each setting the schema gives a scope its physics does not list is removed, wherever it sits: a
    laminar case loses a wall's ``k`` condition and the turbulence advection scheme. A document whose
    physics reads nothing in particular is returned unchanged.

    Returns
    -------
    dict
        A new document.
    """
    reads = schema.reading(document).reads
    if reads is None:
        return dict(document)
    everything = dataclasses.replace(schema, reads=None)
    return _kept(everything, schema.root, document, reads)


def _kept(schema: CaseSchema, kind: str, value: Mapping, reads: frozenset[str]) -> dict:
    """``value``, a mapping of ``kind``, without its settings (at any depth) of a scope not read."""
    kept = {}
    for key, item in value.items():
        field = schema.field(kind, key) if schema.has(kind) else None
        if field is not None and field.get("scope") not in (None, *reads):
            continue
        kept[key] = item if field is None else _kept_inside(schema, field["accepts"], item, reads)
    return kept


def _kept_inside(
    schema: CaseSchema, accepts: Sequence[dict], item: object, reads: frozenset[str]
) -> object:
    """A setting's value with any nested value of it filtered by :func:`_kept`."""
    if isinstance(item, Mapping):
        kind = _kind_of(item, accepts)
        if schema.has(kind):
            return _kept(schema, kind, item, reads)
        table = _of(accepts, "table")
        if table:
            return {name: _kept_inside(schema, table, entry, reads) for name, entry in item.items()}
        return dict(item)
    if isinstance(item, list):
        of = _of(accepts, "list")
        return [_kept_inside(schema, of, entry, reads) for entry in item]
    return item


def add_item(document: Mapping, path: Sequence, kind: str, schema: CaseSchema) -> dict:
    """``document`` with a new value of ``kind`` appended to the list at ``path``."""
    node: object = document
    for step in path:
        node = node[step] if isinstance(node, list) else (node or {}).get(step)
    return set_value(document, path, [*(node or []), {KIND: kind}], schema)


def add_entry(document: Mapping, path: Sequence, name: str, kind: str, schema: CaseSchema) -> dict:
    """``document`` with an entry ``name`` of ``kind`` added to the table at ``path``.

    Raises
    ------
    ValueError
        If ``name`` is empty or the table already has an entry of that name.
    """
    name = name.strip()
    node: object = document
    for step in path:
        node = node[step] if isinstance(node, list) else (node or {}).get(step)
    if not name:
        raise ValueError("an entry needs a name.")
    if isinstance(node, Mapping) and name in node:
        raise ValueError(f"there is already an entry named {name!r}.")
    return set_value(document, (*path, name), {KIND: kind}, schema)


def parse_input(widget: str, text: object) -> tuple[bool, object]:
    """What a box's text means for a row of ``widget``: ``(True, value)``, or ``(False, None)`` to unset.

    An empty box unsets the setting, so it falls back to its default.

    Parameters
    ----------
    widget : str
        The row's widget (see :class:`Row`).
    text : object
        What the page sent: text from a box, or the value of a switch or a choice.

    Returns
    -------
    tuple
        Whether to set, and the value.

    Raises
    ------
    ValueError
        If the text does not read as what the widget holds -- a word where a number belongs, say.
    """
    if widget in ("boolean", "choice"):
        return True, text
    if widget == "choices":
        return (True, list(text)) if text else (False, None)
    text = "" if text is None else str(text).strip()
    if not text:
        return False, None
    if widget == "integer":
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"{text!r} is not a whole number.")
        return True, int(number)
    if widget == "number":
        return True, float(text)
    if widget == "string":
        return True, text
    if widget == "numbers":
        return True, [float(part) for part in re.split(r"[,\s]+", text) if part]
    if widget == "strings":
        return True, [part.strip() for part in text.split(",") if part.strip()]
    if widget == "raw":
        try:
            return True, json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{text!r} is not JSON: {error.msg}.") from error
    raise ValueError(f"a {widget} row is not edited by typing.")
