from inspect import isclass
from typing import Any, Optional

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from docutils.parsers.rst import directives, Directive

from sphinx.application import Sphinx
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.ext.autodoc import ModuleDocumenter, ClassDocumenter, Documenter
from sphinx.domains import Domain, ObjType, Index
from sphinx.domains.python import py_sig_re, PyObject

def is_node(obj):
    from pixelpipes.graph import Node

    if not isclass(obj.object):
        return False
    if not issubclass(obj.object, Node):
        return False

    return not obj.object.hidden()


class NodeIndex(Index):
    name = 'nodeindex'
    localname = _('Node Index')
    shortname = _('nodes')

    def generate(self, docnames=None):
        content = {}
        # list of prefixes to ignore
        ignores = self.domain.env.config['modindex_common_prefix']
        ignores = sorted(ignores, key=len, reverse=True)
        # list of all packages, sorted by package name
        packages = sorted(self.domain.data['nodes'].items(),
                         key=lambda x: x[0].lower())
        # sort out collapsable packages
        prev_pkgname = ''
        num_toplevels = 0
        for pkgname, (docname, synopsis, platforms, deprecated) in packages:
            if docnames and docname not in docnames:
                continue

            for ignore in ignores:
                if pkgname.startswith(ignore):
                    pkgname = pkgname[len(ignore):]
                    stripped = ignore
                    break
            else:
                stripped = ''

            # we stripped the whole package name?
            if not pkgname:
                pkgname, stripped = stripped, ''

            entries = content.setdefault(pkgname[0].lower(), [])

            package = pkgname.split('.')[0]
            if package != pkgname:
                # it's a subpackage
                if prev_pkgname == package:
                    # first subpackage - make parent a group head
                    entries[-1][1] = 1
                elif not prev_pkgname.startswith(package):
                    # subpackage without parent in list, add dummy entry
                    entries.append([stripped + package, 1, '', '', '', '', ''])
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0

            qualifier = deprecated and _('Deprecated') or ''
            entries.append([stripped + pkgname, subtype, docname,
                            'package-' + stripped + pkgname, platforms,
                            qualifier, synopsis])
            prev_pkgname = pkgname

        # apply heuristics when to collapse pkgindex at page load:
        # only collapse if number of toplevel packages is larger than
        # number of subpackages
        collapse = len(packages) - num_toplevels < num_toplevels

        # sort by first letter
        content = sorted(content.items())

        return content, collapse

class NodeDomain(Domain):

    name = 'node'
    label = 'Node'
    object_types = {
        'node': ObjType(_('node'), 'node'),
        'op': ObjType(_('operation'), 'op'),
        'macro':  ObjType(_('macro'),  'macro'),
        'token':  ObjType(_('token'),  'token'),
        'res':  ObjType(_('resource'),  'res'),
    }

    directives = {
        'op':      PyObject,
        'node':    PyObject,
        'macro':    PyObject,
    }
    roles = {
#        'node' :  GolangXRefRole(),
    }
    initial_data = {
        'nodes': {}, 
    }
    indices = [
        NodeIndex,
    ]

    def clear_doc(self, docname):
        for fullname in list(self.data['nodes'].keys()):
            fn, _ = self.data['nodes'].get(fullname)
            if fn == docname:
                self.data['nodes'].pop(fullname)


class NodesDocumenter(ModuleDocumenter):
    domain = 'node'
    objtype = 'nodes'
    directivetype = ModuleDocumenter.objtype
    priority = 10 + ModuleDocumenter.priority
    option_spec = dict(ModuleDocumenter.option_spec)

    def get_module_members(self):
        members = super().get_module_members()
        return {k: v for k, v in members.items() if is_node(v)}



class NodeDocumenter(ClassDocumenter):
    domain = 'node'
    objtype = 'node'
    directivetype = ClassDocumenter.objtype
    priority = 100 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(cls,
                            member: Any, membername: str,
                            isattr: bool, parent: Any) -> bool:
        from pixelpipes.graph import Node
        try:
            return issubclass(member, Node)
        except TypeError:
            return False

    def document_members(self, all_members: bool = False) -> None:
        pass

    def format_signature(self, **kwargs: Any) -> str:
        """Format the signature (arguments and return annotation) of the object.

        Let the user process it via the ``autodoc-process-signature`` event.
        """
        from pixelpipes.graph import Node
        node_object: Node = self.object

        args = []

        for arg_name, _ in node_object.attributes().items():
            args.append(arg_name)

        return " (" + ", ".join(args) + ")"

    def add_directive_header(self, sig: str) -> None:
        from sphinx.util import inspect
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'
        
        Documenter.add_directive_header(self, sig)

        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and self.fullname != canonical_fullname:
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)


    def add_content(self,
                    more_content: Optional[StringList],
                    no_docstring: bool = False
                    ) -> None:

        from pixelpipes.graph import Node, Input
        from attributee import Integer, String, Float, Any, Boolean, is_undefined

        super().add_content(more_content, no_docstring)

        source_name = self.get_sourcename()

        node_object: Node = self.object

        self.add_line('', source_name)

        for arg_name, arg_value in node_object.attributes().items():
            arg_type = ""
            if isinstance(arg_value, Input):
                arg_type = str(arg_value.reftype())
            elif isinstance(arg_value, Integer):
                arg_type = "int"
            elif isinstance(arg_value, Float):
                arg_type = "float"
            elif isinstance(arg_value, String):
                arg_type = "str"
            elif isinstance(arg_value, Boolean):
                arg_type = "bool"
            elif isinstance(arg_value, Any):
                arg_type = "any"
            

            if not is_undefined(arg_value.default):
                arg_type += f" = {arg_value.default!s}"

            self.add_line(
                f"**{arg_name}** [{arg_type}]: {arg_value.description}", source_name)
            self.add_line('', source_name)


def setup(app: Sphinx) -> None:
    app.setup_extension('sphinx.ext.autodoc')  # Require autodoc extension
    app.add_autodocumenter(NodesDocumenter)
    app.add_autodocumenter(NodeDocumenter)
    app.add_domain(NodeDomain)