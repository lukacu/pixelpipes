from inspect import isclass
from typing import Any, NamedTuple, Optional, Tuple, List
import typing

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from docutils.parsers.rst import directives, Directive

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.util.nodes import make_id
from sphinx.roles import XRefRole
#from sphinx.ext.autodoc import ModuleDocumenter, ClassDocumenter, Documenter
from autoapi.documenters import AutoapiDocumenter as Documenter, AutoapiModuleDocumenter as ModuleDocumenter, AutoapiClassDocumenter as ClassDocumenter
from sphinx.domains import Domain, ObjType, Index
from sphinx.domains.python import py_sig_re, PyObject
from sphinx.environment import BuildEnvironment

def is_node(obj):
    from pixelpipes.graph import Node

    if not isclass(obj.object):
        return False
    if not issubclass(obj.object, Node):
        return False

    return not obj.object.hidden()

class IndexEntry(NamedTuple):
    docname: str
    node_id: str
    objtype: str
    aliased: bool

class NodeEntry(PyObject):

    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update({
        'final': directives.flag,
    })

    allow_nesting = True

    def get_signature_prefix(self, sig: str) -> List[nodes.Node]:
        return [nodes.Text(self.objtype), addnodes.desc_sig_space()]

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        if self.objtype == 'node':
            if not modname:
                return _('%s (built-in class)') % name_cls[0]
            return _('%s (node in %s)') % (name_cls[0], modname)
        else:
            return ''

    def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                             signode: addnodes.desc_signature) -> None:
        modname = self.options.get('module', self.env.ref_context.get('py:module'))
        fullname = (modname + '.' if modname else '') + name_cls[0]
        node_id = make_id(self.env, self.state.document, '', fullname)
        signode['ids'].append(node_id)

        # Assign old styled node_id(fullname) not to break old hyperlinks (if possible)
        # Note: Will removed in Sphinx-5.0  (RemovedInSphinx50Warning)
        if node_id != fullname and fullname not in self.state.document.ids:
            signode['ids'].append(fullname)

        self.state.document.note_explicit_target(signode)

        domain = typing.cast(NodeDomain, self.env.get_domain('nodes'))
        domain.note_object(fullname, self.objtype, node_id, location=signode)

        canonical_name = self.options.get('canonical')
        if canonical_name:
            domain.note_object(canonical_name, self.objtype, node_id, aliased=True,
                               location=signode)

        if 'noindexentry' not in self.options:
            indextext = self.get_index_text(modname, name_cls)
            if indextext:
                self.indexnode['entries'].append(('single', indextext, node_id, '', None))

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

class NodeXRefRole(XRefRole):
    def process_link(self, env: BuildEnvironment, refnode: nodes.Element,
                     has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode['py:module'] = env.ref_context.get('py:module')
        refnode['py:class'] = env.ref_context.get('py:class')
        if not has_explicit_title:
            title = title.lstrip('.')    # only has a meaning for the target
            target = target.lstrip('~')  # only has a meaning for the title
            # if the first character is a tilde, don't display the module/class
            # parts of the contents
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        # if the first character is a dot, search more specific namespaces first
        # else search builtins first
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        return title, target

class NodeDomain(Domain):

    name = 'nodes'
    label = 'Pixelpipes'
    object_types = {
        'module': ObjType(_('module'), 'module'),
        'node': ObjType(_('node'), 'node'),
        'operation': ObjType(_('operation'), 'operation', 'node'),
        'macro':  ObjType(_('macro'),  'macro', 'node'),
        'token':  ObjType(_('token'),  'token'),
        'res':  ObjType(_('resource'),  'resource'),
    }

    directives = {
        'module':      NodeEntry,
        'operation':   NodeEntry,
        'node':        NodeEntry,
        'macro':       NodeEntry,
    }
    roles = {
        'node':  NodeXRefRole(),
    }
    initial_data = {
        'nodes': {},
    }
    indices = [
        NodeIndex,
    ]

    def clear_doc(self, docname):
        pass
        #for fullname in list(self.data['nodes'].keys()):
        #    fn, _ = self.data['nodes'].get(fullname)
        #    if fn == docname:
        #        self.data['nodes'].pop(fullname)


    def note_object(self, name: str, objtype: str, node_id: str,
                    aliased: bool = False, location: Any = None) -> None:
        """Note an object for cross reference.

        .. versionadded:: 2.1
        """
        
        if name in self.data["nodes"]:
            other = self.data["nodes"][name]
            if other.aliased and aliased is False:
                # The original definition found. Override it!
                pass
            elif other.aliased is False and aliased:
                # The original definition is already registered.
                return

        self.data["nodes"][name] = IndexEntry(self.env.docname, node_id, "node", aliased)

class NodesDocumenter(ModuleDocumenter):
    domain = NodeDomain.name
    objtype = 'nodes'
    directivetype = ModuleDocumenter.directivetype
    priority = 10 + ModuleDocumenter.priority
    option_spec = dict(ModuleDocumenter.option_spec)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any
                            ) -> bool:
        from pixelpipes.graph import Node
        try:
            return issubclass(member, Node)
        except TypeError:
            return False

    def add_directive_header(self, sig: str) -> None:
        pass

    def add_content(self,
                    more_content: Optional[StringList],
                    no_docstring: bool = False
                    ) -> None:
        pass

    def get_module_members(self):
        members = super().get_module_members()
        print(members)
        return {k: v for k, v in members.items() if is_node(v)}


class NodeDocumenter(ClassDocumenter):
    domain = NodeDomain.name
    objtype = 'node'
    directivetype = ClassDocumenter.directivetype
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

        return "  (" + ", ".join(args) + ")"

    def add_directive_header(self, sig: str) -> None:
        from sphinx.util import inspect
        sourcename = self.get_sourcename()

        if self.doc_as_attr:
            self.directivetype = 'attribute'

        from pixelpipes.graph import Node, Operation, Macro
        node_object: Node = self.object

        if issubclass(node_object, Operation):
            self.directivetype = "operation"
        elif issubclass(node_object, Macro):
            self.directivetype = "macro"
        else:
            self.directivetype = "node"

        Documenter.add_directive_header(self, sig)

        canonical_fullname = self.get_canonical_fullname()
        if not self.doc_as_attr and canonical_fullname and self.fullname != canonical_fullname:
            self.add_line('   :canonical: %s' % canonical_fullname, sourcename)

        #self.add_line('   .. term: %s' % canonical_fullname, sourcename)

    def add_content(self,
                    more_content: Optional[StringList]
                    ) -> None:

        from pixelpipes.graph import Node, Input
        from attributee import Integer, String, Float, Any, Boolean, is_undefined

        super().add_content(more_content)

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
