# based on http://protips.readthedocs.io/link-roles.html
import re
import os

from docutils import nodes


def run_cmd_get_output(cmd):
    return os.popen(cmd).read().strip()

def get_github_rev():
    path = run_cmd_get_output('git rev-parse --short HEAD')
    tag = run_cmd_get_output('git describe --exact-match')
    print('Git commit ID: ', path)
    if len(tag):
        print('Git tag: ', tag)
        path = tag
    return path


def setup(app):
    baseurl = 'https://github.com/lukacu/pixelpipes'
    rev = get_github_rev()
    app.add_role('example', autolink('{}/tree/{}/examples/%s'.format(baseurl, rev)))
    app.add_role('example_file', autolink('{}/blob/{}/examples/%s'.format(baseurl, rev)))
    app.add_role('example_raw', autolink('{}/raw/{}/examples/%s'.format(baseurl, rev)))

def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern % (text,)
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []
    return role