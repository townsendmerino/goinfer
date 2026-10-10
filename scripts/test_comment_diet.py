#!/usr/bin/env python3
"""Tests for comment_diet.py: python3 scripts/test_comment_diet.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import comment_diet as cd

SRC = '''package p

//go:build linux

// A documents A.
// Found 2026-09-01 by bisecting.
func A() {}

const k = `
// not a comment, a kernel line
// nor this
`

/* block
// inside a block comment
*/
// B stays.
func B() {}
'''


class T(unittest.TestCase):
    def test_raw_string_and_block_comment_lines_are_not_comments(self):
        _, bl = cd.blocks(SRC)
        self.assertEqual(bl, [(5, 6), (17, 17)])

    def test_directive_is_never_part_of_a_block(self):
        _, bl = cd.blocks(SRC)
        self.assertFalse(any(a <= 3 <= b for a, b in bl))

    def test_apply_moves_old_text_verbatim_and_refuses_a_kernel_line(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                os.makedirs('pkg')
                open('pkg/p.go', 'w').write(SRC)
                open('plan.py', 'w').write("FILE='pkg/p.go'\nEDITS=[dict(start=5,end=6,heading='A',new=['A documents A.','History: docs/code-notes/pkg.md#A.'])]\n")
                cd.cmd_apply('plan.py')
                self.assertIn('// A documents A.\n// History: docs/code-notes/pkg.md#A.\nfunc A()', open('pkg/p.go').read())
                notes = open('docs/code-notes/pkg.md').read()
                self.assertIn('## A\n', notes)
                self.assertIn('A documents A.\nFound 2026-09-01 by bisecting.\n```', notes)
                open('plan2.py', 'w').write("FILE='pkg/p.go'\nEDITS=[dict(start=10,end=11,heading=None,new=['x'])]\n")
                with self.assertRaises(SystemExit):
                    cd.cmd_apply('plan2.py')
            finally:
                os.chdir(old)


if __name__ == '__main__':
    unittest.main()
