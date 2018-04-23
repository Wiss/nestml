#
# ASTCompoundStmt.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
from pynestml.meta_model.ASTNode import ASTNode
from pynestml.meta_model.ASTIfStmt import ASTIfStmt
from pynestml.meta_model.ASTWhileStmt import ASTWhileStmt
from pynestml.meta_model.ASTForStmt import ASTForStmt


class ASTCompoundStmt(ASTNode):
    """
    This class is used to store compound statements.
    Grammar:
        compoundStmt : ifStmt
                | forStmt
                | whileStmt;
    """
    __ifStmt = None
    __whileStmt = None
    __forStmt = None

    def __init__(self, if_stmt=None, while_stmt=None, for_stmt=None, source_position=None):
        """
        Standard constructor.
        :param if_stmt: a if statement object
        :type if_stmt: ASTIfStmt
        :param while_stmt: a while statement object
        :type while_stmt: ASTWhileStmt
        :param for_stmt: a for statement object
        :type for_stmt: ASTForStmt
        :param source_position: The source position of the assignment
        :type source_position: ASTSourceLocation
        """
        assert (if_stmt is None or isinstance(if_stmt, ASTIfStmt)), \
            '(PyNestML.AST.CompoundStmt) Wrong type of if-statement provided (%s)!' % type(if_stmt)
        assert (while_stmt is None or isinstance(while_stmt, ASTWhileStmt)), \
            '(PyNestML.AST.CompoundStmt) Wrong type of while-statement provided (%s)!' % type(while_stmt)
        assert (for_stmt is None or isinstance(for_stmt, ASTForStmt)), \
            '(PyNestML.AST.CompoundStmt) Wrong type of for-statement provided (%s)!' % type(for_stmt)
        super(ASTCompoundStmt, self).__init__(source_position)
        self.__ifStmt = if_stmt
        self.__whileStmt = while_stmt
        self.__forStmt = for_stmt
        return

    def is_if_stmt(self):
        """
        Returns whether it is an "if" statement or not.
        :return: True if if stmt, False else.
        :rtype: bool
        """
        return self.__ifStmt is not None and isinstance(self.__ifStmt, ASTIfStmt)

    def get_if_stmt(self):
        """
        Returns the "if" statement.
        :return: the "if" statement.
        :rtype: ASTIfStmt
        """
        return self.__ifStmt

    def is_while_stmt(self):
        """
        Returns whether it is an "while" statement or not.
        :return: True if "while" stmt, False else.
        :rtype: bool
        """
        return self.__whileStmt is not None and isinstance(self.__whileStmt, ASTWhileStmt)

    def get_while_stmt(self):
        """
        Returns the while statement.
        :return: the while statement.
        :rtype: ASTWhileStmt
        """
        return self.__whileStmt

    def is_for_stmt(self):
        """
        Returns whether it is an "for" statement or not.
        :return: True if "for" stmt, False else.
        :rtype: bool
        """
        return self.__forStmt is not None and isinstance(self.__forStmt, ASTForStmt)

    def get_for_stmt(self):
        """
        Returns the for statement.
        :return: the for statement.
        :rtype: ASTForStmt
        """
        return self.__forStmt

    def get_parent(self, ast=None):
        """
        Indicates whether a this node contains the handed over node.
        :param ast: an arbitrary meta_model node.
        :type ast: AST_
        :return: AST if this or one of the child nodes contains the handed over element.
        :rtype: AST_ or None
        """
        if self.is_if_stmt():
            if self.get_if_stmt() is ast:
                return self
            elif self.get_if_stmt().get_parent(ast) is not None:
                return self.get_if_stmt().get_parent(ast)
        if self.is_while_stmt():
            if self.get_while_stmt() is ast:
                return self
            elif self.get_while_stmt().get_parent(ast) is not None:
                return self.get_while_stmt().get_parent(ast)
        if self.is_for_stmt():
            if self.is_for_stmt() is ast:
                return self
            elif self.get_for_stmt().get_parent(ast) is not None:
                return self.get_for_stmt().get_parent(ast)
        return None

    def equals(self, other=None):
        """
        The equals method.
        :param other: a different object.
        :type other: object
        :return: True if equal, otherwise False.
        :rtype: bool
        """
        if not isinstance(other, ASTCompoundStmt):
            return False
        if self.get_for_stmt() is not None and other.get_for_stmt() is not None and \
                not self.get_for_stmt().equals(other.get_for_stmt()):
            return False
        if self.get_while_stmt() is not None and other.get_while_stmt() is not None and \
                not self.get_while_stmt().equals(other.get_while_stmt()):
            return False
        if self.get_if_stmt() is not None and other.get_if_stmt() is not None and \
                not self.get_if_stmt().equals(other.get_if_stmt()):
            return False
        return True
