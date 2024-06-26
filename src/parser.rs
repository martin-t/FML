use std::{
    cmp::PartialEq,
    fmt::{self, Debug, Display, Formatter},
};

use serde::{Deserialize, Serialize};

#[rustfmt::skip]
#[allow(clippy::vec_box)] // LATER(martin-t) perf
#[allow(clippy::upper_case_acronyms)]
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub enum AST {
    Integer(i32),
    Boolean(bool),
    Null,

    Variable { name: Identifier, value: Box<AST> },
    Array { size: Box<AST>, value: Box<AST> },
    Object { extends: Box<AST>, members: Vec<Box<AST>> },

    AccessVariable { name: Identifier },
    AccessField { object: Box<AST>, field: Identifier },
    AccessArray { array: Box<AST>, index: Box<AST> },

    AssignVariable { name: Identifier, value: Box<AST> },
    AssignField { object: Box<AST>, field: Identifier, value: Box<AST> },
    AssignArray { array: Box<AST>, index: Box<AST>, value: Box<AST> },

    Function { name: Identifier, parameters: Vec<Identifier>, body: Box<AST> },

    CallFunction { name: Identifier, arguments: Vec<Box<AST>> },
    CallMethod { object: Box<AST>, name: Identifier, arguments: Vec<Box<AST>> },

    Top (Vec<Box<AST>>),
    Block (Vec<Box<AST>>),
    Loop { condition: Box<AST>, body: Box<AST> },
    Conditional { condition: Box<AST>, consequent: Box<AST>, alternative: Box<AST> },

    Print { format: String, arguments: Vec<Box<AST>> },
}

impl AST {
    pub fn integer(i: i32) -> Self {
        Self::Integer(i)
    }
    pub fn boolean(b: bool) -> Self {
        Self::Boolean(b)
    }

    pub fn null() -> Self {
        Self::Null
    }

    pub fn variable(name: Identifier, value: AST) -> Self {
        Self::Variable {
            name,
            value: value.into_boxed(),
        }
    }

    pub fn array(size: AST, value: AST) -> Self {
        Self::Array {
            size: size.into_boxed(),
            value: value.into_boxed(),
        }
    }

    pub fn object(extends: AST, members: Vec<AST>) -> Self {
        Self::Object {
            extends: extends.into_boxed(),
            members: members.into_boxed(),
        }
    }

    pub fn access_variable(name: Identifier) -> Self {
        Self::AccessVariable { name }
    }

    pub fn access_field(object: AST, field: Identifier) -> Self {
        Self::AccessField {
            object: object.into_boxed(),
            field,
        }
    }

    pub fn access_array(array: AST, index: AST) -> Self {
        Self::AccessArray {
            array: array.into_boxed(),
            index: index.into_boxed(),
        }
    }

    pub fn assign_variable(name: Identifier, value: AST) -> Self {
        Self::AssignVariable {
            name,
            value: value.into_boxed(),
        }
    }

    pub fn assign_field(object: AST, field: Identifier, value: AST) -> Self {
        Self::AssignField {
            object: object.into_boxed(),
            field,
            value: value.into_boxed(),
        }
    }

    pub fn assign_array(array: AST, index: AST, value: AST) -> Self {
        Self::AssignArray {
            array: array.into_boxed(),
            index: index.into_boxed(),
            value: value.into_boxed(),
        }
    }

    pub fn function(name: Identifier, parameters: Vec<Identifier>, body: AST) -> Self {
        Self::Function {
            name,
            parameters,
            body: body.into_boxed(),
        }
    }

    pub fn operator(operator: Operator, parameters: Vec<Identifier>, body: AST) -> Self {
        Self::Function {
            name: Identifier::from(operator),
            parameters,
            body: body.into_boxed(),
        }
    }

    pub fn call_function(name: Identifier, arguments: Vec<AST>) -> Self {
        Self::CallFunction {
            name,
            arguments: arguments.into_boxed(),
        }
    }

    pub fn call_method(object: AST, name: Identifier, arguments: Vec<AST>) -> Self {
        Self::CallMethod {
            object: object.into_boxed(),
            name,
            arguments: arguments.into_boxed(),
        }
    }

    pub fn call_operator(object: AST, operator: Operator, arguments: Vec<AST>) -> Self {
        Self::CallMethod {
            object: object.into_boxed(),
            name: Identifier::from(operator),
            arguments: arguments.into_boxed(),
        }
    }

    pub fn operation(operator: Operator, left: AST, right: AST) -> Self {
        Self::CallMethod {
            object: left.into_boxed(),
            name: Identifier::from(operator),
            arguments: vec![right.into_boxed()],
        }
    }

    pub fn top(statements: Vec<AST>) -> Self {
        Self::Top(statements.into_boxed())
    }

    pub fn block(statements: Vec<AST>) -> Self {
        Self::Block(statements.into_boxed())
    }

    pub fn loop_de_loop(condition: AST, body: AST) -> Self {
        Self::Loop {
            condition: condition.into_boxed(),
            body: body.into_boxed(),
        }
    }

    pub fn conditional(condition: AST, consequent: AST, alternative: AST) -> Self {
        Self::Conditional {
            condition: condition.into_boxed(),
            consequent: consequent.into_boxed(),
            alternative: alternative.into_boxed(),
        }
    }

    pub fn print(format: String, arguments: Vec<AST>) -> Self {
        Self::Print {
            format,
            arguments: arguments.into_boxed(),
        }
    }

    pub fn from_binary_expression(first_operand: AST, other_operators_and_operands: Vec<(Operator, AST)>) -> Self {
        other_operators_and_operands
            .into_iter()
            .fold(first_operand, |left, (operator, right)| {
                AST::operation(operator, left, right)
            })
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Serialize, Deserialize)]
pub struct Identifier(pub String);

impl From<Operator> for Identifier {
    fn from(op: Operator) -> Self {
        Identifier(op.to_string())
    }
}

impl From<&str> for Identifier {
    fn from(s: &str) -> Self {
        Identifier(s.to_owned())
    }
}

impl From<String> for Identifier {
    fn from(s: String) -> Self {
        Identifier(s)
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Identifier {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Operator {
    Multiplication,
    Division,
    Module,
    Addition,
    Subtraction,
    Inequality,
    Equality,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Disjunction,
    Conjunction,
}

impl Operator {
    #[rustfmt::skip]
    pub fn as_str(&self) -> &str {
        match self {
            Operator::Multiplication => "*",
            Operator::Division       => "/",
            Operator::Module         => "%",
            Operator::Addition       => "+",
            Operator::Subtraction    => "-",
            Operator::Inequality     => "!=",
            Operator::Equality       => "==",
            Operator::Less           => "<",
            Operator::LessEqual      => "<=",
            Operator::Greater        => ">",
            Operator::GreaterEqual   => ">=",
            Operator::Disjunction    => "|",
            Operator::Conjunction    => "&",
        }
    }
}

impl From<&str> for Operator {
    #[rustfmt::skip]
    fn from(s: &str) -> Self {
        match s {
            "*"  => Operator::Multiplication,
            "/"  => Operator::Division,
            "%"  => Operator::Module,
            "+"  => Operator::Addition,
            "-"  => Operator::Subtraction,
            "!=" => Operator::Inequality,
            "==" => Operator::Equality,
            "<"  => Operator::Less,
            "<=" => Operator::LessEqual,
            ">"  => Operator::Greater,
            ">=" => Operator::GreaterEqual,
            "|"  => Operator::Disjunction,
            "&"  => Operator::Conjunction,

            other => panic!("Cannot parse {other} as Operator"),
        }
    }
}

impl From<String> for Operator {
    fn from(s: String) -> Self {
        Operator::from(s.as_str())
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub trait IntoBoxed {
    type Into;
    fn into_boxed(self) -> Self::Into;
}

impl IntoBoxed for AST {
    type Into = Box<Self>;
    fn into_boxed(self) -> Self::Into {
        Box::new(self)
    }
}

impl IntoBoxed for Vec<AST> {
    type Into = Vec<Box<AST>>;
    fn into_boxed(self) -> Self::Into {
        self.into_iter().map(|ast| ast.into_boxed()).collect()
    }
}

impl IntoBoxed for Option<AST> {
    type Into = Option<Box<AST>>;
    fn into_boxed(self) -> Self::Into {
        self.map(|ast| ast.into_boxed())
    }
}
