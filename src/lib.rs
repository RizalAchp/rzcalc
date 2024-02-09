mod lexer;
mod parser;

use core::fmt;
use std::{borrow::Cow, collections::HashMap};

pub use parser::{Err as Error, Expr};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Value<'s> {
    Bool(bool),
    Real(f64),
    Int(i64),
    Function(Function<'s>),
}

impl<'s> Value<'s> {
    #[inline]
    fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Real(r) => r.is_normal(),
            Value::Int(n) => *n > 0,
            Value::Function(_) => true,
        }
    }
}

pub type Function<'s> = fn(&mut CalcCtx, Vec<Value<'s>>) -> Result<Value<'s>, Error<'s>>;

#[derive(Default)]
pub struct CalcCtx<'s> {
    inputs: Vec<Cow<'s, str>>,
    scope: HashMap<String, Value<'s>>,
}

impl<'s> CalcCtx<'s> {
    pub fn get_var(&self, var: &str) -> Option<Value<'s>> {
        self.scope.get(var).copied()
    }
    pub fn set_var(&mut self, name: String, value: impl Into<Value<'s>>) {
        self.scope.insert(name, value.into());
    }

    pub fn evaluate_input(&mut self, input: &'s str) -> Result<Value<'s>, Error<'s>> {
        let expr = Expr::parse(input)?;
        self.inputs.push(input.into());
        expr.eval(self)
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{b:?}"),
            Value::Real(n) => write!(f, "{n}"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Function(_) => f.write_str("<function>"),
        }
    }
}

impl From<f64> for Value<'_> {
    fn from(value: f64) -> Self {
        Value::Real(value)
    }
}

impl From<i64> for Value<'_> {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<bool> for Value<'_> {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl<'s> From<Function<'s>> for Value<'s> {
    fn from(value: Function<'s>) -> Self {
        Value::Function(value)
    }
}
