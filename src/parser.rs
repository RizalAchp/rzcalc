use core::fmt;

use crate::{
    lexer::{Lexer, Op, Span, Token, TokenKind},
    CalcCtx, Value,
};

#[derive(Debug)]
pub enum Err<'s> {
    UnknwonToken(Token<'s>),
    UnexpectedToken(Token<'s>),

    InvalidNumber(Span, &'s str, String),
    UndefinedVariable(&'s str),

    Unary(Op, Expr<'s>),
    Binary(Op, Expr<'s>, Expr<'s>),

    ExpectedFunction(Value<'s>),
}

type Result<'s, T = Expr<'s>> = ::std::result::Result<T, Err<'s>>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Expr<'s> {
    Im(Value<'s>),
    Var(&'s str),
    MonOp(Op, Box<Expr<'s>>),
    BinOp(Op, Box<Expr<'s>>, Box<Expr<'s>>),
    ApplyFunction(Box<Expr<'s>>, Vec<Expr<'s>>),
}

impl<'s> Expr<'s> {
    #[inline]
    fn expect_token<D>(lex: &mut Lexer<'s>, token: TokenKind<'s>, default: D) -> Result<'s, D> {
        if let Some(peeked) = lex.get_peek() {
            if peeked.kind != token {
                return Err(Err::UnexpectedToken(*peeked));
            }
            lex.set_next();
        }
        Ok(default)
    }
    fn parse_list(lex: &mut Lexer<'s>, closing: TokenKind<'s>) -> Result<'s, Vec<Self>> {
        let mut args = vec![];
        while lex.peek_tok() != closing {
            args.push(Self::parse_binop(lex, 0)?);
            if lex.peek_tok() != TokenKind::Comma {
                break;
            }
            lex.set_next();
        }
        Self::expect_token(lex, closing, args)
    }

    pub fn parse_number(num: &'s str, span: Span) -> Result<'s, Expr<'s>> {
        let ferr = match num.parse::<f64>() {
            Ok(n) => return Ok(Expr::Im(Value::Real(n))),
            Err(ferr) => ferr,
        };
        match num.get(..2) {
            Some("0x") => {
                return i64::from_str_radix(&num[2..], 16)
                    .map_err(|err| Err::InvalidNumber(span, num, format!("{err}")))
                    .map(Value::Int)
                    .map(Expr::Im)
            }
            Some("0o") => {
                return i64::from_str_radix(&num[2..], 8)
                    .map_err(|err| Err::InvalidNumber(span, num, format!("{err}")))
                    .map(Value::Int)
                    .map(Expr::Im)
            }
            Some("0b") => {
                return i64::from_str_radix(&num[2..], 2)
                    .map_err(|err| Err::InvalidNumber(span, num, format!("{err}")))
                    .map(Value::Int)
                    .map(Expr::Im)
            }
            _ => {}
        }
        match num.parse::<i64>() {
            Ok(ok) => Ok(Expr::Im(Value::Int(ok))),
            Err(err) => {
                return Err(Err::InvalidNumber(
                    span,
                    num,
                    format!("(Error: {ferr} - {err})"),
                ));
            }
        }
    }

    fn parse_primitive(lex: &mut Lexer<'s>) -> Result<'s> {
        match lex.next_tok() {
            TokenKind::False => Ok(Self::Im(Value::Bool(false))),
            TokenKind::True => Ok(Self::Im(Value::Bool(true))),
            TokenKind::Ident(s) => Ok(Self::Var(s)),
            TokenKind::Num(n) => Self::parse_number(n, *lex.span()),
            TokenKind::OpenParen => {
                let expr = Self::parse_binop(lex, 0)?;
                Self::expect_token(lex, TokenKind::CloseParen, expr)
            }
            _ => Err(Err::UnexpectedToken(*lex.get_prev())),
        }
    }
    fn parse_apply(lex: &mut Lexer<'s>) -> Result<'s> {
        let mut out = Self::parse_primitive(lex)?;
        while let TokenKind::OpenParen = lex.peek_tok() {
            lex.set_next();
            let args = Self::parse_list(lex, TokenKind::CloseParen)?;
            out = Self::ApplyFunction(Box::new(out), args)
        }
        Ok(out)
    }
    fn parse_monop(lex: &mut Lexer<'s>) -> Result<'s> {
        match lex.peek_tok() {
            TokenKind::Op(op) if matches!(op, Op::Add | Op::Sub | Op::Not) => {
                lex.set_next();
                let arg = Self::parse_monop(lex)?;
                Ok(Self::MonOp(op, Box::new(arg)))
            }
            _ => Self::parse_apply(lex),
        }
    }
    fn parse_binop(lex: &mut Lexer<'s>, prec: u8) -> Result<'s> {
        let mut lhs = Self::parse_monop(lex)?;
        loop {
            match lex.peek_tok() {
                TokenKind::Op(op) if prec <= op.prec() => {
                    lex.set_next();
                    let rhs = Self::parse_binop(lex, op.prec() + 1)?;
                    lhs = Self::BinOp(op, Box::new(lhs), Box::new(rhs));
                }
                _ => break Ok(lhs),
            }
        }
    }
    pub fn parse(input: &'s str) -> Result<'s> {
        let mut lex = Lexer::new(input);
        match (Self::parse_binop(&mut lex, 0)?, lex.get_next()) {
            (_, Some(t)) => Err(Err::UnexpectedToken(*t)),
            (out, _) => Ok(out),
        }
    }
}

impl<'s> Expr<'s> {
    pub fn eval(&self, ctx: &mut CalcCtx<'s>) -> Result<'s, Value<'s>> {
        match self {
            Self::Im(v) => Ok(*v),
            Self::Var(var) => match ctx.get_var(var) {
                Some(v) => Ok(v),
                None => Err(Err::UndefinedVariable(var)),
            },
            Expr::MonOp(op, expr) => Self::eval_monop(ctx, *op, expr),
            Expr::BinOp(op, lhs, rhs) => Self::eval_binop(ctx, *op, lhs, rhs),
            Expr::ApplyFunction(n, args) => Self::eval_apply(ctx, n, args),
        }
    }

    fn eval_apply(ctx: &mut CalcCtx<'s>, func: &Self, args: &[Self]) -> Result<'s, Value<'s>> {
        let args_values = args
            .iter()
            .map(|x| x.eval(ctx))
            .collect::<Result<Vec<_>>>()?;
        match func.eval(ctx)? {
            Value::Function(funcions) => (funcions)(ctx, args_values),
            v => Err(Err::ExpectedFunction(v)),
        }
    }

    #[inline]
    fn eval_binop(ctx: &mut CalcCtx<'s>, op: Op, lhs: &Self, rhs: &Self) -> Result<'s, Value<'s>> {
        use std::cmp::Ordering::Equal;
        use Value::Bool as B;
        use Value::Int as I;
        use Value::Real as R;

        #[rustfmt::skip]
        let out = match (op, lhs.eval(ctx)?, rhs.eval(ctx)?) {
            (Op::Mul,     R(x), R(y)) => R(x * y),
            (Op::Mul,     I(x), I(y)) => I(x * y),
            (Op::Mul,     I(x), R(y)) => R((x as f64) * y),
            (Op::Mul,     R(x), I(y)) => R(x * (y as f64)),

            (Op::Div,     R(x), R(y)) => R(x / y),
            (Op::Div,     I(x), I(y)) => I(x / y),
            (Op::Div,     I(x), R(y)) => R((x as f64) / y),
            (Op::Div,     R(x), I(y)) => R(x / (y as f64)),

            (Op::Rem,     R(x), R(y)) => R(x % y),
            (Op::Rem,     I(x), I(y)) => I(x %  y),
            (Op::Rem,     I(x), R(y)) => R((x as f64) % y),
            (Op::Rem,     R(x), I(y)) => R(x % (y as f64)),

            (Op::Add,     R(x), R(y)) => R(x + y),
            (Op::Add,     I(x), I(y)) => I(x + y),
            (Op::Add,     I(x), R(y)) => R((x as f64) + y),
            (Op::Add,     R(x), I(y)) => R(x + (y as f64)),

            (Op::Sub,     R(x), R(y)) => R(x - y),
            (Op::Sub,     I(x), I(y)) => I(x - y),
            (Op::Sub,     I(x), R(y)) => R((x as f64) - y),
            (Op::Sub,     R(x), I(y)) => R(x - (y as f64)),

            (Op::Shr,     R(x), R(y)) => I((x as i64) >> (y as i64)),
            (Op::Shr,     I(x), I(y)) => I(x >> y),
            (Op::Shr,     I(x), R(y)) => I(x >> (y as i64)),
            (Op::Shr,     R(x), I(y)) => I((x as i64) >> y),

            (Op::Shl,     R(x), R(y)) => I((x as i64) << (y as i64)),
            (Op::Shl,     I(x), I(y)) => I(x << y),
            (Op::Shl,     I(x), R(y)) => I(x << (y as i64)),
            (Op::Shl,     R(x), I(y)) => I((x as i64) << y),

            (Op::BitOr,   R(x), R(y)) => I((x as i64) | (y as i64)),
            (Op::BitOr,   I(x), I(y)) => I(x | y),
            (Op::BitOr,   I(x), R(y)) => I(x | (y as i64)),
            (Op::BitOr,   R(x), I(y)) => I((x as i64) | y),

            (Op::BitAnd,  R(x), R(y)) => I((x as i64) & (y as i64)),
            (Op::BitAnd,  I(x), I(y)) => I(x & y),
            (Op::BitAnd,  I(x), R(y)) => I(x & (y as i64)),
            (Op::BitAnd,  R(x), I(y)) => I((x as i64) & y),

            (Op::BitXor,  R(x), R(y)) => I((x as i64) ^ (y as i64)),
            (Op::BitXor,  I(x), I(y)) => I(x ^ y),
            (Op::BitXor,  I(x), R(y)) => I(x ^ (y as i64)),
            (Op::BitXor,  R(x), I(y)) => I((x as i64) ^ y),

            (Op::BitOr,   B(x), B(y)) => B(x | y),
            (Op::BitAnd,  B(x), B(y)) => B(x & y),
            (Op::BitXor,  B(x), B(y)) => B(x ^ y),

            (Op::Eq,      l,    r)    => B(l == r),
            (Op::Neq,     l,    r)    => B(l != r),
            (Op::And,     l,    r)    => B(l.as_bool() && r.as_bool()),
            (Op::Or,      l,    r)    => B(l.as_bool() || r.as_bool()),
            (Op::Lt,      l,    r)    => B(l.partial_cmp(&r).unwrap_or(Equal).is_lt()),
            (Op::Gt,      l,    r)    => B(l.partial_cmp(&r).unwrap_or(Equal).is_gt()),
            (Op::Lte,     l,    r)    => B(l.partial_cmp(&r).unwrap_or(Equal).is_le()),
            (Op::Gte,     l,    r)    => B(l.partial_cmp(&r).unwrap_or(Equal).is_ge()),

            _ => return Err(Err::Binary(op, lhs.clone(), rhs.clone())),
        };

        Ok(out)
    }

    #[inline]
    fn eval_monop(ctx: &mut CalcCtx<'s>, op: Op, expr: &Self) -> Result<'s, Value<'s>> {
        use Value::Bool as B;
        use Value::Int as I;
        use Value::Real as R;

        let expr_value = expr.eval(ctx)?;
        let out = match (op, expr_value) {
            (Op::Add, B(x)) => B(x),
            (Op::Not, B(x)) => B(!x),
            (Op::Add, I(x)) => I(x),
            (Op::Sub, I(x)) => I(-x),
            (Op::Not, I(x)) => I(!x),
            (Op::Add, R(x)) => R(x),
            (Op::Sub, R(x)) => R(-x),
            (Op::Not, x) => B(!x.as_bool()),
            _ => return Err(Err::Unary(op, expr.clone())),
        };

        Ok(out)
    }
}

impl<'s> fmt::Display for Err<'s> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Err::UnknwonToken(tok) => write!(
                f,
                "Unknown Token: {} - ({}:{})",
                tok.kind, tok.span.begin, tok.span.end
            ),
            Err::UnexpectedToken(tok) => write!(
                f,
                "Unexpected Token: {} - ({}:{})",
                tok.kind, tok.span.begin, tok.span.end
            ),
            Err::InvalidNumber(span, n, err) => write!(
                f,
                "Invalid Number: `{n}` - {err} - ({}:{})",
                span.begin, span.end
            ),
            Err::UndefinedVariable(v) => write!(f, "Undefined Variable of: '{v}'"),
            Err::Unary(op, expr) => write!(f, "Invalid Unary operator: '{op}' on '{expr}'"),
            Err::Binary(op, rhs, lhs) => {
                write!(f, "Invalid Binary operator: '{op}' on '{lhs}' and '{rhs}' ")
            }
            Err::ExpectedFunction(val) => write!(f, "Expected function but got: {val}"),
        }
    }
}

impl fmt::Display for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Im(im) => write!(f, "(Immidiate: {im})"),
            Expr::Var(v) => write!(f, "(Var: {v})"),
            Expr::MonOp(op, expr) => write!(f, "(Monop: {op} {expr})"),
            Expr::BinOp(op, lhs, rhs) => write!(f, "(Binop: {lhs} {op} {rhs})"),
            Expr::ApplyFunction(name, args) => write!(f, "(function: {name} ({args:?}))"),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{CalcCtx, Expr, Value};

    use std::sync::Arc;
    use std::sync::Mutex;

    thread_local!(static GLOBAL_CONTEXT: Arc<Mutex<CalcCtx<'static>>> = Default::default());

    macro_rules! check_eval {
        ($input:expr, $expected:expr) => {{
            let root = dbg!(Expr::parse($input));
            let ctx_lock = GLOBAL_CONTEXT.with(|m| m.clone());
            let mut ctx = ctx_lock.lock().unwrap();
            match root.and_then(|x| x.eval(&mut ctx)) {
                Ok(ok) => assert_eq!(
                    ok, $expected,
                    "<expr: '{}'> is not equal to {}",
                    $input, $expected
                ),
                Err(err) => {
                    panic!("{err}")
                }
            }
        }};
    }

    #[test]
    fn test_operator() {
        check_eval!("1 + 2 * 3", Value::Real(7.));
        check_eval!("1 * 2 + 3", Value::Real(5.));

        check_eval!("2 * 3", Value::Real(6.));
        check_eval!("4 / 2", Value::Real(2.));
        check_eval!("1 + 2", Value::Real(3.));
        check_eval!("2 - 1", Value::Real(1.));
        check_eval!("5 % 2", Value::Real(1.));
        check_eval!("1 << 10", Value::Int(1 << 10));
        check_eval!("10 >> 2", Value::Int(10 >> 2));
        check_eval!("10 | 2", Value::Int(10 | 2));
        check_eval!("10 & 2", Value::Int(10 & 2));
        check_eval!("10 ^ 2", Value::Int(10 ^ 2));

        check_eval!("1 and 0", Value::Bool(false));
        check_eval!("0 and 2", Value::Bool(false));
        check_eval!("3 or 0", Value::Bool(true));
        check_eval!("0 or 4", Value::Bool(true));

        check_eval!("1 and 2 or 3 and 4", Value::Bool(true));
        check_eval!("0 and 2 or 3 and 4", Value::Bool(true));
        check_eval!("0 and 0 or 3 and 4", Value::Bool(true));
        check_eval!("0 and 0 or 0 and 4", Value::Bool(false));
        check_eval!("0 and 0 or 0 and 0", Value::Bool(false));

        check_eval!("1 or 2 and 3 or 4", Value::Bool(true));
        check_eval!("0 or 2 and 3 or 4", Value::Bool(true));
        check_eval!("0 or 0 and 3 or 4", Value::Bool(true));
        check_eval!("0 or 0 and 0 or 4", Value::Bool(true));
        check_eval!("0 or 0 and 0 or 0", Value::Bool(false));
    }

    #[test]
    fn test_operator_extra() {
        check_eval!("(5 << 2) % (2 * 6)", Value::Real(8.));
        check_eval!("2 * 3.5", Value::Real(7.0));
        check_eval!("4.8 / 2.4", Value::Real(2.0));
        check_eval!("0x10 + 0x20", Value::Int(48));
        check_eval!("0b1010 - 0b1101", Value::Int(-3));
        check_eval!("(0x1A * 0b11) % 7", Value::Real(1.));
        check_eval!("1.5 + 0x1A - 0b1010", Value::Real(17.5));

        const EXPECT: f64 = ((0x20 + 0b1101) as f64 * 2.5) - 1.25;
        check_eval!("(0x20 + 0b1101) * 2.5 - 1.25", Value::Real(EXPECT));
        // check_eval!("pi * 2", Value::Real(std::f64::consts::TAU));
    }

    #[test]
    fn test_cmp() {
        check_eval!("1 == 1", Value::Bool(true));
        check_eval!("1 != 1", Value::Bool(false));
        check_eval!("1 <= 1", Value::Bool(true));
        check_eval!("1 >= 1", Value::Bool(true));
        check_eval!("1 <= 1", Value::Bool(true));
        check_eval!("1 >= 1", Value::Bool(true));

        check_eval!("1 == 2", Value::Bool(false));
        check_eval!("1 != 2", Value::Bool(true));
        check_eval!("1 <= 2", Value::Bool(true));
        check_eval!("1 >= 2", Value::Bool(false));
        check_eval!("1 <= 2", Value::Bool(true));
        check_eval!("1 >= 2", Value::Bool(false));
        check_eval!("1 == 1 && 2 == 2", Value::Bool(true));
    }

    #[test]
    fn test_complex_expr() {
        check_eval!("((5 + 3) * 2) / (4 - 1)", Value::Real(5.333333333333333));
        check_eval!("((0b1010 + 0x3) * 0b10) - (0x8 / 0b10)", Value::Int(22));
        check_eval!("(0x1A + (0b1101 * 0b10)) % (0x11 - 0b100)", Value::Int(0));
        check_eval!("(3.5 * 0b10) + (0x1A / 0b11) - 0x8", Value::Real(7.0));
        check_eval!("(0x1A - (0b1101 * 0x2)) * (0b10 / 0x2)", Value::Int(0));
        check_eval!("((0b1010 + 0b10) * 0x2) / (0x8 - 0b10)", Value::Int(4));
        check_eval!(
            "((5 + 3) * (0b10 - 0x1)) + (0x1A / 0b11)",
            Value::Real(16.0)
        );
    }
}
