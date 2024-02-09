use core::fmt;
use std::iter::Peekable;

pub struct Lexer<'s> {
    pub input: &'s str,
    pub tokens: Vec<Token<'s>>,
    pub idx: usize,
}

impl<'s> Lexer<'s> {
    pub fn new(input: &'s str) -> Self {
        let mut stream = input.chars().enumerate().peekable();
        let tokens = (0..)
            .map(|_| Self::parse_token(input, &mut stream))
            .take_while(Option::is_some)
            .flatten()
            .collect::<Vec<_>>();

        Self {
            input,
            tokens,
            idx: 0,
        }
    }
}

impl<'s> Lexer<'s> {
    fn parse_token<IT: Iterator<Item = (usize, char)>>(
        input: &'s str,
        stream: &mut Peekable<IT>,
    ) -> Option<Token<'s>> {
        while stream.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let (i, c) = stream.next()?;

        if matches!(c, '.' | '0'..='9') {
            let begin = i;
            let mut end = begin;
            while stream
                .next_if(|(_, c)| {
                    c.is_ascii_digit()
                        || c.is_ascii_hexdigit()
                        || *c == '.'
                        || *c == '_'
                        || *c == 'x'
                        || *c == 'o'
                        || *c == 'b'
                        || *c == 'e'
                })
                .is_some()
            {
                end += 1;
            }
            return Some(Token::num(&input[begin..=end], begin, end));
        } else if c.is_alphabetic() {
            let begin = i;
            let mut end = begin;
            while stream.next_if(|(_, x)| x.is_alphanumeric()).is_some() {
                end += 1;
            }
            let s = &input[begin..=end];
            return Some(match s {
                "true" | "TRUE" | "True" => Token::new(TokenKind::True, begin, end),
                "false" | "FALSE" | "False" => Token::new(TokenKind::False, begin, end),
                "or" => Token::op(Op::Or, begin, end),
                "and" => Token::op(Op::And, begin, end),
                "not" => Token::op(Op::Not, begin, end),
                _ => Token::new(TokenKind::Ident(s), begin, end),
            });
        } else if let Some(&(peek_i, peek_c)) = stream.peek() {
            let mut consume = |tok| {
                stream.next();
                Some(tok)
            };
            match (c, peek_c) {
                ('&', '&') => return consume(Token::op(Op::And, i, peek_i)),
                ('|', '|') => return consume(Token::op(Op::Or, i, peek_i)),
                ('=', '=') => return consume(Token::op(Op::Eq, i, peek_i)),
                ('!', '=') => return consume(Token::op(Op::Neq, i, peek_i)),
                ('<', '=') => return consume(Token::op(Op::Lte, i, peek_i)),
                ('<', '<') => return consume(Token::op(Op::Shl, i, peek_i)),
                ('>', '=') => return consume(Token::op(Op::Gte, i, peek_i)),
                ('>', '>') => return consume(Token::op(Op::Shr, i, peek_i)),
                _ => {}
            }
        }

        Some(match c {
            '(' => Token::new(TokenKind::OpenParen, i, i),
            ')' => Token::new(TokenKind::CloseParen, i, i),
            ',' => Token::new(TokenKind::Comma, i, i),
            '+' => Token::op(Op::Add, i, i),
            '-' => Token::op(Op::Sub, i, i),
            '*' => Token::op(Op::Mul, i, i),
            '/' => Token::op(Op::Div, i, i),
            '%' => Token::op(Op::Rem, i, i),
            '|' => Token::op(Op::BitOr, i, i),
            '&' => Token::op(Op::BitAnd, i, i),
            '^' => Token::op(Op::BitXor, i, i),
            '>' => Token::op(Op::Gt, i, i),
            '<' => Token::op(Op::Lt, i, i),
            '!' => Token::op(Op::Not, i, i),
            _ => Token::new(TokenKind::Unknown(c), i, i),
        })
    }
}

impl<'s> Lexer<'s> {
    #[inline(always)]
    pub fn get_peek(&mut self) -> Option<&Token<'s>> {
        self.tokens.get(self.idx)
    }

    pub fn get_prev(&mut self) -> &Token<'s> {
        self.set_prev();
        &self.tokens[self.idx]
    }

    #[inline(always)]
    pub fn get_next(&mut self) -> Option<&Token<'s>> {
        let tok = self.tokens.get(self.idx);
        self.idx += 1;
        tok
    }

    #[inline(always)]
    pub fn peek_tok(&mut self) -> TokenKind<'s> {
        match self.get_peek() {
            Some(x) => x.kind,
            None => TokenKind::End,
        }
    }
    #[inline(always)]
    pub fn next_tok(&mut self) -> TokenKind<'s> {
        match self.get_next() {
            Some(x) => x.kind,
            None => TokenKind::End,
        }
    }

    #[inline(always)]
    pub fn set_prev(&mut self) {
        if self.idx > 0 {
            self.idx -= 1;
        }
    }
    #[inline(always)]
    pub fn set_next(&mut self) {
        self.idx += 1;
    }

    #[inline(always)]
    pub fn span(&self) -> &Span {
        &self.tokens[self.idx - 1].span
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    Or = 1,
    And,
    BitOr,
    BitXor,
    BitAnd,
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    Neq,
    Shr,
    Shl,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Not,
}
impl Op {
    #[inline]
    pub const fn prec(self) -> u8 {
        match self {
            Op::Or => 1,
            Op::And => 2,
            Op::BitOr => 3,
            Op::BitXor => 4,
            Op::BitAnd => 5,
            Op::Lt | Op::Gt | Op::Lte | Op::Gte => 6,
            Op::Eq | Op::Neq => 7,
            Op::Shr | Op::Shl => 8,
            Op::Add | Op::Sub => 9,
            Op::Mul | Op::Div | Op::Rem => 10,
            Op::Not => 11,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenKind<'s> {
    Op(Op),
    Unknown(char),
    Num(&'s str),
    Ident(&'s str),
    True,
    False,
    Comma,
    OpenParen,
    CloseParen,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    pub begin: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Token<'s> {
    pub span: Span,
    pub kind: TokenKind<'s>,
}

impl<'s> Token<'s> {
    const fn new(kind: TokenKind<'s>, begin: usize, end: usize) -> Self {
        Self {
            span: Span { begin, end },
            kind,
        }
    }
    const fn op(op: Op, begin: usize, end: usize) -> Self {
        Self::new(TokenKind::Op(op), begin, end)
    }
    const fn num(num: &'s str, begin: usize, end: usize) -> Self {
        Self::new(TokenKind::Num(num), begin, end)
    }
}
impl Op {
    pub const fn as_str(self) -> &'static str {
        match self {
            Op::Or => "||",
            Op::And => "&&",
            Op::BitOr => "|",
            Op::BitXor => "^",
            Op::BitAnd => "&",
            Op::Lt => "<",
            Op::Gt => ">",
            Op::Lte => "<=",
            Op::Gte => ">=",
            Op::Eq => "==",
            Op::Neq => "!=",
            Op::Shr => ">>",
            Op::Shl => "<<",
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Rem => "%",
            Op::Not => "!",
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for TokenKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Op(op) => f.write_str(op.as_str()),
            TokenKind::Unknown(c) => write!(f, "{c}"),
            TokenKind::Num(n) => write!(f, "{n}"),
            TokenKind::Ident(s) => write!(f, "ident <{s}>"),
            TokenKind::True => f.write_str("true"),
            TokenKind::False => f.write_str("false"),
            TokenKind::End => f.write_str("end"),
            TokenKind::OpenParen => f.write_str("("),
            TokenKind::CloseParen => f.write_str(")"),
            TokenKind::Comma => f.write_str(","),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lexer, Op, TokenKind};

    macro_rules! test_match_tokens {
        ($input: expr, $tokens: expr) => {{
            let mut lexer = Lexer::new($input);
            for tok in $tokens {
                assert_eq!(lexer.next_tok(), tok);
            }
            assert_eq!(lexer.next_tok(), TokenKind::End);
        }};
    }

    #[test]
    fn test_operators() {
        let string = "+ - * / not and or == != < > <= >= % | ^ & >> <<";
        let tokens = [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Not,
            Op::And,
            Op::Or,
            Op::Eq,
            Op::Neq,
            Op::Lt,
            Op::Gt,
            Op::Lte,
            Op::Gte,
            Op::Rem,
            Op::BitOr,
            Op::BitXor,
            Op::BitAnd,
            Op::Shr,
            Op::Shl,
        ]
        .into_iter()
        .map(TokenKind::Op);

        test_match_tokens!(string, tokens);
    }

    #[test]
    fn test_tokens() {
        let string = "( ) , ?";
        let tokens = [
            TokenKind::OpenParen,
            TokenKind::CloseParen,
            TokenKind::Comma,
            TokenKind::Unknown('?'),
        ];
        test_match_tokens!(string, tokens);
    }

    #[test]
    fn test_idents() {
        let string = "true false or and not foo";
        let tokens = [
            TokenKind::True,
            TokenKind::False,
            TokenKind::Op(Op::Or),
            TokenKind::Op(Op::And),
            TokenKind::Op(Op::Not),
            TokenKind::Ident("foo"),
        ];

        test_match_tokens!(string, tokens);
    }

    #[test]
    fn test_numbers() {
        let string = "1 .2 3. 4.5 0x1A 0b1010 0o12 1e3 -2.5 6.022e23";
        let tokens = [
            TokenKind::Num("1"),
            TokenKind::Num(".2"),
            TokenKind::Num("3."),
            TokenKind::Num("4.5"),
            TokenKind::Num("0x1A"),
            TokenKind::Num("0b1010"),
            TokenKind::Num("0o12"),
            TokenKind::Num("1e3"),
            TokenKind::Op(Op::Sub),
            TokenKind::Num("2.5"),
            TokenKind::Num("6.022e23"),
        ];
        test_match_tokens!(string, tokens);
    }

    #[test]
    fn test_basic() {
        let string = "compare(a, ~)";
        let tokens = [
            TokenKind::Ident("compare"),
            TokenKind::OpenParen,
            TokenKind::Ident("a"),
            TokenKind::Comma,
            TokenKind::Unknown('~'),
            TokenKind::CloseParen,
        ];

        test_match_tokens!(string, tokens);
    }

    #[test]
    fn test_prev_peek_next() {
        let string = "a b c";
        let mut lexer = Lexer::new(string);

        let a = TokenKind::Ident("a");
        let b = TokenKind::Ident("b");
        let c = TokenKind::Ident("c");

        assert_eq!(lexer.peek_tok(), a);
        assert_eq!(lexer.next_tok(), a);
        assert_eq!(lexer.peek_tok(), b);
        lexer.set_prev();
        assert_eq!(lexer.peek_tok(), a);
        assert_eq!(lexer.next_tok(), a);
        assert_eq!(lexer.next_tok(), b);
        assert_eq!(lexer.next_tok(), c);
        lexer.set_prev();
        assert_eq!(lexer.next_tok(), c);
        assert_eq!(lexer.next_tok(), TokenKind::End);
    }
}
