#[macro_use]
extern crate lalrpop_util;

// Load module synthesized by LALRPOP
// The generated code triggers some clippy lints so ignore those: https://github.com/lalrpop/lalrpop/issues/614
lalrpop_mod!(#[allow(clippy::all)] pub fml);

mod bytecode;
mod parser;

#[cfg(test)]
#[rustfmt::skip] // LATER(martin-t) Would be nice to format but it's too much code that needs manual fixes
mod tests;

use std::fmt::{self, Debug, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use clap::{Args, Parser};

use crate::bytecode::interpreter::evaluate_with_memory_config;
use crate::bytecode::program::Program;
use crate::bytecode::serializable::Serializable;
use crate::fml::TopLevelParser;
use crate::parser::AST;

#[derive(Parser, Debug)]
#[clap(version, author)]
enum Action {
    Run(RunAction),
    Disassemble(BytecodeDisassemblerAction),
    Execute(BytecodeInterpreterAction),
    Compile(CompilerAction),
    Parse(ParserAction),
}

impl Action {
    pub fn execute(&self) {
        match self {
            Self::Run(action) => action.run(),
            Self::Disassemble(action) => action.debug(),
            Self::Execute(action) => action.interpret(),
            Self::Compile(action) => action.compile(),
            Self::Parse(action) => action.parse(),
        }
    }
}

#[derive(Args, Debug)]
#[clap(about = "Run an FML program")]
struct RunAction {
    #[clap(name = "FILE")]
    pub input: Option<PathBuf>,
    #[clap(
        long = "heap-size",
        name = "BYTES",
        help = "Heap size to trigger GC in bytes (supports k/M/G as suffix)"
    )]
    pub heap_gc_size: Option<String>,
    #[clap(
        long = "heap-log",
        name = "LOG_FILE",
        help = "Path to heap log, if none, the log is not produced"
    )]
    pub heap_log: Option<PathBuf>,
}

#[derive(Args, Debug)]
#[clap(about = "Print FML bytecode in human-readable form")]
struct BytecodeDisassemblerAction {
    #[clap(name = "FILE")]
    pub input: Option<PathBuf>,
}

#[derive(Args, Debug)]
#[clap(about = "Interpret FML bytecode")]
struct BytecodeInterpreterAction {
    #[clap(name = "FILE")]
    pub input: Option<PathBuf>,
    #[clap(
        long = "heap-size",
        name = "BYTES",
        help = "Heap size to trigger GC in bytes (supports k/M/G as suffix)"
    )]
    // LATER(martin-t) Would be nice to parse it using clap and use Option<usize>
    pub heap_gc_size: Option<String>,
    #[clap(
        long = "heap-log",
        name = "LOG_FILE",
        help = "Path to heap log, if none, the log is not produced"
    )]
    pub heap_log: Option<PathBuf>,
}

#[derive(Args, Debug)]
#[clap(about = "Compiles an FML AST into bytecode")]
struct CompilerAction {
    #[clap(short = 'o', long = "output-path", alias = "output-dir")]
    pub output: Option<PathBuf>,

    #[clap(name = "FILE")]
    pub input: Option<PathBuf>,

    #[clap(
        long = "output-format",
        alias = "bc",
        name = "AST_FORMAT",
        help = "The output format for the bytecode: bytes or string"
    )]
    pub output_format: Option<BCSerializer>,

    #[clap(
        long = "input-format",
        alias = "ast",
        name = "BC_FORMAT",
        help = "The output format of the AST: JSON, LISP, YAML"
    )]
    pub input_format: Option<ASTSerializer>,
}

#[derive(Args, Debug)]
#[clap(about = "Parses FML source code and outputs an AST")]
struct ParserAction {
    #[clap(short = 'o', long = "output-path", alias = "output-dir")]
    pub output: Option<PathBuf>,

    #[clap(name = "FILE")]
    pub input: Option<PathBuf>,

    #[clap(
        long = "format",
        alias = "ast",
        name = "FORMAT",
        help = "The output format of the AST: JSON, LISP, YAML, or Rust"
    )]
    pub format: Option<ASTSerializer>,
}

macro_rules! prepare_file_path_from_input_and_serializer {
    ($self:expr) => {
        $self.output.as_ref().map(|path| {
            if path.is_dir() {
                let mut file = path.clone();
                let filename = match $self.selected_input().unwrap().name {
                    Stream::File(file) => PathBuf::from(file).file_name().unwrap().to_str().unwrap().to_owned(),
                    Stream::Console => "ast".to_owned(),
                };
                let extension = $self.selected_output_format().extension();
                file.push(filename);
                file.set_extension(extension);
                file
            } else {
                path.clone()
            }
        })
    };
}

fn parse_size(mut size: &str) -> Result<usize> {
    let multiplier = if size.ends_with('G') {
        1024 * 1024 * 1024
    } else if size.ends_with('M') {
        1024 * 1024
    } else if size.ends_with('k') {
        1024
    } else {
        1
    };
    if multiplier > 1 {
        size = &size[..size.len() - 1];
    }
    let size = size.parse::<usize>()?;
    Ok(size * multiplier)
}

impl RunAction {
    pub fn run(&self) {
        let source = self.selected_input().expect("Cannot open FML program.");

        let ast: AST = TopLevelParser::new()
            .parse(&source.into_string().expect("Error reading input"))
            .expect("Parse error");

        let program = bytecode::compile(&ast).expect("Compiler error");

        let gc_size = self
            .heap_gc_size
            .as_ref()
            .map(|size| parse_size(size).expect("Cannot parse heap size"));

        evaluate_with_memory_config(&program, gc_size, self.heap_log.clone()).expect("Interpreter error");
    }

    pub fn selected_input(&self) -> Result<NamedSource> {
        NamedSource::from(self.input.as_ref())
    }
}

impl BytecodeInterpreterAction {
    pub fn interpret(&self) {
        let mut source = self
            .selected_input()
            .expect("Cannot open an input for the bytecode interpreter.");

        let program = BCSerializer::Bytes
            .deserialize(&mut source)
            .expect("Cannot parse bytecode from input.");

        let gc_size = self
            .heap_gc_size
            .as_ref()
            .map(|size| parse_size(size).expect("Cannot parse heap size"));

        evaluate_with_memory_config(&program, gc_size, self.heap_log.clone()).expect("Interpreter error");
    }

    pub fn selected_input(&self) -> Result<NamedSource> {
        NamedSource::from(self.input.as_ref())
    }
}

impl BytecodeDisassemblerAction {
    pub fn debug(&self) {
        let mut source = self
            .selected_input()
            .expect("Cannot open an input for the bytecode interpreter.");

        let program = BCSerializer::Bytes
            .deserialize(&mut source)
            .expect("Cannot parse bytecode from input.");

        println!("{program}");
    }

    pub fn selected_input(&self) -> Result<NamedSource> {
        NamedSource::from(self.input.as_ref())
    }
}

impl CompilerAction {
    pub fn compile(&self) {
        let source = self.selected_input().expect("Cannot open an input for the compiler.");
        let mut sink = self.selected_output().expect("Cannot open an output for the compiler.");
        let input_serializer = self
            .selected_input_format()
            .expect("Cannot derive input format from file path. Consider setting it explicitly.");
        let output_serializer = self.selected_output_format();

        let source = source.into_string().expect("Error reading input file");
        let ast = input_serializer
            .deserialize(&source)
            .expect("Error parsing AST from input file");

        let program = bytecode::compile(&ast).expect("Compiler Error");

        output_serializer
            .serialize(&program, &mut sink)
            .expect("Cannot serialize program to output.");
    }

    pub fn selected_input(&self) -> Result<NamedSource> {
        NamedSource::from(self.input.as_ref())
    }

    pub fn selected_output_format(&self) -> BCSerializer {
        self.output_format.unwrap_or(BCSerializer::Bytes)
    }

    pub fn selected_input_format(&self) -> Option<ASTSerializer> {
        if self.input_format.is_some() {
            self.input_format
        } else {
            self.selected_input()
                .unwrap()
                .extension()
                .and_then(|s| ASTSerializer::from_extension(s.as_str()))
        }
    }

    pub fn selected_output(&self) -> Result<NamedSink> {
        let maybe_file = prepare_file_path_from_input_and_serializer!(self);
        NamedSink::from(maybe_file)
    }
}

impl ParserAction {
    pub fn parse(&self) {
        let source = self.selected_input().expect("Cannot open an input for the parser.");
        let mut sink = self.selected_output().expect("Cannot open an output for the parser.");
        let serializer = self.selected_output_format();

        let ast: AST = TopLevelParser::new()
            .parse(&source.into_string().expect("Error reading input"))
            .expect("Parse error");

        let result = serializer.serialize(&ast).expect("Cannot serialize AST");

        write!(sink, "{result}").expect("Cannot write to output");
    }

    pub fn selected_input(&self) -> Result<NamedSource> {
        NamedSource::from(self.input.as_ref())
    }

    pub fn selected_output_format(&self) -> ASTSerializer {
        self.format.unwrap_or_else(|| {
            self.output
                .as_ref()
                .and_then(|path| path.extension())
                .and_then(|extension| extension.to_str().map(|s| s.to_owned()))
                .and_then(|extension| ASTSerializer::from_extension(extension.as_str()))
                .unwrap_or(ASTSerializer::Internal)
        })
    }

    pub fn selected_output(&self) -> Result<NamedSink> {
        let maybe_file = prepare_file_path_from_input_and_serializer!(self);
        NamedSink::from(maybe_file)
    }
}

#[derive(Debug, PartialOrd, PartialEq, Ord, Eq, Copy, Clone, Hash)]
enum BCSerializer {
    Bytes,
    String,
}

impl BCSerializer {
    pub fn serialize(&self, program: &Program, sink: &mut NamedSink) -> Result<()> {
        match self {
            BCSerializer::Bytes => program.serialize(sink),
            BCSerializer::String => unimplemented!(),
        }
    }

    pub fn deserialize(&self, source: &mut NamedSource) -> Result<Program> {
        Ok(Program::from_bytes(source))
    }

    pub fn extension(&self) -> &'static str {
        match self {
            BCSerializer::Bytes => "bc",
            BCSerializer::String => "bc.txt",
        }
    }
}

impl std::str::FromStr for BCSerializer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bytes" | "bc" | "bytecode" => Ok(Self::Bytes),
            "string" | "str" | "pp" | "pretty" | "print" => Ok(Self::String),
            format => Err(anyhow::anyhow!("Unknown BC serialization format: {}", format)),
        }
    }
}

#[derive(Debug, PartialOrd, PartialEq, Ord, Eq, Copy, Clone, Hash)]
enum ASTSerializer {
    Lisp,
    Json,
    Yaml,
    Internal,
}
impl ASTSerializer {
    pub fn serialize(&self, ast: &AST) -> Result<String> {
        let string = match self {
            ASTSerializer::Lisp => serde_lexpr::to_string(&ast)?,
            ASTSerializer::Json => serde_json::to_string(&ast)?,
            ASTSerializer::Yaml => serde_yaml::to_string(&ast)?,
            ASTSerializer::Internal => format!("{ast:?}"),
        };
        Ok(format!("{string}\n"))
    }

    pub fn deserialize(&self, source: &str) -> Result<AST> {
        match self {
            ASTSerializer::Lisp => Ok(serde_lexpr::from_str(source)?),
            ASTSerializer::Json => Ok(serde_json::from_str(source)?),
            ASTSerializer::Yaml => Ok(serde_yaml::from_str(source)?),
            ASTSerializer::Internal => {
                bail!("No deserializer implemented for Rust/INTERNAL format")
            }
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            ASTSerializer::Lisp => "lisp",
            ASTSerializer::Json => "json",
            ASTSerializer::Yaml => "yaml",
            ASTSerializer::Internal => "internal",
        }
    }

    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_lowercase().as_str() {
            "lisp" => Some(ASTSerializer::Lisp),
            "json" => Some(ASTSerializer::Json),
            "yaml" => Some(ASTSerializer::Yaml),
            _ => None,
        }
    }
}

impl std::str::FromStr for ASTSerializer {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "lisp" | "sexp" | "sexpr" => Ok(Self::Lisp),
            "yaml" => Ok(Self::Yaml),
            "rust" | "internal" | "debug" => Ok(Self::Internal),
            format => Err(anyhow::anyhow!("Unknown AST serialization format: {}", format)),
        }
    }
}

#[derive(Clone, Hash, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum Stream {
    File(String),
    Console,
}

impl Stream {
    #[allow(dead_code)]
    pub fn from(path: &PathBuf) -> Result<Self> {
        if let Some(str) = path.as_os_str().to_str() {
            Ok(Stream::File(str.to_owned()))
        } else {
            Err(anyhow!("Cannot convert path into UTF string: {:?}", path))
        }
    }
}

struct NamedSource {
    name: Stream,
    source: Box<dyn BufRead>,
}

impl NamedSource {
    fn from(maybe_file: Option<&PathBuf>) -> Result<NamedSource> {
        match maybe_file {
            Some(path) => NamedSource::from_file(path),
            None => NamedSource::console(),
        }
    }

    fn console() -> Result<NamedSource> {
        let named_source = NamedSource {
            name: Stream::Console,
            source: Box::new(BufReader::new(std::io::stdin())),
        };
        Ok(named_source)
    }

    fn from_file(path: &PathBuf) -> Result<Self> {
        if let Some(name) = path.as_os_str().to_str() {
            File::open(path)
                .map(|file| NamedSource {
                    name: Stream::File(name.to_owned()),
                    source: Box::new(BufReader::new(file)),
                })
                .map_err(|error| anyhow!("Cannot open file for reading \"{}\": {}", name, error))
            // LATER(kondziu) maybe directories too?
        } else {
            bail!("Cannot convert path into UTF string: {:?}", path)
        }
    }

    fn into_string(mut self) -> Result<String> {
        let mut string = String::new();
        self.source.read_to_string(&mut string)?;
        Ok(string)
    }

    fn extension(&self) -> Option<String> {
        match &self.name {
            Stream::File(file) => PathBuf::from(file).extension().map(|s| s.to_str().unwrap().to_owned()),
            Stream::Console => None,
        }
    }
}

impl Read for NamedSource {
    fn read(&mut self, buf: &mut [u8]) -> std::result::Result<usize, std::io::Error> {
        self.source.read(buf)
    }
}

impl BufRead for NamedSource {
    fn fill_buf(&mut self) -> std::result::Result<&[u8], std::io::Error> {
        self.source.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.source.consume(amt)
    }
}

impl Debug for NamedSource {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("<")?;
        match &self.name {
            Stream::File(file) => f.write_str(file),
            Stream::Console => f.write_str("stdin"),
        }?;
        Ok(())
    }
}

struct NamedSink {
    name: Stream,
    sink: Box<dyn Write>,
}

impl NamedSink {
    fn from(maybe_file: Option<PathBuf>) -> Result<NamedSink> {
        match maybe_file {
            Some(path) => NamedSink::from_file(&path),
            None => NamedSink::console(),
        }
    }

    fn console() -> Result<Self> {
        let named_sink = NamedSink {
            name: Stream::Console,
            sink: Box::new(std::io::stdout()),
        };
        Ok(named_sink)
    }

    fn from_file(path: &PathBuf) -> Result<Self> {
        if let Some(name) = path.as_os_str().to_str() {
            File::create(path)
                .map(|file| NamedSink {
                    name: Stream::File(name.to_owned()),
                    sink: Box::new(BufWriter::new(file)),
                })
                .map_err(|error| anyhow!("Cannot open file for writing \"{}\": {}", name, error))
        } else {
            bail!("Cannot convert path into UTF string: {:?}", path)
        }
    }

    #[allow(dead_code)]
    fn extension(&self) -> Option<String> {
        match &self.name {
            Stream::File(file) => PathBuf::from(file).extension().map(|s| s.to_str().unwrap().to_owned()),
            Stream::Console => None,
        }
    }
}

impl Write for NamedSink {
    fn write(&mut self, buf: &[u8]) -> std::result::Result<usize, std::io::Error> {
        self.sink.write(buf)
    }

    fn flush(&mut self) -> std::result::Result<(), std::io::Error> {
        self.sink.flush()
    }
}

impl Debug for NamedSink {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(">")?;
        match &self.name {
            Stream::File(file) => f.write_str(file),
            Stream::Console => f.write_str("stout"),
        }?;
        Ok(())
    }
}

fn main() {
    Action::parse().execute();
}
