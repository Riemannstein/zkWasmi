use nalgebra::DMatrix;
use crate::engine::bytecode::Instruction;
use crate::engine::exec_context::ExecutionContext;
use crate::AsContextMut;
use std::collections::BTreeMap;

type GlobalStepCounter = u64;
type VariableIndex = u64;
pub type VariableValue = u64;
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct R1CS<T>{
  pub variable_count : u64,
  pub A : Option<DMatrix<T>>,
  pub B : Option<DMatrix<T>>,
  pub C:  Option<DMatrix<T>>,
  pub trace : BTreeMap<VariableIndex, TraceVariable>
}

#[derive(Debug)]
pub struct TraceVariable{
  pub global_step_counter : GlobalStepCounter,
  pub index : VariableIndex,
  pub value : VariableValue
}

pub struct QuadraticConstraint
{
  pub variables : Vec<TraceVariable>,
  
}

impl<T> R1CS<T>
{
  pub fn new() -> Self{
    let mut trace: BTreeMap<GlobalStepCounter, TraceVariable> = BTreeMap::new();
    // insert global variable counter
    trace.insert(0, TraceVariable{global_step_counter : 0, index:0, value:0});
    R1CS::<T>{
      variable_count: 1,
      A : None,
      B : None,
      C : None,
      trace : trace
    }
  }

  pub fn process_instruction(&mut self, instr : &Instruction, exec_ctx : &mut ExecutionContext<&mut impl AsContextMut>){
    use Instruction as Instr;

    match instr{

      Instr::Br(target) => {

      }
      Instr::BrIfEqz(target) => {

      }
      Instr::BrIfNez(target)=> {

      }

      _ => {}
    }
  }
}