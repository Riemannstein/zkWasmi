use nalgebra::DMatrix;
use crate::engine::bytecode::Instruction;
use crate::engine::exec_context::ExecutionContext;
use crate::AsContextMut;

#[allow(non_snake_case)]
pub struct R1CS<T>{
  variable_count : u64,
  A : Option<DMatrix<T>>,
  B : Option<DMatrix<T>>,
  C:  Option<DMatrix<T>>
}

impl<T> R1CS<T>
{
  pub fn new() -> Self{
    R1CS::<T>{ 
      variable_count: 1,
      A : None,
      B : None,
      C : None
    }
  }

  pub fn process_instruction(&mut self, instr : &Instruction, exec_ctx : &mut ExecutionContext<&mut impl AsContextMut>){
    
  }
}