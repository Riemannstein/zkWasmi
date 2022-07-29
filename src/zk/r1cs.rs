use nalgebra::DMatrix;
use crate::engine::bytecode::Instruction;
use crate::exec_context::ExecutionContext;
use crate::AsContextMut;

struct R1CS<T>{
  variable_count : u64,
  A : Option<DMatrix<T>>,
  B : Option<DMatrix<T>>,
  C:  Option<DMatrix<T>>
}

impl<T> R1CS<T>
{
  const fn new() -> Self{
    R1CS::<T>{ 
      variable_count: 1,
      A : None,
      B : None,
      C : None
    }
  }

  fn process_instruction(instr : &Instruction, exec_ctx : &mut ExecutionContext<&mut impl AsContextMut>){
    
  }
}

static R1CS_INSTANCE : R1CS<u64> = R1CS::new();