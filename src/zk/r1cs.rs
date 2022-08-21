use nalgebra::DMatrix;
use crate::engine::bytecode::Instruction;
use crate::engine::exec_context::ExecutionContext;
use crate::AsContextMut;
use std::collections::BTreeMap;
use ark_bls12_381::Fq;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::MVPolynomial;
use ark_ff::fields::Field;

type GlobalStepCount = u64;
type VariableIndex = usize;
pub type VariableValue = Fq;
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct R1CS<T : Field>{
  pub variable_count : u64,
  pub global_step_count : u64,
  pub A : DMatrix<T>,
  pub B : DMatrix<T>,
  pub C:  DMatrix<T>,
  pub pc : TraceVariable<T>,
  pub variables : Vec<TraceVariable<T>>,
  pub inputs : Vec<TraceVariable<T>>,
  // pub trace : BTreeMap<VariableIndex, TraceVariable>,
  pub constraints : Vec<SparsePolynomial<VariableValue,SparseTerm>>
}

#[derive(Debug)]
pub enum TraceVariableKind
{
  PC,
}

#[derive(Debug)]
pub struct TraceVariable<T: Field>{
  pub kind : TraceVariableKind,
  pub global_step_count : GlobalStepCount,
  pub index : VariableIndex,
  pub value : T,
}

impl<T: Field> TraceVariable<T> {
  pub fn new(kind : TraceVariableKind, global_step_count : GlobalStepCount, index : Option<VariableIndex>, value : T) -> Self{
    let mut some_index:VariableIndex = VariableIndex::default();
    match &kind {
      TraceVariableKind::PC => some_index = 1,
      _ => some_index = index.unwrap()
    }
    TraceVariable { kind : kind, global_step_count: global_step_count, index: some_index, value: value }
  }
}

impl<T : Field> R1CS<T>
{
  pub fn new() -> Self{
    // let mut trace: BTreeMap<VariableIndex, TraceVariable> = BTreeMap::new();
    // insert global variable counter
    // trace.insert(0, TraceVariable{global_step_count : 0, index:0, value: VariableValue::default()});
    // Initialize the proram counter
    // [1, pc,]
    let A : DMatrix<T> = DMatrix::from_vec(1, 2, vec![T::zero(), T::one()]);
    let B : DMatrix<T> = DMatrix::from_vec(1, 2, vec![T::one(), T::zero()]);
    let C : DMatrix<T> = DMatrix::from_vec(1, 2, vec![T::zero(), T::zero()]);

    //


    R1CS::<T>{
      variable_count: 1,
      global_step_count: 0,
      A : A,
      B : B,
      C : C,
      pc : TraceVariable { kind: TraceVariableKind::PC, global_step_count: 0, index: 1, value: T::zero()},
      variables : vec![],
      inputs : vec![],
      // trace : trace,
      constraints: vec![]
    }
  }

  // pub fn process_instruction(&mut self, instr : &Instruction, exec_ctx : &mut ExecutionContext<&mut impl AsContextMut>){
  //   use Instruction as Instr;

  //   match instr{

  //     Instr::Br(target) => {

  //     }
  //     Instr::BrIfEqz(target) => {

  //     }
  //     Instr::BrIfNez(target)=> {

  //     }

  //     _ => {}
  //   }
  // }

  pub fn compile(&mut self){
    // Compile to constraints to matrix form
    for constraint in &self.constraints{
      for term in constraint.terms(){

      }
    }
  }
}